"""Data loading and assembly for the BGBW DML trend pipeline.

Provides functions to load individual-level survey data, grid-aggregated
surveys, and geospatial embeddings, and to assemble them into analysis-ready
DataFrames keyed by (bng_ref, year).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bgbw_trends.aggregate import aggregate_surveys
from osbng.indexing import xy_to_bng
from bgbw_trends.aggregate import BNG_RESOLUTION
from bgbw_trends.config import (
    AEF_DIR,
    BNG_SURVEYS_DIR,
    HARMONIZED_DIR,
    JANUARY_WEATHER_PATH,
    WEATHER_PATH,
    RunConfig,
)

logger = logging.getLogger(__name__)


# ── Raw data loaders ────────────────────────────────────────────────────────


def load_harmonized(year: int, directory: Path = HARMONIZED_DIR) -> pd.DataFrame:
    """Load individual-level harmonized survey data for a single year."""
    path = directory / f"bgbw_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No harmonized data for {year}: {path}")
    df = pd.read_parquet(path)
    logger.info("Loaded harmonized %d: %d surveys", year, len(df))
    return df


def load_bng_surveys(year: int, directory: Path = BNG_SURVEYS_DIR) -> pd.DataFrame:
    """Load grid-aggregated (1 km BNG) survey data for a single year."""
    path = directory / f"bng1km_surveys_{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No BNG survey data for {year}: {path}")
    df = pd.read_parquet(path)
    # Drop the WKB geometry column — we join spatially via bng_ref
    if "geometry" in df.columns:
        df = df.drop(columns=["geometry"])
    logger.info("Loaded BNG surveys %d: %d grid cells", year, len(df))
    return df


# Mapping from parquet stat name to column prefix
_STAT_PREFIXES: dict[str, str] = {
    "mean": "emb",
    "std": "emb_std",
    "median": "emb_med",
    "p10": "emb_p10",
    "p90": "emb_p90",
}


def load_embeddings(
    year: int = 2019,
    directory: Path = AEF_DIR,
    summaries: list[str] | None = None,
) -> pd.DataFrame:
    """Load AEF geospatial embeddings and expand list columns into a matrix.

    Returns a DataFrame indexed by ``bng_ref`` with columns
    ``emb_0..{d-1}`` (mean) plus any requested distributional summaries:
    ``emb_std_0..{d-1}``, ``emb_p10_0..{d-1}``, ``emb_p90_0..{d-1}``
    and ``n_pixels``.

    Parameters
    ----------
    year : int
        Embedding snapshot year.
    directory : Path
        Directory containing AEF parquet files.
    summaries : list[str] or None
        Which distributional summaries to include. Must be a subset of
        {"mean", "std", "median", "p10", "p90"}. Defaults to
        ["mean", "std"] (mean + spread stats).
        "mean" is always included even if absent from the list.
    """
    if summaries is None:
        summaries = ["mean", "std"]
    # Ensure "mean" is always present (provides the emb_0..d-1 columns)
    if "mean" not in summaries:
        summaries = ["mean"] + list(summaries)

    path = directory / f"aef_bng1km_{year}_uk.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No embedding data for {year}: {path}")
    raw = pd.read_parquet(path, columns=["bng_ref", "n_pixels"] + summaries)

    # Keep only cells with actual content (non-zero mean embeddings)
    mean_matrix = np.stack(raw["mean"].values)
    emb_dim = mean_matrix.shape[1]
    has_signal = mean_matrix.any(axis=1)

    parts: list[pd.DataFrame] = [raw[["bng_ref", "n_pixels"]].reset_index(drop=True)]
    for stat in summaries:
        prefix = _STAT_PREFIXES[stat]
        matrix = np.stack(raw[stat].values)
        cols = [f"{prefix}_{i}" for i in range(emb_dim)]
        parts.append(pd.DataFrame(matrix, columns=cols, index=raw.index))

    result = pd.concat(parts, axis=1)
    result = result[has_signal].copy()
    result = result.set_index("bng_ref")
    logger.info(
        "Loaded embeddings %d: %d cells, summaries=%s, dim=%d",
        year,
        len(result),
        summaries,
        emb_dim,
    )
    return result


def load_mean_embeddings(
    years: list[int] | None = None,
    directory: Path = AEF_DIR,
    summaries: list[str] | None = None,
) -> pd.DataFrame:
    """Load AEF embeddings for multiple years and return their cell-wise mean.

    Averaging across years reduces sensitivity to interannual variation while preserving
    the stable spatial signal of landscape character.  The intersection of
    cells present in all requested years is retained (should be full).

    Parameters
    ----------
    years : list[int] or None
        Years to average over.  If None, all ``aef_bng1km_*_uk.parquet``
        files found in *directory* are used.
    directory : Path
        Directory containing AEF parquet files.
    summaries : list[str] or None
        Passed through to :func:`load_embeddings`.
    """
    if years is None:
        years = sorted(
            int(p.stem.split("_")[2]) for p in directory.glob("aef_bng1km_*_uk.parquet")
        )
        if not years:
            raise FileNotFoundError(f"No AEF embedding files found in {directory}")

    frames = [
        load_embeddings(y, directory=directory, summaries=summaries) for y in years
    ]

    # Restrict to cells present in every year
    common_index = frames[0].index
    for f in frames[1:]:
        common_index = common_index.intersection(f.index)
    frames = [f.loc[common_index] for f in frames]

    # Average all numeric columns across years
    avg_cols = [c for c in frames[0].columns]
    result = frames[0].copy()
    for col in avg_cols:
        result[col] = np.mean([f[col].to_numpy() for f in frames], axis=0)

    logger.info(
        "Using mean embeddings over %d years (%s): %d cells "
        "(intersection across all years)",
        len(years),
        ", ".join(str(y) for y in years),
        len(result),
    )
    return result


def load_weather(
    years: list[int],
    path: Path = WEATHER_PATH,
) -> pd.DataFrame:
    """Load HadUK-Grid survey-weekend weather and aggregate to 1 km BNG cells.

    Reads the weather parquet produced by ``sources/weather/fetch_haduk.py``,
    converts OSGB coordinates to BNG refs, then returns one row per
    (bng_ref, year) with two effort-adjacent weather summaries:

    - ``mean_temp_c``  : mean of daily mean temperature across survey days (°C)
    - ``total_rain_mm``: sum of daily rainfall across survey days (mm)

    These are X_effort features (frozen at median for SDM). See DEVELOPER_GUIDE §4.3.

    Years not present in the weather file (e.g. 2025+) are silently skipped;
    their cells will have NaN weather that ``make_effort_features`` fills with
    column medians.
    """
    if not path.exists():
        raise FileNotFoundError(f"Weather parquet not found: {path}")

    df = pd.read_parquet(
        path, columns=["year", "projection_x", "projection_y", "tas_mean", "rainfall"]
    )
    df = df[df["year"].isin(years)]
    if df.empty:
        logger.warning("No weather data found for years %s in %s", years, path)
        return pd.DataFrame(columns=["bng_ref", "year", "mean_temp_c", "total_rain_mm"])

    # Assign bng_ref once per unique grid cell (the weather grid is already 1 km OSGB)
    unique_cells = df[["projection_x", "projection_y"]].drop_duplicates().copy()
    refs = []
    for _, row in unique_cells.iterrows():
        try:
            refs.append(
                xy_to_bng(
                    float(row["projection_x"]),
                    float(row["projection_y"]),
                    resolution=BNG_RESOLUTION,
                ).bng_ref_compact
            )
        except Exception:
            refs.append(None)
    unique_cells["bng_ref"] = refs
    unique_cells = unique_cells.dropna(subset=["bng_ref"])

    df = df.merge(unique_cells, on=["projection_x", "projection_y"], how="inner")

    # Aggregate across survey days per (bng_ref, year)
    agg = (
        df.groupby(["bng_ref", "year"])
        .agg(mean_temp_c=("tas_mean", "mean"), total_rain_mm=("rainfall", "sum"))
        .reset_index()
    )
    logger.info(
        "Loaded weather: %d (bng_ref, year) cells across %d years",
        len(agg),
        agg["year"].nunique(),
    )
    return agg


def load_january_weather(
    years: list[int],
    path: Path = JANUARY_WEATHER_PATH,
) -> pd.DataFrame:
    """Load pre-survey January weather and aggregate to 1 km BNG cells.

    Reads the parquet produced by
    ``sources/weather/summarise_january_weather.py``, converts OSGB coordinates
    to BNG refs, and returns one row per (bng_ref, year) with two
    effort-adjacent summaries covering non-survey January days:

    - ``jan_mean_temp_c``   : mean daily temperature (°C)
    - ``jan_total_rain_mm`` : cumulative rainfall (mm)

    Cold or wet Januaries depress garden bird activity and observer motivation
    independently of conditions on the survey weekend itself.  These are
    X_effort features (frozen at median for SDM, NOT in W).
    See DEVELOPER_GUIDE §4.3.

    Years not present in the parquet (e.g. 2025+) are silently skipped;
    their cells will have NaN that ``make_effort_features`` fills with column
    medians at model-fitting time.
    """
    if not path.exists():
        raise FileNotFoundError(f"January-weather parquet not found: {path}")

    df = pd.read_parquet(
        path,
        columns=[
            "year",
            "projection_x",
            "projection_y",
            "jan_mean_temp",
            "jan_total_rain",
        ],
    )
    df = df[df["year"].isin(years)]
    if df.empty:
        logger.warning("No January-weather data found for years %s in %s", years, path)
        return pd.DataFrame(
            columns=["bng_ref", "year", "jan_mean_temp_c", "jan_total_rain_mm"]
        )

    # Assign bng_ref once per unique grid cell (already 1 km OSGB)
    unique_cells = df[["projection_x", "projection_y"]].drop_duplicates().copy()
    refs = []
    for _, row in unique_cells.iterrows():
        try:
            refs.append(
                xy_to_bng(
                    float(row["projection_x"]),
                    float(row["projection_y"]),
                    resolution=BNG_RESOLUTION,
                ).bng_ref_compact
            )
        except Exception:
            refs.append(None)
    unique_cells["bng_ref"] = refs
    unique_cells = unique_cells.dropna(subset=["bng_ref"])

    df = df.merge(unique_cells, on=["projection_x", "projection_y"], how="inner")

    result = (
        df[["bng_ref", "year", "jan_mean_temp", "jan_total_rain"]]
        .rename(
            columns={
                "jan_mean_temp": "jan_mean_temp_c",
                "jan_total_rain": "jan_total_rain_mm",
            }
        )
        .reset_index(drop=True)
    )
    logger.info(
        "Loaded January weather: %d (bng_ref, year) cells across %d years",
        len(result),
        result["year"].nunique(),
    )
    return result


# ── Assembly ────────────────────────────────────────────────────────────────


def load_multi_year_bng(
    years: list[int], directory: Path = BNG_SURVEYS_DIR
) -> pd.DataFrame:
    """Stack BNG-aggregated surveys across multiple years."""
    frames = []
    for y in years:
        df = load_bng_surveys(y, directory)
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    logger.info("Stacked BNG surveys: %d rows across %d years", len(result), len(years))
    return result


def assemble_analysis_data(cfg: RunConfig) -> pd.DataFrame:
    """Build the main analysis DataFrame.

    Aggregates individual harmonized surveys to 1 km BNG cells with rich
    effort summaries (n_surveys, mean_observers, proportion_adult,
    proportion_online), then joins with geospatial embeddings.

    Each row is a (bng_ref, year) observation.

    Columns produced:
    - year, bng_ref
    - n_surveys, mean_observers, proportion_adult, proportion_online
    - one column per species (mean count across surveys in that cell-year)
    - emb_0 ... emb_{d-1}  (geospatial embedding of the 1 km cell)
    - n_pixels  (number of satellite pixels in the cell)
    - mean_temp_c, total_rain_mm  (survey-weekend weather, X_effort)
    - jan_mean_temp_c, jan_total_rain_mm  (January pre-survey weather, X_effort)

    Parameters
    ----------
    cfg : RunConfig
        Specifies years, species, and embedding years to average.

    Returns
    -------
    pd.DataFrame
        Analysis-ready data, one row per (bng_ref, year).
    """
    # 1. Load and aggregate harmonized surveys to BNG grid
    harmonized = assemble_harmonized_data(cfg)
    surveys = aggregate_surveys(harmonized)

    # 2. Load geospatial embeddings averaged across all available years so that
    #    cell representations are not at the mercy of interannual variation
    #    (e.g. phenology artefacts, cloud-cover differences in any single year).
    embeddings = load_mean_embeddings(cfg.embedding_years)

    # 3. Join embeddings on bng_ref
    merged = surveys.merge(embeddings, left_on="bng_ref", right_index=True, how="inner")

    # 4. Join survey-weekend weather on (bng_ref, year) — left join so cells
    #    without coverage (e.g. years beyond the weather archive) keep NaN,
    #    which make_effort_features fills with column medians.
    weather = load_weather(cfg.years)
    merged = merged.merge(weather, on=["bng_ref", "year"], how="left")

    # 5. Join pre-survey January weather — same left-join convention.
    jan_weather = load_january_weather(cfg.years)
    merged = merged.merge(jan_weather, on=["bng_ref", "year"], how="left")

    logger.info(
        "Assembled analysis data: %d rows (%d cells x %d years, after joins "
        "with embeddings and weather)",
        len(merged),
        merged["bng_ref"].nunique(),
        merged["year"].nunique(),
    )
    return merged


def assemble_harmonized_data(cfg: RunConfig) -> pd.DataFrame:
    """Load and stack individual-level harmonized data across years.

    This is the richer dataset with observation-process covariates (garden
    features, food, weather, etc.) needed for the full DML confounder set.
    Each row is one survey submission.

    Parameters
    ----------
    cfg : RunConfig
        Specifies years.

    Returns
    -------
    pd.DataFrame
        Stacked individual-level data.
    """
    frames = []
    for y in cfg.years:
        df = load_harmonized(y)
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    logger.info(
        "Stacked harmonized data: %d rows across %d years",
        len(result),
        len(cfg.years),
    )
    return result
