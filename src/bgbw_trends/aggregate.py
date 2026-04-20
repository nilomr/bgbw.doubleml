"""Aggregate individual survey data to 1 km BNG grid cells.

For each (bng_ref, year) cell, computes:
  - n_surveys        : number of geocoded surveys
  - mean_observers   : mean total observers (adults + children) per survey
  - proportion_adult : proportion of total observers who are adults
  - proportion_online: proportion of surveys submitted online
  - <species>        : mean count across surveys (one column per species)

Coordinate conversion uses WGS84 -> EPSG:27700 (pyproj) and osbng for
grid references. Adapted from the metadata pipeline's aggregate.py.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from osbng.indexing import xy_to_bng
from pyproj import Transformer

logger = logging.getLogger(__name__)

# ── BNG constants ──────────────────────────────────────────────────────────

BNG_CELL = 1_000  # metres
GRID_COLS = 700
GRID_ROWS = 1_300
BNG_RESOLUTION = "1km"

# Columns that are NOT species counts
_NON_SPECIES_COLS: frozenset[str] = frozenset(
    {
        "year", "datetimestamp", "source", "source_file",
        "garden_or_school", "type",
        "postcode", "place", "area_home", "address1", "address2", "address3",
        "town_city", "address_postcode", "country", "email",
        "adults", "children",
        "garden_size", "nearest_farmland",
        "lawn", "pond", "wildlife_area", "compost_heap", "flower_beds",
        "large_trees", "small_trees", "potted_plants",
        "deciduous_shrubs", "coniferous_shrubs", "climber",
        "birdfoodsupplied", "peanuts", "sunflower_seed", "sunflower_hearts",
        "niger_seed", "seed_mix", "fatballs", "coconut", "suet", "bread",
        "kitchen_scraps", "mealworms", "birdcake", "berry", "other",
        "food_other", "food_none",
        "weather_rain", "weather_wind", "weather_visibility",
        "feedback_comments",
        "latitude", "longitude", "postcode_valid", "country_code",
        "region_code", "urban_rural",
    }
)


def _species_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that are bird species count columns."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in _NON_SPECIES_COLS]


# ── Coordinate helpers ─────────────────────────────────────────────────────

_transformer: Transformer | None = None


def _get_transformer() -> Transformer:
    global _transformer
    if _transformer is None:
        _transformer = Transformer.from_crs(
            "EPSG:4326", "EPSG:27700", always_xy=True
        )
    return _transformer


def _assign_bng_ref(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``bng_ref`` column by converting lat/lon to 1 km BNG grid refs.

    Filters to rows within the valid BNG extent and with resolvable refs.
    """
    t = _get_transformer()
    easting, northing = t.transform(
        df["longitude"].to_numpy(dtype=float),
        df["latitude"].to_numpy(dtype=float),
    )

    in_extent = (
        (easting >= 0)
        & (easting < GRID_COLS * BNG_CELL)
        & (northing >= 0)
        & (northing < GRID_ROWS * BNG_CELL)
    )
    df = df.loc[in_extent].copy()
    easting, northing = easting[in_extent], northing[in_extent]

    # Vectorise bng_ref assignment via unique cell centres
    col_j = (easting // BNG_CELL).astype(int)
    row_i = (northing // BNG_CELL).astype(int)
    df["_col_j"] = col_j
    df["_row_i"] = row_i

    unique = df[["_col_j", "_row_i"]].drop_duplicates()
    east_c = unique["_col_j"].to_numpy() * BNG_CELL + BNG_CELL // 2
    north_c = unique["_row_i"].to_numpy() * BNG_CELL + BNG_CELL // 2

    refs = []
    for e, n in zip(east_c, north_c):
        try:
            refs.append(
                xy_to_bng(float(e), float(n), resolution=BNG_RESOLUTION).bng_ref_compact
            )
        except Exception:
            refs.append(None)
    unique["bng_ref"] = refs
    unique = unique.dropna(subset=["bng_ref"])

    df = df.merge(unique[["_col_j", "_row_i", "bng_ref"]], on=["_col_j", "_row_i"])
    df = df.drop(columns=["_col_j", "_row_i"])
    return df


# ── Aggregation ────────────────────────────────────────────────────────────


def aggregate_surveys(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate individual surveys to 1 km BNG cells.

    Parameters
    ----------
    df : pd.DataFrame
        Individual-level harmonized survey data for one or more years.
        Must contain ``latitude``, ``longitude``, ``postcode_valid``,
        ``year``, ``source``, ``adults``, ``children``, and species
        count columns.

    Returns
    -------
    pd.DataFrame
        One row per (bng_ref, year) with effort summaries and species means.
    """
    required = {"latitude", "longitude", "postcode_valid", "year",
                "source", "adults", "children"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only geocoded rows with valid coordinates
    df = df[
        df["postcode_valid"].fillna(False)
        & df["latitude"].notna()
        & df["longitude"].notna()
    ].copy()

    # Derived columns needed for aggregation
    df["_total_observers"] = df["adults"].fillna(0) + df["children"].fillna(0)
    df["_is_online"] = (df["source"] == "online").astype(np.float64)

    # Assign BNG grid references
    df = _assign_bng_ref(df)
    if df.empty:
        raise ValueError("No surveys fell within valid BNG extent")

    species_cols = _species_columns(df)
    grouped = df.groupby(["bng_ref", "year"])

    # Effort summaries
    effort = grouped.agg(
        n_surveys=("_total_observers", "size"),
        mean_observers=("_total_observers", "mean"),
        _sum_adults=("adults", lambda s: s.fillna(0).sum()),
        _sum_total=("_total_observers", "sum"),
        proportion_online=("_is_online", "mean"),
    )
    effort["proportion_adult"] = np.where(
        effort["_sum_total"] > 0,
        effort["_sum_adults"] / effort["_sum_total"],
        np.nan,
    )
    effort = effort.drop(columns=["_sum_adults", "_sum_total"])

    # Species means
    species_means = grouped[species_cols].mean()

    result = pd.concat([effort, species_means], axis=1).reset_index()

    logger.info(
        "Aggregated %d surveys -> %d (bng_ref, year) cells "
        "(%d unique cells, %d years, %d species)",
        len(df),
        len(result),
        result["bng_ref"].nunique(),
        result["year"].nunique(),
        len(species_cols),
    )
    return result
