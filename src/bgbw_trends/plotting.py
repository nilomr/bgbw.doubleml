"""Plotting utilities for DML trend results."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import TYPE_CHECKING

from bgbw_trends.viz import style
from bgbw_trends.viz._bng import refs_to_en, to_raster

if TYPE_CHECKING:
    from bgbw_trends.pipeline import SpeciesPrediction

logger = logging.getLogger(__name__)

_DEFAULT_UK_SHP = Path(
    "/home/nilomr/projects/bgbw.metadata.pipeline"
    "/data/metadata/uk-shapefile/UK-shapefile.shp"
)

# ── BNG raster helper ────────────────────────────────────────────────────────
_BNG_CELL_M = 1_000  # 1 km BNG grid


def _bng_to_scalar_raster(
    eastings: np.ndarray,
    northings: np.ndarray,
    values: np.ndarray,
    cell_m: int = _BNG_CELL_M,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Rasterise scalar values (keyed by SW-corner easting/northing) to a 2-D array.

    Parameters
    ----------
    eastings, northings : np.ndarray
        SW-corner BNG coordinates in metres (integer multiples of cell_m).
    values : np.ndarray
        Scalar value per cell.
    cell_m : int
        Cell side length in metres (1000 for 1 km BNG).

    Returns
    -------
    raster : (n_rows, n_cols) float32, NaN where no data
    extent : (e_min, e_max, n_min, n_max) — imshow extent in metres
    """
    sw_e = eastings.astype(np.int64)
    sw_n = northings.astype(np.int64)

    e_min, e_max = int(sw_e.min()), int(sw_e.max()) + cell_m
    n_min, n_max = int(sw_n.min()), int(sw_n.max()) + cell_m

    n_cols = (e_max - e_min) // cell_m
    n_rows = (n_max - n_min) // cell_m

    raster = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    col_idx = (sw_e - e_min) // cell_m
    row_idx = (sw_n - n_min) // cell_m
    raster[row_idx, col_idx] = values.astype(np.float32)

    return raster, (float(e_min), float(e_max), float(n_min), float(n_max))


_Z_80 = 1.2816  # one-sided 90th percentile of N(0,1) ⇒ 80% CI half-width


def _load_uk(shp_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shp_path)
    if gdf.crs is not None and gdf.crs.to_epsg() != 27700:
        gdf = gdf.to_crs(epsg=27700)
    return gdf


def _style_map_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")
    ax.set_facecolor(style.WATER)


def _draw_land_and_coast(
    ax: plt.Axes,
    uk: gpd.GeoDataFrame,
    extent: tuple[float, float, float, float],
) -> None:
    e_min, e_max, n_min, n_max = extent
    ax.set_xlim(e_min, e_max)
    ax.set_ylim(n_min, n_max)
    uk.plot(ax=ax, color=style.LAND, linewidth=0, zorder=1)
    uk.boundary.plot(ax=ax, color=style.COAST, linewidth=0.4, zorder=5)


def _panel_chip(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.02,
        1.01,
        text,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        clip_on=False,
        bbox={
            "boxstyle": "round,pad=0.22,rounding_size=0.14",
            "facecolor": style.PAPER,
            "edgecolor": "none",
            "alpha": 0.95,
        },
        **style.FONT_SECTION,
    )


def _header_block(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.06, 0.975, title, ha="left", va="top", **style.FONT_TITLE)
    fig.text(0.06, 0.928, subtitle, ha="left", va="top", **style.FONT_SUBTITLE)
    fig.add_artist(
        mpl.lines.Line2D(
            [0.06, 0.94],
            [0.905, 0.905],
            transform=fig.transFigure,
            color=style.GRIDLINE,
            linewidth=0.8,
        )
    )


def _footer_note(fig: plt.Figure, text: str) -> None:
    fig.text(0.94, 0.055, text, ha="right", va="center", **style.FONT_ANNOTATION)


def _add_compact_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    norm: mpl.colors.Normalize,
    cmap: mpl.colors.Colormap,
    *,
    label: str,
    fmt: str,
) -> None:
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.032, pad=0.022)
    cb.outline.set_visible(False)
    cb.ax.tick_params(
        length=2.2, width=0.5, labelsize=8.6, colors=style.TEXT_MUTED, pad=1.5
    )
    cb.ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(fmt))
    cb.ax.set_xlabel(label, **style.FONT_LABEL, labelpad=3)


def _style_series_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(style.PAPER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(style.GRIDLINE)
    ax.spines["bottom"].set_color(style.GRIDLINE)
    ax.tick_params(colors=style.TEXT_MUTED, labelsize=9)
    ax.grid(axis="y", color=style.GRIDLINE, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def _metric_raster(
    prediction: "SpeciesPrediction",
    metric: str,
) -> tuple[np.ndarray, tuple[float, float, float, float], float]:
    tau_pct = np.asarray(getattr(prediction, metric), dtype=np.float64).ravel() * 100.0
    east, north = refs_to_en(np.asarray(prediction.bng_ref))
    valid = np.isfinite(east) & np.isfinite(north) & np.isfinite(tau_pct)
    raster, extent = to_raster(east[valid], north[valid], tau_pct[valid])
    vmax = max(float(np.nanpercentile(np.abs(tau_pct[valid]), 97.5)), 0.01)
    return raster, extent, vmax


def plot_trend_ppy_panels(
    prediction: "SpeciesPrediction",
    output_path: Path | None = None,
    *,
    uk_shp: Path | str = _DEFAULT_UK_SHP,
    dpi: int = 300,
) -> plt.Figure:
    """Three-panel trend map (occurrence, count, abundance), in % per year.

    This figure intentionally omits the old significant-only panels and shows
    all three trend components side by side using the map style from
    ``scripts/pipeline/make_maps.py`` via ``bgbw_trends.viz.style`` tokens.
    """
    style.apply()
    uk = _load_uk(Path(uk_shp))

    panels = [
        ("occ_ppy", "Occurrence trend"),
        ("count_ppy", "Count trend"),
        ("abd_ppy", "Abundance trend"),
    ]

    rasters: list[np.ndarray] = []
    extents: list[tuple[float, float, float, float]] = []
    vmaxes: list[float] = []
    for metric, _ in panels:
        raster, extent, vmax = _metric_raster(prediction, metric)
        rasters.append(raster)
        extents.append(extent)
        vmaxes.append(vmax)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 7.0), facecolor=style.PAPER)
    cbar_mappables = []

    for ax, raster, extent, vmax, (_, title_suffix) in zip(
        axes, rasters, extents, vmaxes, panels, strict=False
    ):
        _style_map_axis(ax)
        _draw_land_and_coast(ax, uk, extent)

        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        rgba = plt.get_cmap("RdBu")(norm(np.nan_to_num(raster, nan=0.0)))
        rgba[np.isnan(raster), 3] = 0.0
        im = ax.imshow(
            rgba,
            origin="lower",
            extent=extent,
            interpolation="none",
            aspect="equal",
            zorder=3,
        )
        cbar_mappables.append((im, norm))
        _panel_chip(ax, title_suffix)

    for ax, (_, norm) in zip(axes, cbar_mappables, strict=False):
        _add_compact_colorbar(
            fig,
            ax,
            norm,
            plt.get_cmap("RdBu"),
            label="Trend (%/yr)",
            fmt="%.1f",
        )

    species_label = prediction.species.replace("_", " ").title()
    _header_block(fig, species_label, "Spatial trend components across 1 km grid cells")

    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.88])

    if output_path:
        fig.savefig(output_path, dpi=dpi, facecolor=style.PAPER)
        logger.info("Saved 3-panel trend map to %s", output_path)

    return fig


def plot_trend_map(
    prediction: "SpeciesPrediction",
    metric: str = "abd_ppy",
    output_path: Path | None = None,
) -> plt.Figure:
    """Map of spatially varying trend from a ``SpeciesPrediction`` grid.

    Plots the chosen per-year rate (``occ_ppy``, ``count_ppy`` or
    ``abd_ppy``) and flags 80% CI significance. The abundance rate has no
    directly attached variance (it combines two causal forests), so
    significance there is reported only for metrics with ``*_sigma``
    available on the prediction object.
    """
    # Stored trend values are ratios per year; convert to percent-per-year
    # for visualization to match terminal summaries and viz/maps.py.
    tau = np.asarray(getattr(prediction, metric), dtype=np.float64).ravel() * 100.0
    sigma_attr = {"occ_ppy": "occ_sigma", "count_ppy": "count_sigma"}.get(metric)
    sigma = (
        np.asarray(getattr(prediction, sigma_attr), dtype=np.float64).ravel() * 100.0
        if sigma_attr is not None
        else None
    )

    df = pd.DataFrame({"bng_ref": prediction.bng_ref, "tau": tau})
    coords = df["bng_ref"].apply(_parse_bng_approx)
    df["easting"] = coords.apply(lambda c: c[0])
    df["northing"] = coords.apply(lambda c: c[1])
    df = df.dropna(subset=["easting", "northing"])
    east = df["easting"].to_numpy()
    north = df["northing"].to_numpy()
    tau = df["tau"].to_numpy()

    if sigma is not None:
        sigma = sigma[df.index.to_numpy()]
        significant = (np.abs(tau) - _Z_80 * sigma) > 0
    else:
        significant = np.ones_like(tau, dtype=bool)

    vmax = max(abs(np.nanquantile(tau, 0.05)), abs(np.nanquantile(tau, 0.95)))
    vmax = max(vmax, 1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu")

    raster_all, extent = _bng_to_scalar_raster(east, north, tau)
    tau_sig = tau.copy()
    tau_sig[~significant] = np.nan
    raster_sig, _ = _bng_to_scalar_raster(east, north, tau_sig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    e_min, e_max, n_min, n_max = extent

    panels = (
        (axes[0], raster_all, "all trends"),
        (axes[1], raster_sig, "significant (80% CI)"),
    )
    for ax, raster, title_suffix in panels:
        ax.set_facecolor("#cccccc")
        rgba = cmap(norm(np.nan_to_num(raster, nan=0.0)))
        rgba[np.isnan(raster), 3] = 0.0
        ax.imshow(
            rgba,
            origin="lower",
            extent=(e_min, e_max, n_min, n_max),
            interpolation="nearest",
            aspect="equal",
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="annual change (%/year)", shrink=0.7)
        ax.set_title(f"{prediction.species} — {title_suffix}")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")

    fig.suptitle(f"Spatially varying {metric}: {prediction.species}", fontsize=13)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved trend map to %s", output_path)
    return fig


def plot_sdm_prediction_maps(
    prediction: "SpeciesPrediction",
    output_path: Path | None = None,
    *,
    uk_shp: Path | str = _DEFAULT_UK_SHP,
    dpi: int = 300,
) -> plt.Figure:
    """Plot effort-adjusted SDM maps for occurrence and abundance as 1 km BNG rasters."""
    style.apply()
    uk = _load_uk(Path(uk_shp))

    df = pd.DataFrame(
        {
            "bng_ref": prediction.bng_ref,
            "p_occurrence": prediction.p_occurrence,
            "e_abundance": prediction.e_abundance,
        }
    )

    east, north = refs_to_en(df["bng_ref"].to_numpy())
    valid = np.isfinite(east) & np.isfinite(north)
    east = east[valid]
    north = north[valid]
    occ = df["p_occurrence"].to_numpy(dtype=np.float64)[valid]
    e_abund = df["e_abundance"].to_numpy(dtype=np.float64)[valid]

    raster_occ, extent = to_raster(east, north, occ)
    emax = float(np.nanquantile(e_abund, 0.99)) if len(e_abund) else 1.0
    raster_abund, _ = to_raster(east, north, e_abund)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 7.0), facecolor=style.PAPER)

    # Occurrence — scale to data range so spatial variation is visible
    occ_lo = float(np.nanpercentile(occ, 1))
    occ_hi = float(np.nanpercentile(occ, 99))
    norm_occ = mcolors.Normalize(
        vmin=max(occ_lo, 0.0), vmax=min(max(occ_hi, occ_lo + 1e-6), 1.0)
    )
    cmap_occ = plt.get_cmap("viridis")
    rgba_occ = cmap_occ(norm_occ(np.nan_to_num(raster_occ, nan=0.0)))
    rgba_occ[np.isnan(raster_occ), 3] = 0.0
    _style_map_axis(axes[0])
    _draw_land_and_coast(axes[0], uk, extent)
    axes[0].imshow(
        rgba_occ,
        origin="lower",
        extent=extent,
        interpolation="none",
        aspect="equal",
        zorder=3,
    )
    _panel_chip(axes[0], "Probability of occurrence")
    _add_compact_colorbar(
        fig, axes[0], norm_occ, cmap_occ, label="P(occurrence)", fmt="%.2f"
    )

    # Abundance
    norm_abund = mcolors.Normalize(vmin=0, vmax=max(emax, 1e-6))
    cmap_abund = plt.get_cmap("magma")
    rgba_abund = cmap_abund(norm_abund(np.nan_to_num(raster_abund, nan=0.0)))
    rgba_abund[np.isnan(raster_abund), 3] = 0.0
    _style_map_axis(axes[1])
    _draw_land_and_coast(axes[1], uk, extent)
    axes[1].imshow(
        rgba_abund,
        origin="lower",
        extent=extent,
        interpolation="none",
        aspect="equal",
        zorder=3,
    )
    _panel_chip(axes[1], "Expected abundance")
    _add_compact_colorbar(
        fig, axes[1], norm_abund, cmap_abund, label="E[count]", fmt="%.2f"
    )

    species_label = prediction.species.replace("_", " ").title()
    _header_block(fig, species_label, "Effort-adjusted species distribution model")
    _footer_note(
        fig,
        f"Median P(occ) {np.nanmedian(occ):.3f}  •  Median E[abundance] {np.nanmedian(e_abund):.2f}",
    )
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.88])

    if output_path:
        fig.savefig(output_path, dpi=dpi, facecolor=style.PAPER)
        logger.info("Saved SDM prediction maps to %s", output_path)
    return fig


# ── BNG grid reference parsing ──────────────────────────────────────────────

_BNG_LETTERS = {
    "S": (0, 0),
    "T": (5, 0),
    "N": (0, 5),
    "O": (5, 5),
    "H": (0, 10),
}

_BNG_SECOND = {
    "A": (0, 4),
    "B": (1, 4),
    "C": (2, 4),
    "D": (3, 4),
    "E": (4, 4),
    "F": (0, 3),
    "G": (1, 3),
    "H": (2, 3),
    "J": (3, 3),
    "K": (4, 3),
    "L": (0, 2),
    "M": (1, 2),
    "N": (2, 2),
    "O": (3, 2),
    "P": (4, 2),
    "Q": (0, 1),
    "R": (1, 1),
    "S": (2, 1),
    "T": (3, 1),
    "U": (4, 1),
    "V": (0, 0),
    "W": (1, 0),
    "X": (2, 0),
    "Y": (3, 0),
    "Z": (4, 0),
}


def _yearly_trend_summary(obs: pd.DataFrame, ate: float) -> pd.DataFrame:
    """Per-year summary for the deconfounded-trend plot.

    Columns:
    - ``raw_mean``, ``raw_se``: observed yearly mean of Y and its SE
      (binomial for binary ``raw``, sample SEM otherwise).
    - ``cond_mean``: yearly mean of ``m̂(X_i)``, the conditional mean
      model evaluated at each year's actual X rows. This series tracks
      the component of the raw yearly change that is attributable to
      X-drift across years (i.e. confounding as captured by X).
    - ``deconf``: ``Ȳ + ATE·(year − T̄)``. The pure-trend line the
      DML model claims after partialling out X — the direct analogue
      of the "true trend" orange line in Fink's motivating-example.Rmd
      fig1a, but with ``ATE`` substituted for the unknown true slope.
    """
    g = obs.groupby("year", sort=True)
    years = np.asarray(sorted(obs["year"].unique()), dtype=int)
    raw_mean = g["raw"].mean().reindex(years).to_numpy()
    n = g["raw"].count().reindex(years).to_numpy()

    is_binary = set(np.unique(obs["raw"].to_numpy())).issubset({0.0, 1.0})
    if is_binary:
        raw_se = np.sqrt(np.maximum(raw_mean * (1 - raw_mean) / np.maximum(n, 1), 0))
    else:
        raw_se = g["raw"].sem().reindex(years).to_numpy()

    cond_mean = g["m_hat"].mean().reindex(years).to_numpy()

    y_bar = float(obs["raw"].mean())
    t_bar = float(obs["year"].mean())
    deconf = y_bar + ate * (years.astype(float) - t_bar)

    return pd.DataFrame(
        {
            "year": years,
            "n": n,
            "raw_mean": raw_mean,
            "raw_se": raw_se,
            "cond_mean": cond_mean,
            "deconf": deconf,
        }
    )


def plot_yearly_trend(
    prediction: "SpeciesPrediction",
    output_path: Path | None = None,
    *,
    dpi: int = 300,
) -> plt.Figure:
    """Raw vs deconfounded model trend, one panel per stage.

    The DML model's goal is to isolate the *pure time effect* on species
    abundance — the part of year-to-year change that cannot be explained by
    changes in survey effort, weather, or habitat covariates (X).

    The three series relate via the identity (Fink 2023, motivating example):

        raw slope ≈ slope of grey + ATE (orange slope)

    so the **slope of the grey line is the confounding component** — the
    year-to-year drift in raw Y that is due to the X covariates changing over
    time rather than true species change.

    - **Raw mean** (blue): the observed yearly mean of Y (P(occurrence) or
      log1p count).  Confounded — mixes genuine trend with any drift in who
      surveys, when, and where.

    - **Conditional mean m̂(X)** (grey dashed): the yearly mean of
      m̂(Xᵢ) — the marginal forest's fitted values, which depend on X only
      (never on year).  Because X covariates (effort, weather, habitat) shift
      in composition from year to year, this series drifts even though year
      is absent from m̂.  Its slope equals the portion of the raw trend that
      DML attributes to X-composition change.  If grey and blue have the same
      slope, X explains *all* of the apparent trend → ATE ≈ 0 (large
      adjustment).  If grey is flat while blue trends, X doesn't change with
      year → ATE ≈ raw slope (small or no adjustment).

    - **Deconfounded trend** (orange): ``Ȳ + ATE·(y − T̄)`` where
      ``ATE = mean τ̂(W)``.  A straight line through the grand mean with slope
      equal to the causal-forest estimate — the residual year effect after
      partialling out m̂(X).  Direct analogue of the known-truth orange line in
      Fink's motivating-example fig1a.

    Each panel carries a slope annotation: raw OLS slope, ATE, and their
    difference (the deconfounding adjustment = slope of grey).

    Occurrence is on the probability scale; count is on log1p.
    """
    if prediction.obs_occ.empty or prediction.obs_count.empty:
        raise ValueError(
            "SpeciesPrediction has no per-observation data. "
            "Run train_predict_species to populate obs_occ / obs_count."
        )

    summary_occ = _yearly_trend_summary(prediction.obs_occ, prediction.ate_occ)
    summary_count = _yearly_trend_summary(prediction.obs_count, prediction.ate_count)

    style.apply()
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0), facecolor=style.PAPER)
    level_panels = [
        (
            axes[0, 0],
            summary_occ,
            prediction.ate_occ,
            "Occurrence",
            "P(occurrence)",
            True,
        ),
        (
            axes[0, 1],
            summary_count,
            prediction.ate_count,
            "Count (log1p)",
            "log(1 + count)",
            False,
        ),
    ]
    diff_panels = [
        (
            axes[1, 0],
            summary_occ,
            prediction.ate_occ,
            "Occurrence — year-on-year Δ",
            "ΔP(occurrence)",
        ),
        (
            axes[1, 1],
            summary_count,
            prediction.ate_count,
            "Count — year-on-year Δ",
            "Δlog(1 + count)",
        ),
    ]
    _RAW = "#2C6E91"
    _TREND = "#C5672F"
    _COND = "#B5AFA8"

    # ── Top row: levels ──────────────────────────────────────────────────
    for ax, s, ate, title, ylabel, is_binary in level_panels:
        _style_series_axis(ax)
        years = s["year"].to_numpy()

        # SE band
        ax.fill_between(
            years,
            s["raw_mean"] - s["raw_se"],
            s["raw_mean"] + s["raw_se"],
            color=_RAW,
            alpha=0.13,
            zorder=2,
        )
        ax.plot(
            years,
            s["raw_mean"],
            "-o",
            color=_RAW,
            linewidth=1.4,
            markersize=3.5,
            markeredgewidth=0,
            label="Raw mean ± SE",
            zorder=3,
        )
        ax.plot(
            years,
            s["cond_mean"],
            "--",
            color=_COND,
            linewidth=1.0,
            dashes=(5, 3),
            label="Conditional mean m̂(X)",
            zorder=2,
        )
        ax.plot(
            years,
            s["deconf"],
            "-",
            color=_TREND,
            linewidth=2.0,
            label=f"Deconfounded ({ate:+.4f}/yr)",
            zorder=4,
        )

        all_vals = np.concatenate(
            [
                s["raw_mean"] - s["raw_se"],
                s["raw_mean"] + s["raw_se"],
                s["cond_mean"],
                s["deconf"],
            ]
        )
        finite = all_vals[np.isfinite(all_vals)]
        if finite.size:
            ylo, yhi = float(finite.min()), float(finite.max())
            pad = max((yhi - ylo) * 0.10, 1e-3)
            ax.set_ylim(max(0.0, ylo - pad) if is_binary else ylo - pad, yhi + pad)

        ax.set_xlabel("Year", **style.FONT_LABEL)
        ax.set_ylabel(ylabel, **style.FONT_LABEL)
        tick_years = years[::2] if len(years) > 8 else years
        ax.set_xticks(tick_years)
        _panel_chip(ax, title)
        ax.legend(
            loc="upper right", frameon=False, fontsize=8.2, labelcolor=style.TEXT_MUTED
        )

    # ── Bottom row: first differences ────────────────────────────────────
    # Identity: Δraw ≈ Δm̂(X) + ATE
    # Blue bars  = Δraw  (year-on-year change actually observed)
    # Grey bars  = Δm̂(X) (change attributed to X-composition drift)
    # Orange line = ATE  (the model's constant per-year true trend)
    # The two bars should stack to match the orange line if X fully
    # explains any deviation of the raw changes from ATE.
    for ax, s, ate, title, ylabel in diff_panels:
        _style_series_axis(ax)
        years = s["year"].to_numpy()
        if len(years) < 2:
            ax.set_visible(False)
            continue

        diff_years = years[1:]
        d_raw = np.diff(s["raw_mean"].to_numpy())
        d_cond = np.diff(s["cond_mean"].to_numpy())

        bar_w = 0.35
        offsets = np.arange(len(diff_years), dtype=float)
        ax.bar(
            offsets - bar_w / 2,
            d_raw,
            width=bar_w,
            color=_RAW,
            alpha=0.75,
            label="Δ raw mean",
            zorder=3,
        )
        ax.bar(
            offsets + bar_w / 2,
            d_cond,
            width=bar_w,
            color=_COND,
            alpha=0.75,
            label="Δ m̂(X)  [confounding]",
            zorder=3,
        )

        # ATE as horizontal reference: constant true trend per year
        ax.axhline(
            ate,
            color=_TREND,
            linewidth=1.8,
            linestyle="-",
            label=f"ATE {ate:+.4f}/yr  [true trend]",
            zorder=4,
        )
        ax.axhline(0, color=style.TEXT_MUTED, linewidth=0.6, linestyle="--", zorder=1)

        ax.set_xticks(offsets)
        ax.set_xticklabels(
            [f"{diff_years[i] - 1}→{diff_years[i]}" for i in range(len(diff_years))],
            fontsize=7.5,
            rotation=0,
            ha="center",
        )
        ax.set_xlabel("Year transition", **style.FONT_LABEL)
        ax.set_ylabel(ylabel, **style.FONT_LABEL)
        _panel_chip(ax, title)
        ax.legend(
            loc="upper right", frameon=False, fontsize=8.2, labelcolor=style.TEXT_MUTED
        )

    species_label = prediction.species.replace("_", " ").title()
    _header_block(
        fig,
        species_label,
        "Yearly raw means, confounding drift, and deconfounded trend",
    )
    _footer_note(
        fig,
        f"Occurrence ATE {prediction.ate_occ:+.4f}/yr  •  Count ATE {prediction.ate_count:+.4f}/yr  "
        f"•  bottom row: Δraw ≈ Δm̂(X) + ATE",
    )
    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.88])

    if output_path:
        fig.savefig(output_path, dpi=dpi, facecolor=style.PAPER)
        logger.info("Saved yearly trend plot to %s", output_path)
    return fig


def _reliability(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reliability curve via equal-mass (quantile) bins.

    Equal-mass bins are more honest than equal-width: each bucket
    contains a similar number of observations, so the observed
    frequency per bucket has comparable sampling error.
    """
    order = np.argsort(y_prob)
    p_sorted = y_prob[order]
    y_sorted = y_true[order]
    splits = np.array_split(np.arange(len(p_sorted)), n_bins)
    pmean = np.array([p_sorted[s].mean() for s in splits if len(s)])
    ymean = np.array([y_sorted[s].mean() for s in splits if len(s)])
    n = np.array([len(s) for s in splits if len(s)])
    return pmean, ymean, n


def _finite_values(values: np.ndarray, *, scale: float = 1.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel() * scale
    return arr[np.isfinite(arr)]


def _positive_log_bins(values: np.ndarray, n_bins: int = 48) -> np.ndarray | None:
    positive = values[values > 0.0]
    if positive.size == 0:
        return None
    lo = max(float(np.nanpercentile(positive, 0.5)), 1e-6)
    hi = max(float(np.nanpercentile(positive, 99.5)), lo * 1.01)
    return np.geomspace(lo, hi, n_bins)


def _plot_distribution_panel(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    color: str,
    title: str,
    xlabel: str,
    log_x: bool = False,
) -> None:
    _style_series_axis(ax)
    clean = _finite_values(values)
    if clean.size == 0:
        _panel_chip(ax, title)
        ax.text(
            0.5,
            0.5,
            "No finite values",
            ha="center",
            va="center",
            **style.FONT_ANNOTATION,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return

    if log_x:
        bins = _positive_log_bins(clean)
        positive = clean[clean > 0.0]
        if bins is None or positive.size == 0:
            _panel_chip(ax, title)
            ax.text(
                0.5,
                0.5,
                "No positive values",
                ha="center",
                va="center",
                **style.FONT_ANNOTATION,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            return
        ax.hist(positive, bins=bins, color=color, alpha=0.78)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
        zero_share = 100.0 * (clean <= 0.0).mean()
        note = f"median {np.median(clean):.3f}  •  zero cells {zero_share:.1f}%"
    else:
        lo, hi = np.nanpercentile(clean, [0.5, 99.5])
        hist_values = clean[(clean >= lo) & (clean <= hi)]
        ax.hist(hist_values, bins=55, color=color, alpha=0.78)
        note = f"median {np.median(clean):.2f}"

    _panel_chip(ax, title)
    ax.set_xlabel(xlabel, **style.FONT_LABEL)
    ax.set_ylabel("Grid cells", **style.FONT_LABEL)
    ax.text(
        0.02,
        0.04,
        note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        **style.FONT_ANNOTATION,
    )


def plot_prediction_distribution_summary(
    prediction: "SpeciesPrediction",
    ensemble: object | None = None,
    output_path: Path | None = None,
    *,
    dpi: int = 300,
) -> plt.Figure:
    """Concise summary of SDM and trend output distributions.

    The figure is designed for quick interpretation of a species run:
    occurrence-probability and abundance distributions, the three grid-level
    trend distributions, and a compact summary panel with pipeline means and,
    when available, ensemble abundance uncertainty.
    """
    style.apply()

    p_occ = _finite_values(prediction.p_occurrence)
    e_abundance = _finite_values(prediction.e_abundance)
    occ_ppy = _finite_values(prediction.occ_ppy, scale=100.0)
    count_ppy = _finite_values(prediction.count_ppy, scale=100.0)
    abd_ppy = _finite_values(prediction.abd_ppy, scale=100.0)
    occ_sigma = _finite_values(getattr(prediction, "occ_sigma", np.array([])))
    count_sigma = _finite_values(getattr(prediction, "count_sigma", np.array([])))

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.4), facecolor=style.PAPER)
    ax_occ, ax_abd = axes[0]
    ax_trend, ax_summary = axes[1]

    _plot_distribution_panel(
        ax_occ,
        prediction.p_occurrence,
        color="#2C6E91",
        title="Occurrence probability distribution",
        xlabel="P(occurrence)",
        log_x=True,
    )
    _plot_distribution_panel(
        ax_abd,
        prediction.e_abundance,
        color="#B9652D",
        title="Expected abundance distribution",
        xlabel="E[abundance]",
        log_x=True,
    )

    _style_series_axis(ax_trend)
    _panel_chip(ax_trend, "Trend distributions")
    for values, color, label in [
        (occ_ppy, "#2C6E91", "Occurrence"),
        (count_ppy, "#C5672F", "Count"),
        (abd_ppy, "#5A7F62", "Abundance"),
    ]:
        if values.size == 0:
            continue
        lo, hi = np.nanpercentile(values, [1, 99])
        clipped = values[(values >= lo) & (values <= hi)]
        ax_trend.hist(
            clipped,
            bins=80,
            density=True,
            histtype="step",
            linewidth=1.9,
            color=color,
            label=f"{label} median {np.median(values):+.2f}%/yr",
        )
        ax_trend.axvline(np.median(values), color=color, linewidth=1.0, alpha=0.85)
    ax_trend.set_xlabel("Trend (%/yr)", **style.FONT_LABEL)
    ax_trend.set_ylabel("Density", **style.FONT_LABEL)
    ax_trend.legend(
        frameon=False,
        fontsize=8.4,
        labelcolor=style.TEXT_MUTED,
        loc="upper left",
    )

    _style_series_axis(ax_summary)
    _panel_chip(ax_summary, "Summary")
    ax_summary.grid(False)
    ax_summary.set_xticks([])
    ax_summary.set_yticks([])
    for spine in ax_summary.spines.values():
        spine.set_visible(False)

    summary_lines = [
        f"Grid cells  {len(np.asarray(prediction.bng_ref)):,}",
        f"Median P(occ)  {np.nanmedian(p_occ):.3f}",
        f"Median E[abundance]  {np.nanmedian(e_abundance):.2f}",
        f"Occurrence trend  {np.nanmedian(occ_ppy):+.2f}%/yr",
        f"Count trend  {np.nanmedian(count_ppy):+.2f}%/yr",
        f"Abundance trend  {np.nanmedian(abd_ppy):+.2f}%/yr",
    ]
    if np.asarray(getattr(prediction, "occ_sigma", np.array([]))).size:
        occ_sigma_finite = (
            np.isfinite(np.asarray(prediction.occ_sigma, dtype=np.float64)).mean()
            * 100.0
        )
        summary_lines.append(f"Finite occ sigma  {occ_sigma_finite:.2f}%")
    if np.asarray(getattr(prediction, "count_sigma", np.array([]))).size:
        count_sigma_finite = (
            np.isfinite(np.asarray(prediction.count_sigma, dtype=np.float64)).mean()
            * 100.0
        )
        summary_lines.append(f"Finite count sigma  {count_sigma_finite:.2f}%")

    ax_summary.text(
        0.02,
        0.88,
        "\n".join(summary_lines),
        transform=ax_summary.transAxes,
        ha="left",
        va="top",
        linespacing=1.5,
        fontsize=9.4,
        color=style.TEXT,
        fontfamily=style.FONT_FAMILY,
    )

    compare_ax = ax_summary.inset_axes([0.08, 0.05, 0.84, 0.40])
    _style_series_axis(compare_ax)
    compare_ax.grid(axis="x", color=style.GRIDLINE, linewidth=0.8, alpha=0.8)
    compare_ax.grid(axis="y", visible=False)
    compare_ax.set_ylabel("")

    _CLR_OCC = "#2C6E91"
    _CLR_CNT = "#C5672F"
    _CLR_ABD = "#5A7F62"

    # Each row: (label, center, lower, upper, marker, color)
    rows: list[tuple[str, float, float | None, float | None, str, str]] = [
        ("Pipeline  occ",   float(np.nanmedian(occ_ppy)),   None, None, "o", _CLR_OCC),
        ("Pipeline  count", float(np.nanmedian(count_ppy)), None, None, "o", _CLR_CNT),
        ("Pipeline  abd",   float(np.nanmedian(abd_ppy)),   None, None, "o", _CLR_ABD),
    ]
    if ensemble is not None:
        def _ens(attr: str) -> float | None:
            arr = _finite_values(getattr(ensemble, attr, np.array([])), scale=100.0)
            return float(np.nanmedian(arr)) if arr.size else None

        ens_occ   = _ens("occ_ppy")
        ens_occ_lo, ens_occ_hi     = _ens("occ_ppy_lower"),   _ens("occ_ppy_upper")
        ens_cnt   = _ens("count_ppy")
        ens_cnt_lo, ens_cnt_hi     = _ens("count_ppy_lower"), _ens("count_ppy_upper")
        ens_abd   = _ens("abd_ppy")
        ens_abd_lo, ens_abd_hi     = _ens("abd_ppy_lower"),   _ens("abd_ppy_upper")

        if ens_occ is not None:
            rows.append(("Ensemble  occ",   ens_occ,   ens_occ_lo, ens_occ_hi, "s", _CLR_OCC))
        if ens_cnt is not None:
            rows.append(("Ensemble  count", ens_cnt,   ens_cnt_lo, ens_cnt_hi, "s", _CLR_CNT))
        if ens_abd is not None:
            rows.append(("Ensemble  abd",   ens_abd,   ens_abd_lo, ens_abd_hi, "s", _CLR_ABD))

    y = np.arange(len(rows))[::-1]
    centers = [row[1] for row in rows]
    lowers = [row[2] if row[2] is not None else row[1] for row in rows]
    uppers = [row[3] if row[3] is not None else row[1] for row in rows]
    lo = min(lowers + [0.0])
    hi = max(uppers + [0.0])
    pad = max((hi - lo) * 0.18, 0.5)
    compare_ax.set_xlim(lo - pad, hi + pad)
    compare_ax.axvline(0.0, color=style.GRIDLINE, linewidth=1.0)

    for yi, (label, center, lower, upper, marker, color) in zip(y, rows, strict=False):
        if lower is not None and upper is not None and (lower != center or upper != center):
            compare_ax.hlines(yi, lower, upper, color=color, linewidth=2.0, alpha=0.6)
        compare_ax.scatter([center], [yi], s=52, marker=marker, color=color, zorder=3)
    compare_ax.set_yticks(y, [row[0] for row in rows])
    compare_ax.tick_params(axis="y", labelsize=8.0)
    compare_ax.set_xlabel("Trend (%/yr)", **style.FONT_LABEL)

    species_label = prediction.species.replace("_", " ").title()
    subtitle = (
        "Occurrence and abundance distributions, trend components, and run statistics"
    )
    if ensemble is not None:
        subtitle = "Occurrence and abundance distributions, trend components, and ensemble uncertainty"
    _header_block(fig, species_label, subtitle)

    footer = (
        f"Occurrence trend {np.nanmedian(occ_ppy):+.2f}%/yr  •  "
        f"Count trend {np.nanmedian(count_ppy):+.2f}%/yr  •  "
        f"Abundance trend {np.nanmedian(abd_ppy):+.2f}%/yr"
    )
    _footer_note(fig, footer)
    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.88])

    if output_path:
        fig.savefig(output_path, dpi=dpi, facecolor=style.PAPER)
        logger.info("Saved prediction distribution summary to %s", output_path)

    return fig


def plot_raw_vs_fitted(
    prediction: "SpeciesPrediction",
    output_path: Path | None = None,
    n_occ_bins: int = 12,
) -> plt.Figure:
    """Model-fit diagnostics: calibration for occurrence, scatter for count.

    The "fitted" value used here is the PLR reconstruction at each
    training row: ``Ŷ_i = m̂(X_i) + τ̂(W_i)·(T_i − ŝ(X_i))``. Binary
    stage fitted probabilities are clipped to ``[0, 1]`` only for
    display of the reliability diagram.

    - **Occurrence (left)**: equal-mass reliability diagram. Points on
      the diagonal mean the fitted probabilities match observed
      frequencies. Deviations indicate miscalibration of the combined
      nuisance + CATE fit.
    - **Count (right)**: raw vs fitted log1p, one dot per detected
      observation, coloured by year to reveal systematic year drift in
      the fit. The diagonal is a perfect fit.
    """
    if prediction.obs_occ.empty or prediction.obs_count.empty:
        raise ValueError(
            "SpeciesPrediction has no per-observation data. "
            "Run train_predict_species to populate obs_occ / obs_count."
        )

    obs_occ = prediction.obs_occ
    obs_count = prediction.obs_count

    fitted_occ = np.clip(
        obs_occ["m_hat"].to_numpy()
        + obs_occ["tau"].to_numpy()
        * (obs_occ["year"].to_numpy() - obs_occ["s_hat"].to_numpy()),
        0.0,
        1.0,
    )
    fitted_count = obs_count["m_hat"].to_numpy() + obs_count["tau"].to_numpy() * (
        obs_count["year"].to_numpy() - obs_count["s_hat"].to_numpy()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Occurrence reliability ─────────────────────────────────────────
    ax = axes[0]
    pmean, ymean, n_per = _reliability(
        obs_occ["raw"].to_numpy(), fitted_occ, n_occ_bins
    )
    sizes = 20 + 180 * (n_per / n_per.max())
    ax.scatter(
        pmean,
        ymean,
        s=sizes,
        color="steelblue",
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Fitted P(occurrence)")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title(
        f"Occurrence — reliability (Brier={prediction.diagnostics.occurrence_brier:.3f})"
    )

    # ── Count scatter coloured by year ─────────────────────────────────
    ax = axes[1]
    years = np.asarray(sorted(obs_count["year"].unique()))
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=years.min(), vmax=years.max())
    ax.scatter(
        fitted_count,
        obs_count["raw"].to_numpy(),
        c=obs_count["year"].to_numpy(),
        cmap=cmap,
        norm=norm,
        alpha=0.35,
        s=8,
        linewidth=0,
    )
    lo = float(min(obs_count["raw"].min(), fitted_count.min()))
    hi = float(max(obs_count["raw"].max(), fitted_count.max()))
    ax.plot([lo, hi], [lo, hi], "--", color="grey", linewidth=0.8)
    ax.set_xlabel("Fitted log(1 + count)")
    ax.set_ylabel("Raw log(1 + count)")
    ax.set_title(f"Count — raw vs fitted (R²={prediction.diagnostics.count_r2:.3f})")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Year", shrink=0.8)

    fig.suptitle(f"Model-fit diagnostics: {prediction.species}", fontsize=13)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved raw-vs-fitted plot to %s", output_path)
    return fig


# ── BNG grid reference parsing ──────────────────────────────────────────────


def _parse_bng_approx(ref: str) -> tuple[float, float]:
    """Parse a BNG grid reference to approximate easting/northing (metres)."""
    try:
        first, second = ref[0], ref[1]
        digits = ref[2:]
        half = len(digits) // 2
        e_digits, n_digits = digits[:half], digits[half:]

        base_e, base_n = _BNG_LETTERS.get(first, (0, 0))
        sq_e, sq_n = _BNG_SECOND.get(second, (0, 0))

        easting = (base_e + sq_e) * 100_000 + int(e_digits) * 10 ** (5 - half)
        northing = (base_n + sq_n) * 100_000 + int(n_digits) * 10 ** (5 - half)
        return (easting, northing)
    except (IndexError, ValueError, TypeError):
        return (np.nan, np.nan)
