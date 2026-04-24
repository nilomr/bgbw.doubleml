#!/usr/bin/env python3
"""Hexagonal abundance maps for BGBW model output.

Aggregates 1 km BNG grid-cell predictions for abundance and trend in abundance into a regular pointy-top hex grid,
clips each cell to the UK coastline

Usage
-----
    uv run python scripts/make_hex_map.py
    uv run python scripts/make_hex_map.py --results output/run1 --species blackbird
    uv run python scripts/make_hex_map.py --hex-size 15000   # 15 km circumradius
    uv run python scripts/make_hex_map.py --dpi 150 --species starling

Output
------
For each species found:
    <results_dir>/figures/maps/<species>/hex_abundance.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bgbw_trends.viz import style
from bgbw_trends.viz._bng import refs_to_en

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Viridis: perceptually uniform, dark-purple (sparse) → bright-yellow (dense).
# Brighter always means more abundant; works for colour-blind readers.
_HEX_CMAP = plt.get_cmap("viridis")

# Neutral warm grey for land cells with no survey coverage.
# Visually reads as "blank / unknown", not as low abundance.
_NO_DATA = "#9C9590"

_DEFAULT_UK_SHP = Path(
    "/home/nilomr/projects/bgbw.metadata.pipeline"
    "/data/metadata/uk-shapefile/UK-shapefile.shp"
)

_COMMON_NAMES: dict[str, str] = {
    "blackbird": "Common Blackbird  —  Turdus merula",
    "blue_tit": "Blue Tit  —  Cyanistes caeruleus",
    "house_sparrow": "House Sparrow  —  Passer domesticus",
    "starling": "Common Starling  —  Sturnus vulgaris",
    "robin": "European Robin  —  Erithacus rubecula",
    "wren": "Eurasian Wren  —  Troglodytes troglodytes",
    "great_tit": "Great Tit  —  Parus major",
    "chaffinch": "Common Chaffinch  —  Fringilla coelebs",
    "wood_pigeon": "Common Wood Pigeon  —  Columba palumbus",
    "goldfinch": "European Goldfinch  —  Carduelis carduelis",
    "greenfinch": "European Greenfinch  —  Chloris chloris",
    "song_thrush": "Song Thrush  —  Turdus philomelos",
}


# ── Hex grid construction ──────────────────────────────────────────────────


def _hex_polygon(cx: float, cy: float, r: float) -> Polygon:
    """Pointy-top hexagonal Polygon centred at (cx, cy) with circumradius r."""
    angles = np.radians(np.arange(30, 391, 60))
    return Polygon(zip(cx + r * np.cos(angles), cy + r * np.sin(angles)))


def _make_hex_grid(
    bounds: tuple[float, float, float, float],
    hex_size: float,
) -> gpd.GeoDataFrame:
    """Regular pointy-top hex grid covering *bounds* in EPSG:27700.

    Parameters
    ----------
    bounds:
        (minx, miny, maxx, maxy) in metres (EPSG:27700).
    hex_size:
        Circumradius (centre to vertex) in metres.
    """
    minx, miny, maxx, maxy = bounds
    col_step = hex_size * np.sqrt(3)  # horizontal centre-to-centre spacing
    row_step = hex_size * 1.5  # vertical centre-to-centre spacing
    half_col = col_step / 2.0  # odd-row horizontal offset

    cols = np.arange(minx - col_step, maxx + col_step * 2, col_step)
    rows = np.arange(miny - row_step, maxy + row_step * 2, row_step)

    polys = []
    for r_idx, cy in enumerate(rows):
        x_shift = half_col if r_idx % 2 else 0.0
        for cx in cols + x_shift:
            polys.append(_hex_polygon(cx, cy, hex_size))

    return gpd.GeoDataFrame(geometry=polys, crs="EPSG:27700")


# ── Aggregation ────────────────────────────────────────────────────────────


def _aggregate_to_hex(
    east: np.ndarray,
    north: np.ndarray,
    values: np.ndarray,
    hex_gdf: gpd.GeoDataFrame,
    min_cells: int = 3,
) -> gpd.GeoDataFrame:
    """Spatial-join 1 km cell centroids to hexes; summarise by median.

    Parameters
    ----------
    east, north:
        Centroid coordinates of 1 km BNG cells in metres (EPSG:27700).
    values:
        Abundance values aligned to east/north.
    hex_gdf:
        Target hex grid (must already have CRS EPSG:27700).
    min_cells:
        Minimum 1 km cells per hex; hexes with fewer are dropped.
    """
    pts = gpd.GeoDataFrame(
        {"value": values},
        geometry=gpd.points_from_xy(east, north),
        crs="EPSG:27700",
    )
    joined = gpd.sjoin(
        pts, hex_gdf.reset_index(drop=True), how="left", predicate="within"
    )
    grouped = joined.groupby("index_right")["value"]
    agg = grouped.median().rename("abundance")
    counts = grouped.count().rename("n_cells")

    out = hex_gdf.copy().reset_index(drop=True)
    out = out.join(agg).join(counts)
    return out[out["n_cells"].fillna(0) >= min_cells]


# ── Trend colour scale ─────────────────────────────────────────────────────


def _truncate_cmap(cmap: mc.Colormap, lo: float, hi: float) -> mc.Colormap:
    return mc.LinearSegmentedColormap.from_list(
        f"{cmap.name}_trunc", cmap(np.linspace(lo, hi, 256))
    )


def _make_trend_scale(
    tau_vals: np.ndarray,
    pct_vmax: float,
) -> tuple[mc.Normalize, mc.Colormap]:
    """Return (norm, cmap) that maximise contrast for signed trend values.

    * All one sign → truncated half of RdBu + plain Normalize so the full
      colorbar width shows only the relevant colours across the real data
      range (no wasted neutral/opposite half).
    * Mixed signs → asymmetric TwoSlopeNorm on the full RdBu so zero stays
      at the visual mid-point but each side scales independently.
    """
    v_lo = float(np.nanpercentile(tau_vals, 100.0 - pct_vmax))
    v_hi = float(np.nanpercentile(tau_vals, pct_vmax))
    base = style.TREND_CMAP  # RdBu: 0.0 = deep red, 0.5 = white, 1.0 = deep blue

    if v_hi <= 0:
        # All declining: map [v_lo, 0] → deep red → white
        return mc.Normalize(vmin=v_lo, vmax=0.0), _truncate_cmap(base, 0.0, 0.5)
    if v_lo >= 0:
        # All increasing: map [0, v_hi] → white → deep blue
        return mc.Normalize(vmin=0.0, vmax=v_hi), _truncate_cmap(base, 0.5, 1.0)
    # Mixed: asymmetric TwoSlopeNorm, zero stays at colour-space midpoint
    return (
        mc.TwoSlopeNorm(vmin=v_lo, vcenter=0.0, vmax=v_hi),
        base,
    )


# ── Figure ─────────────────────────────────────────────────────────────────


def plot_hex_abundance(
    df: pd.DataFrame,
    *,
    title: str,
    species_label: str | None = None,
    abd_col: str = "e_abundance",
    trend_col: str = "abd_ppy",
    hex_size: float = 20_000.0,
    min_cells: int = 3,
    pct_vmax: float = 98.0,
    pct_trend_vmax: float = 95.0,
    gamma: float = 0.5,
    output_path: Path | str | None = None,
    uk_shp: Path | str = _DEFAULT_UK_SHP,
    dpi: int = 300,
) -> plt.Figure:
    """Two-panel hexagonal map: expected abundance and abundance trend.

    Panel A — effort-adjusted expected abundance (viridis, dark=sparse,
    bright=dense).  Panel B — annual abundance trend in % yr⁻¹ (RdBu
    diverging, red=decline, blue=increase); omitted if *trend_col* is absent
    from *df*.

    Both panels share the same hex grid.  1 km BNG cell values are
    aggregated by median into each hexagon, then the hexes are clipped to
    the UK land boundary so coastal edges are clean.

    Parameters
    ----------
    df:
        DataFrame with ``bng_ref``, ``abd_col``, and optionally ``trend_col``.
    title:
        Short species identifier used for logging and fallback label.
    species_label:
        Full human-readable name shown in the figure title.
    abd_col:
        Abundance column (default ``e_abundance``).
    trend_col:
        Trend column in ratio units (default ``abd_ppy``); multiplied by 100
        for display as % yr⁻¹.
    hex_size:
        Circumradius of each hexagon in metres (default 20 km).
    min_cells:
        Minimum 1 km cells required inside a hex for it to be rendered.
    pct_vmax:
        Percentile of hex-level abundance anchoring the colour scale top.
    pct_trend_vmax:
        Percentile of |trend| anchoring the diverging scale (default 97.5).
    gamma:
        Power-norm exponent for the abundance scale (< 1 lifts low values).
    output_path:
        Save path; parent directories are created automatically.
    uk_shp:
        UK boundary shapefile; re-projected to EPSG:27700 if needed.
    dpi:
        Output resolution in dots per inch.
    """
    style.apply()

    # ── UK boundary ────────────────────────────────────────────────────────
    uk = gpd.read_file(uk_shp)
    if uk.crs.to_epsg() != 27700:
        uk = uk.to_crs(epsg=27700)
    uk_union = uk.geometry.union_all()
    bounds = uk.total_bounds  # (minx, miny, maxx, maxy)

    # ── 1 km cell centroids ────────────────────────────────────────────────
    # refs_to_en returns SW corner; +500 m shifts to true centroid.
    east_sw, north_sw = refs_to_en(df["bng_ref"].to_numpy())
    coords_ok = np.isfinite(east_sw) & np.isfinite(north_sw)

    abd_raw = df[abd_col].to_numpy(dtype=np.float64)
    mask_abd = coords_ok & np.isfinite(abd_raw) & (abd_raw >= 0)
    east_a = east_sw[mask_abd] + 500.0
    north_a = north_sw[mask_abd] + 500.0
    abd = abd_raw[mask_abd]

    has_trend = trend_col in df.columns
    if has_trend:
        tau_raw = df[trend_col].to_numpy(dtype=np.float64)
        mask_tau = coords_ok & np.isfinite(tau_raw)
        east_t = east_sw[mask_tau] + 500.0
        north_t = north_sw[mask_tau] + 500.0
        tau = tau_raw[mask_tau] * 100.0  # ratio → % yr⁻¹

    # ── Hex grid → aggregate → clip ────────────────────────────────────────
    logger.info("Building %g km hex grid...", hex_size / 1_000)
    hex_base = _make_hex_grid(bounds, hex_size)

    hex_gdf = _aggregate_to_hex(east_a, north_a, abd, hex_base, min_cells=min_cells)
    logger.info("Clipping %d abundance hexes to UK boundary...", len(hex_gdf))
    hex_gdf["geometry"] = hex_gdf.geometry.intersection(uk_union)
    hex_gdf = hex_gdf[~hex_gdf.geometry.is_empty & hex_gdf["abundance"].notna()]

    if has_trend:
        hex_tau = _aggregate_to_hex(east_t, north_t, tau, hex_base, min_cells=min_cells)
        logger.info("Clipping %d trend hexes to UK boundary...", len(hex_tau))
        hex_tau["geometry"] = hex_tau.geometry.intersection(uk_union)
        hex_tau = hex_tau[~hex_tau.geometry.is_empty & hex_tau["abundance"].notna()]

    # ── Colour scales ──────────────────────────────────────────────────────
    abd_vals = hex_gdf["abundance"].to_numpy(dtype=np.float64)
    abd_vmin = float(np.nanpercentile(abd_vals, 1.0))
    abd_vmax = float(np.nanpercentile(abd_vals, pct_vmax))
    abd_norm = mc.PowerNorm(gamma=gamma, vmin=abd_vmin, vmax=abd_vmax)
    n_clip_abd = int(((abd_vals > abd_vmax) | (abd_vals < abd_vmin)).sum())
    if n_clip_abd:
        logger.warning(
            "[%s] Abundance colour scale clipped: %d hex%s (%.1f%%) outside "
            "[%.3f, %.3f] — shown as end-of-scale colour. "
            "Pass --pct-vmax to widen the range.",
            title, n_clip_abd, "es" if n_clip_abd != 1 else "",
            100.0 * n_clip_abd / len(abd_vals), abd_vmin, abd_vmax,
        )

    if has_trend:
        tau_vals = hex_tau["abundance"].to_numpy(dtype=np.float64)
        tau_norm, tau_cmap = _make_trend_scale(tau_vals, pct_trend_vmax)
        tau_lo = float(np.nanpercentile(tau_vals, 100.0 - pct_trend_vmax))
        tau_hi = float(np.nanpercentile(tau_vals, pct_trend_vmax))
        n_clip_tau = int(((tau_vals > tau_hi) | (tau_vals < tau_lo)).sum())
        if n_clip_tau:
            logger.warning(
                "[%s] Trend colour scale clipped: %d hex%s (%.1f%%) outside "
                "[%.2f, %.2f]%% yr⁻¹ — shown as end-of-scale colour. "
                "Pass --pct-trend-vmax to widen the range.",
                title, n_clip_tau, "es" if n_clip_tau != 1 else "",
                100.0 * n_clip_tau / len(tau_vals), tau_lo, tau_hi,
            )

    # ── Layout ─────────────────────────────────────────────────────────────
    ncols = 2 if has_trend else 1
    panel_w = 5.4
    panel_h = 7.95
    gap = 0.44
    top_margin = 1.08
    cbar_h = 0.22
    cbar_gap = 0.16
    foot_h = 0.22
    foot_gap = 0.10
    bot_margin = 0.34
    side_margin = 0.34

    total_w = side_margin + ncols * panel_w + (ncols - 1) * gap + side_margin
    total_h = top_margin + panel_h + cbar_gap + cbar_h + foot_gap + foot_h + bot_margin

    fig = plt.figure(figsize=(total_w, total_h), facecolor=style.PAPER)

    map_axes, cbar_axes = [], []
    for i in range(ncols):
        ax_x = (side_margin + i * (panel_w + gap)) / total_w
        ax_y = (bot_margin + foot_h + foot_gap + cbar_h + cbar_gap) / total_h
        map_axes.append(
            fig.add_axes([ax_x, ax_y, panel_w / total_w, panel_h / total_h])
        )
        cbar_axes.append(
            fig.add_axes(
                [
                    ax_x + 0.04 * panel_w / total_w,
                    (bot_margin + foot_h + foot_gap) / total_h,
                    0.92 * panel_w / total_w,
                    cbar_h / total_h,
                ]
            )
        )

    foot_ax = fig.add_axes(
        [
            side_margin / total_w,
            bot_margin / total_h,
            (total_w - 2 * side_margin) / total_w,
            foot_h / total_h,
        ]
    )
    foot_ax.set_axis_off()

    # ── Map panels ─────────────────────────────────────────────────────────
    margin = hex_size * 0.6
    xlim = (bounds[0] - margin, bounds[2] + margin)
    ylim = (bounds[1] - margin, bounds[3] + margin)

    panels = [
        (
            map_axes[0],
            cbar_axes[0],
            hex_gdf,
            abd_norm,
            _HEX_CMAP,
            "Expected abundance",
            "E[abundance] = P(occ) × E[count | occ]  (individuals per visit)",
            False,
        ),
    ]
    if has_trend:
        panels.append(
            (
                map_axes[1],
                cbar_axes[1],
                hex_tau,
                tau_norm,
                tau_cmap,
                "Annual abundance trend",
                "Annual change  (% yr⁻¹)",
                False,
            )
        )

    for ax, cax, gdf, norm, cmap, panel_title, cbar_label, symmetric_cb in panels:
        ax.set_facecolor(style.WATER)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_aspect("equal")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        uk.plot(ax=ax, color=_NO_DATA, linewidth=0, zorder=1)
        gdf.plot(
            ax=ax,
            column="abundance",
            norm=norm,
            cmap=cmap,
            linewidth=0,
            edgecolor="none",
            zorder=3,
            legend=False,
        )
        uk.boundary.plot(ax=ax, color=style.COAST, linewidth=0.45, zorder=5)

        ax.text(
            0.02,
            0.98,
            panel_title,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.22,rounding_size=0.14",
                "facecolor": style.PAPER,
                "edgecolor": "none",
                "alpha": 0.95,
            },
            **style.FONT_SECTION,
        )

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.outline.set_visible(False)
        cb.ax.tick_params(
            length=2.5, width=0.5, labelsize=7.5, colors=style.TEXT_MUTED, pad=2
        )
        cb.ax.xaxis.set_ticks_position("bottom")
        cb.ax.set_xlabel(cbar_label, **style.FONT_LABEL, labelpad=4)
        cb.locator = mpl.ticker.MaxNLocator(nbins=5, symmetric=symmetric_cb)
        cb.update_ticks()
        cax.set_facecolor(style.PAPER)
        for spine in cax.spines.values():
            spine.set_visible(False)

    # "No data" legend on the first panel only
    leg = map_axes[0].legend(
        handles=[
            Patch(facecolor=_NO_DATA, edgecolor="none", label="No survey coverage")
        ],
        loc="lower left",
        frameon=True,
        facecolor=style.PAPER,
        edgecolor=style.GRIDLINE,
        framealpha=0.92,
        fontsize=7.5,
        labelcolor=style.TEXT_MUTED,
        handlelength=1.2,
        handleheight=1.0,
        borderpad=0.6,
    )
    leg.set_zorder(6)

    # ── Title block ────────────────────────────────────────────────────────
    sp_label = species_label or title.replace("_", " ").title()
    fig.text(0.06, 0.972, sp_label, ha="left", va="top", **style.FONT_TITLE)
    subtitle = (
        f"Effort-adjusted abundance and annual trend  ·  {hex_size / 1_000:.0f} km hexagonal grid"
        if has_trend
        else f"Effort-adjusted expected abundance  ·  {hex_size / 1_000:.0f} km hexagonal grid"
    )
    fig.text(0.06, 0.944, subtitle, ha="left", va="top", **style.FONT_SUBTITLE)
    fig.add_artist(
        mpl.lines.Line2D(
            [0.06, 0.94],
            [0.928, 0.928],
            transform=fig.transFigure,
            color=style.GRIDLINE,
            linewidth=0.8,
        )
    )

    # ── Footer ─────────────────────────────────────────────────────────────
    n_hex = len(hex_gdf)
    median_abd = float(hex_gdf["abundance"].median())
    parts = [f"{n_hex} hexagonal cells", f"median E[abundance] {median_abd:.3f}"]
    if has_trend:
        median_tau = float(hex_tau["abundance"].median())
        parts.append(f"grid median trend {median_tau:+.2f}% yr⁻¹")
    foot_ax.text(
        1.0, 0.5, "  ·  ".join(parts), ha="right", va="center", **style.FONT_ANNOTATION
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, facecolor=style.PAPER)
        logger.info("Saved → %s", output_path)

    return fig


# ── CLI ────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--results",
        default="output/prototype",
        help="directory with species parquet files (default: output/prototype)",
    )
    p.add_argument(
        "--species",
        nargs="+",
        default=None,
        help="species to map; defaults to all found",
    )
    p.add_argument(
        "--hex-size",
        type=float,
        default=3_000.0,
        dest="hex_size",
        help="hex circumradius in metres (default: 3000 = 3 km diameter ~6 km)",
    )
    p.add_argument(
        "--min-cells",
        type=int,
        default=3,
        dest="min_cells",
        help="minimum 1 km cells per hex to render it (default: 3)",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="power-norm exponent for colour stretch; <1 lifts low values (default: 0.5)",
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--pct-vmax",
        type=float,
        default=99.0,
        dest="pct_vmax",
        help="abundance percentile anchoring the top of the colour scale (default: 99)",
    )
    p.add_argument(
        "--uk-shp",
        default=None,
        help="UK boundary shapefile; uses built-in default if omitted",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    results_dir = Path(args.results)
    maps_dir = results_dir / "figures" / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    hierarchical = sorted(results_dir.glob("predictions/species/*/prediction.parquet"))
    if hierarchical:
        found = {p.parent.name: p for p in hierarchical}
    else:
        flat = sorted(results_dir.glob("species_*.parquet"))
        found = {p.stem.removeprefix("species_"): p for p in flat}

    if not found:
        logger.error(
            "No species parquet files found in %s "
            "(tried predictions/species/*/prediction.parquet and species_*.parquet)",
            results_dir,
        )
        sys.exit(1)

    species_list = args.species or list(found.keys())
    missing = [s for s in species_list if s not in found]
    if missing:
        logger.warning("No parquet found for: %s", ", ".join(missing))
    species_list = [s for s in species_list if s in found]
    if not species_list:
        logger.error("No valid species to map.")
        sys.exit(1)

    shp_path = args.uk_shp or _DEFAULT_UK_SHP
    logger.info("Mapping %d species → %s", len(species_list), maps_dir)

    for sp in species_list:
        df = pd.read_parquet(found[sp])
        label = _COMMON_NAMES.get(sp, sp.replace("_", " ").title())
        out_dir = maps_dir / sp
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "hex_abundance.png"
        logger.info("[%s] hex map → %s", sp, out_path)
        plot_hex_abundance(
            df,
            title=sp,
            species_label=label,
            hex_size=args.hex_size,
            min_cells=args.min_cells,
            gamma=args.gamma,
            pct_vmax=args.pct_vmax,
            output_path=out_path,
            uk_shp=shp_path,
            dpi=args.dpi,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
