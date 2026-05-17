#!/usr/bin/env python3
"""Occurrence-count decoupling map for BGBW model output.

For each species, renders a three-panel hexagonal map:

  Panel A — spatial trend in occurrence probability  (τ_occ, % yr⁻¹)
  Panel B — spatial trend in conditional count       (τ_count, % yr⁻¹)
  Panel C — bivariate choropleth: (τ_occ, τ_count) simultaneously

The bivariate panel encodes both components in a single 2-D colour space:
  (−, −) deep red   — both declining (coupled loss)
  (+, +) teal-green — both increasing (coupled gain)
  (+, −) amber      — range expanding but count thinning (decoupled)
  (−, +) purple     — range contracting but count densifying (decoupled)
Regions on the red–green diagonal are coupled; off-diagonal cells are
decoupled. A small legend square is inset in panel C.

Usage
-----
    uv run python scripts/products/make_decoupling_map.py
    uv run python scripts/products/make_decoupling_map.py --species collared_dove
    uv run python scripts/products/make_decoupling_map.py --hex-size 10000 --dpi 150

Output
------
    <results_dir>/figures/maps/<species>/hex_decoupling.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import yaml
import matplotlib as mpl
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from bgbw_trends.viz import style
from bgbw_trends.viz._bng import refs_to_en


# ── Oklab helpers (minimal copy from bgbw_trends.viz.maps) ────────────────


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float64)
    return np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float64)
    return np.where(
        arr <= 0.0031308, 12.92 * arr, 1.055 * np.power(arr, 1.0 / 2.4) - 0.055
    )


def _rgb_to_oklab(hex_or_rgb) -> np.ndarray:
    r_lin, g_lin, b_lin = _srgb_to_linear(
        np.asarray(mc.to_rgb(hex_or_rgb), dtype=np.float64)
    )
    l = 0.4122214708 * r_lin + 0.5363325363 * g_lin + 0.0514459929 * b_lin
    m = 0.2119034982 * r_lin + 0.6806995451 * g_lin + 0.1073969566 * b_lin
    s = 0.0883024619 * r_lin + 0.2817188376 * g_lin + 0.6299787005 * b_lin
    l_c, m_c, s_c = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    return np.array(
        [
            0.2104542553 * l_c + 0.7936177850 * m_c - 0.0040720468 * s_c,
            1.9779984951 * l_c - 2.4285922050 * m_c + 0.4505937099 * s_c,
            0.0259040371 * l_c + 0.7827717662 * m_c - 0.8086757660 * s_c,
        ],
        dtype=np.float64,
    )


def _oklab_to_hex(lab: np.ndarray) -> str:
    L, a, b = np.asarray(lab, dtype=np.float64)
    l_c = L + 0.3963377774 * a + 0.2158037573 * b
    m_c = L - 0.1055613458 * a - 0.0638541728 * b
    s_c = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_c**3, m_c**3, s_c**3
    rgb_lin = np.array(
        [
            +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
        ],
        dtype=np.float64,
    )
    rgb = np.clip(_linear_to_srgb(np.clip(rgb_lin, 0.0, None)), 0.0, 1.0)
    return mc.to_hex(tuple(rgb))


# ── Bivariate colour palettes (bivario corner convention) ─────────────────
# Each palette is defined by four corner colours:
#   low      = bottom-left  (weak occ,    weak count)
#   accent_a = top-left     (weak occ,    strong count)
#   accent_b = bottom-right (strong occ,  weak count)
#   high     = top-right    (strong occ,  strong count)
# x-axis = τ_occ (left→right: weak→strong)
# y-axis = τ_count (bottom→top: weak→strong)
_BIVARIATE_PALETTES: dict[str, dict[str, tuple]] = {
    "electric_neon": {
        "low": (0.99, 0.97, 0.88),
        "accent_a": (0.12, 0.86, 0.78),
        "accent_b": (0.94, 0.30, 0.34),
        "high": (0.14, 0.16, 0.32),
    },
    "blade_runner": {
        "low": (0.99, 0.97, 0.86),
        "accent_a": (0.10, 0.60, 0.70),
        "accent_b": (0.96, 0.50, 0.18),
        "high": (0.14, 0.16, 0.34),
    },
    "sunrise_harbor": {
        "low": (0.98, 0.94, 0.60),
        "accent_a": (0.95, 0.40, 0.20),
        "accent_b": (0.10, 0.62, 0.65),
        "high": (0.12, 0.18, 0.50),
    },
    "kaleidoscope": {
        "low": (0.98, 0.72, 0.20),
        "accent_a": (0.12, 0.96, 0.82),
        "accent_b": (0.96, 0.28, 0.54),
        "high": (0.46, 0.30, 0.96),
    },
}
_DEFAULT_PALETTE = "blade_runner"

TAU_OCC_LABEL = r"$\tau_{\mathrm{occ}}$"
TAU_COUNT_LABEL = r"$\tau_{\mathrm{count}}$"


def _get_palette_corners(
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (c00, c10, c01, c11) Oklab arrays for the named palette.

    low/high are swapped so that the dark 'high' colour sits at the
    bottom-left (strong declines) and the light 'low' colour sits at the
    top-right (strong increases).
    """
    p = _BIVARIATE_PALETTES[name]
    return (
        _rgb_to_oklab(p["high"]),  # c00: bottom-left  (strong decline, strong decline)
        _rgb_to_oklab(p["accent_a"]),  # c10: top-left     (weak occ,   strong count)
        _rgb_to_oklab(p["accent_b"]),  # c01: bottom-right (strong occ, weak count)
        _rgb_to_oklab(p["low"]),  # c11: top-right    (weak decline, weak decline)
    )


def _make_occ_count_palette(
    n: int,
    *,
    c00: np.ndarray,
    c10: np.ndarray,
    c01: np.ndarray,
    c11: np.ndarray,
) -> list[list[str]]:
    """n×n palette; row i = τ_count bin (low→high), col j = τ_occ bin.

    Grid layout when rendered with imshow(origin='lower'):
      bottom-left  palette[0][0]     = c00  (weak count, weak occ)
      top-left     palette[n-1][0]   = c10  (strong count, weak occ)
      bottom-right palette[0][n-1]   = c01  (weak count, strong occ)
      top-right    palette[n-1][n-1] = c11  (strong count, strong occ)

    Hex colour lookup: palette[count_bin][occ_bin]
    """
    pal: list[list[str]] = []
    for i in range(n):
        a = i / max(n - 1, 1)
        row: list[str] = []
        for j in range(n):
            t = j / max(n - 1, 1)
            lab = (
                (1 - a) * (1 - t) * c00
                + a * (1 - t) * c10
                + (1 - a) * t * c01
                + a * t * c11
            )
            row.append(_oklab_to_hex(lab))
        pal.append(row)
    return pal


def _marginal_colors(pal: list[list[str]], n: int, axis: str) -> list[str]:
    """Average palette colours along the other axis to get n marginal colours.

    axis='occ'   → one colour per occ bin (col j), averaged over count rows.
    axis='count' → one colour per count bin (row i), averaged over occ cols.
    """
    colors: list[str] = []
    for i in range(n):
        if axis == "occ":
            rgbs = [np.array(mc.to_rgba(pal[j][i])[:3]) for j in range(n)]
        else:
            rgbs = [np.array(mc.to_rgba(pal[i][j])[:3]) for j in range(n)]
        colors.append(mc.to_hex(np.clip(np.mean(rgbs, axis=0), 0.0, 1.0)))
    return colors


def _full_range_norm(arr: np.ndarray, pct: float) -> np.ndarray:
    """Map values to [-1, 1] using each variable's own low/high range.

    This guarantees that each axis uses the full colour span available,
    even if all values are mostly on one side of zero.
    """
    lo = float(np.nanpercentile(arr, 100.0 - pct))
    hi = float(np.nanpercentile(arr, pct))
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float64)
    scaled = 2.0 * ((arr - lo) / (hi - lo)) - 1.0
    return np.clip(scaled, -1.0, 1.0)


def _norm_to_bins(vals: np.ndarray, n: int) -> np.ndarray:
    """Convert normalised values in [-1,1] to 0-based bin indices."""
    idx = np.floor((np.clip(vals, -1.0, 1.0) + 1.0) * 0.5 * n).astype(int)
    return np.clip(idx, 0, n - 1)


def _axis_legend_text(values: np.ndarray, *, kind: str) -> tuple[str, str, str]:
    """Return axis title plus low/high endpoint labels for the legend."""
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if kind == "occ":
        if hi <= 0.0:
            return "occurrence decline", "stronger", "weaker"
        if lo >= 0.0:
            return "occurrence increase", "weaker", "stronger"
        return "occurrence trend", "less", "more"
    # kind == "count"
    if hi <= 0.0:
        return "count decline", "stronger", "weaker"
    if lo >= 0.0:
        return "count increase", "weaker", "stronger"
    return "count trend", "fewer", "more"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_NO_DATA = "#9C9590"

_DEFAULT_UK_SHP = Path(
    "/home/nilomr/projects/bgbw.metadata.pipeline"
    "/data/metadata/uk-shapefile/UK-shapefile.shp"
)

_COMMON_NAMES: dict[str, str] = {
    "blackbird": "Common Blackbird  —  Turdus merula",
    "blue_tit": "Blue Tit  —  Cyanistes caeruleus",
    "collared_dove": "Eurasian Collared Dove  —  Streptopelia decaocto",
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
    minx, miny, maxx, maxy = bounds
    col_step = hex_size * np.sqrt(3)
    row_step = hex_size * 1.5
    half_col = col_step / 2.0

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
    """Spatial-join 1 km cell centroids into hexes; summarise by median."""
    pts = gpd.GeoDataFrame(
        {"value": values},
        geometry=gpd.points_from_xy(east, north),
        crs="EPSG:27700",
    )
    joined = gpd.sjoin(
        pts, hex_gdf.reset_index(drop=True), how="left", predicate="within"
    )
    grouped = joined.groupby("index_right")["value"]
    agg = grouped.median().rename("val")
    counts = grouped.count().rename("n_cells")

    out = hex_gdf.copy().reset_index(drop=True)
    out = out.join(agg).join(counts)
    return out[out["n_cells"].fillna(0) >= min_cells]


# ── Colour scale ───────────────────────────────────────────────────────────


def _truncate_cmap(cmap: mc.Colormap, lo: float, hi: float) -> mc.Colormap:
    return mc.LinearSegmentedColormap.from_list(
        f"{cmap.name}_trunc", cmap(np.linspace(lo, hi, 256))
    )


def _make_trend_scale(
    vals: np.ndarray,
    pct_vmax: float,
) -> tuple[mc.Normalize, mc.Colormap]:
    """Diverging RdBu scale centred at zero; handles all-one-sign data."""
    v_lo = float(np.nanpercentile(vals, 100.0 - pct_vmax))
    v_hi = float(np.nanpercentile(vals, pct_vmax))
    base = style.TREND_CMAP  # RdBu: 0=deep red, 0.5=white, 1=deep blue

    if v_hi <= 0:
        return mc.Normalize(vmin=v_lo, vmax=0.0), _truncate_cmap(base, 0.0, 0.5)
    if v_lo >= 0:
        return mc.Normalize(vmin=0.0, vmax=v_hi), _truncate_cmap(base, 0.5, 1.0)
    return mc.TwoSlopeNorm(vmin=v_lo, vcenter=0.0, vmax=v_hi), base


# ── Shared data preparation ───────────────────────────────────────────────────


def _common_map_setup(
    df: pd.DataFrame,
    *,
    occ_col: str,
    count_col: str,
    hex_size: float,
    min_cells: int,
    uk_shp: Path | str,
) -> tuple[gpd.GeoDataFrame, np.ndarray, gpd.GeoDataFrame]:
    """Load UK boundary, build hex grid, aggregate 1 km cells to hexagons.

    Returns *(uk_gdf, total_bounds, hex_joint)* where *hex_joint* has columns
    ``occ_val``, ``count_val``, ``n_cells``, and UK-clipped geometry.
    """
    uk = gpd.read_file(uk_shp)
    if uk.crs.to_epsg() != 27700:
        uk = uk.to_crs(epsg=27700)
    uk_union = uk.geometry.union_all()
    bounds = uk.total_bounds

    east_sw, north_sw = refs_to_en(df["bng_ref"].to_numpy())
    coords_ok = np.isfinite(east_sw) & np.isfinite(north_sw)
    occ_raw = df[occ_col].to_numpy(dtype=np.float64)
    count_raw = df[count_col].to_numpy(dtype=np.float64)
    mask = coords_ok & np.isfinite(occ_raw) & np.isfinite(count_raw)

    east = east_sw[mask] + 500.0
    north = north_sw[mask] + 500.0
    occ_vals = occ_raw[mask] * 100.0  # ratio → % yr⁻¹
    count_vals = count_raw[mask] * 100.0

    logger.info("Building %g km hex grid...", hex_size / 1_000)
    hex_base = _make_hex_grid(bounds, hex_size)

    pts = gpd.GeoDataFrame(
        {"occ": occ_vals, "count": count_vals},
        geometry=gpd.points_from_xy(east, north),
        crs="EPSG:27700",
    )
    joined = gpd.sjoin(
        pts, hex_base.reset_index(drop=True), how="left", predicate="within"
    )
    grp = joined.groupby("index_right")
    hex_joint = hex_base.copy().reset_index(drop=True)
    hex_joint["occ_val"] = grp["occ"].median()
    hex_joint["count_val"] = grp["count"].median()
    hex_joint["n_cells"] = grp["occ"].count()
    hex_joint = hex_joint[hex_joint["n_cells"].fillna(0) >= min_cells].copy()
    hex_joint["geometry"] = hex_joint.geometry.intersection(uk_union)
    hex_joint = hex_joint[
        ~hex_joint.geometry.is_empty
        & hex_joint["occ_val"].notna()
        & hex_joint["count_val"].notna()
    ].copy()
    logger.info("Hexes after clip: %d", len(hex_joint))
    return uk, bounds, hex_joint


# ── Figure ─────────────────────────────────────────────────────────────────


def plot_trend_panels(
    df: pd.DataFrame,
    *,
    title: str,
    species_label: str | None = None,
    occ_col: str = "occ_ppy",
    count_col: str = "count_ppy",
    hex_size: float = 20_000.0,
    min_cells: int = 3,
    pct_vmax: float = 97.5,
    output_path: Path | str | None = None,
    uk_shp: Path | str = _DEFAULT_UK_SHP,
    dpi: int = 300,
) -> plt.Figure:
    """Two-panel hexagonal map: τ_occ (left) and τ_count (right)."""
    style.apply()
    uk, bounds, hex_joint = _common_map_setup(
        df,
        occ_col=occ_col,
        count_col=count_col,
        hex_size=hex_size,
        min_cells=min_cells,
        uk_shp=uk_shp,
    )
    hex_occ = hex_joint[["geometry", "occ_val"]].rename(columns={"occ_val": "val"})
    hex_count = hex_joint[["geometry", "count_val"]].rename(
        columns={"count_val": "val"}
    )
    occ_norm, occ_cmap = _make_trend_scale(hex_occ["val"].to_numpy(), pct_vmax)
    count_norm, count_cmap = _make_trend_scale(hex_count["val"].to_numpy(), pct_vmax)

    # ── Layout ────────────────────────────────────────────────────────────
    ncols = 2
    panel_w, panel_h = 4.6, 7.2
    gap, top_margin = 0.38, 1.08
    cbar_h, cbar_gap = 0.22, 0.18
    foot_h, foot_gap = 0.22, 0.10
    bot_margin = side_margin = 0.36

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

    uni_panels = [
        (
            map_axes[0],
            cbar_axes[0],
            hex_occ,
            occ_norm,
            occ_cmap,
            "Occurrence trend  τ_occ",
            "τ_occ(W)  (% yr⁻¹)",
        ),
        (
            map_axes[1],
            cbar_axes[1],
            hex_count,
            count_norm,
            count_cmap,
            "Count trend  τ_count",
            "τ_count(W)  (% yr⁻¹)",
        ),
    ]
    for ax, cax, gdf, norm, cmap, panel_title, cbar_label in uni_panels:
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
            column="val",
            norm=norm,
            cmap=cmap,
            linewidth=0,
            edgecolor="none",
            alpha=1.0,
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
        cb.locator = mpl.ticker.MaxNLocator(
            nbins=5, symmetric=isinstance(norm, mc.TwoSlopeNorm)
        )
        cb.update_ticks()
        cax.set_facecolor(style.PAPER)
        for spine in cax.spines.values():
            spine.set_visible(False)

    leg = map_axes[0].legend(
        handles=[
            Patch(facecolor=_NO_DATA, edgecolor="none", label="No survey coverage")
        ],
        loc="lower left",
        frameon=True,
        facecolor=style.PAPER,
        edgecolor="none",
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
    fig.text(0.04, 0.972, sp_label, ha="left", va="top", **style.FONT_TITLE)
    fig.text(
        0.04,
        0.944,
        f"Spatial trends in occurrence and conditional count"
        f"  ·  {hex_size / 1_000:.0f} km hexagonal grid",
        ha="left",
        va="top",
        **style.FONT_SUBTITLE,
    )
    fig.add_artist(
        mpl.lines.Line2D(
            [0.04, 0.96],
            [0.928, 0.928],
            transform=fig.transFigure,
            color=style.GRIDLINE,
            linewidth=0.8,
        )
    )

    # ── Footer ─────────────────────────────────────────────────────────────
    med_occ = float(hex_joint["occ_val"].median())
    med_count = float(hex_joint["count_val"].median())
    foot_ax.text(
        1.0,
        0.5,
        f"median τ_occ {med_occ:+.2f}% yr⁻¹  ·  "
        f"median τ_count {med_count:+.2f}% yr⁻¹  ·  "
        f"hexes n = {len(hex_joint):,}",
        ha="right",
        va="center",
        **style.FONT_ANNOTATION,
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, facecolor=style.PAPER, bbox_inches="tight")
        logger.info("Saved → %s", output_path)
    return fig


def plot_bivariate_map(
    df: pd.DataFrame,
    *,
    title: str,
    species_label: str | None = None,
    occ_col: str = "occ_ppy",
    count_col: str = "count_ppy",
    hex_size: float = 20_000.0,
    min_cells: int = 3,
    pct_vmax: float = 97.5,
    n_classes: int = 5,
    palette: str = _DEFAULT_PALETTE,
    output_path: Path | str | None = None,
    uk_shp: Path | str = _DEFAULT_UK_SHP,
    dpi: int = 300,
) -> plt.Figure:
    """Standalone bivariate decoupling map with dedicated colour legend.

    The legend is a square n×n grid placed to the right of the map.  Corner
    cells carry pure anchor colours (see ``_BV_C00`` … ``_BV_C11``); interior
    cells are bilinear Oklab blends.  Low/high tick labels, axis titles, and
    an interpretation caption are rendered outside the grid.
    """
    style.apply()
    uk, bounds, hex_joint = _common_map_setup(
        df,
        occ_col=occ_col,
        count_col=count_col,
        hex_size=hex_size,
        min_cells=min_cells,
        uk_shp=uk_shp,
    )

    # ── Bivariate palette and bin assignment ──────────────────────────────
    c00, c10, c01, c11 = _get_palette_corners(palette)
    pal = _make_occ_count_palette(n_classes, c00=c00, c10=c10, c01=c01, c11=c11)
    occ_norm = _full_range_norm(
        hex_joint["occ_val"].to_numpy(dtype=np.float64), pct_vmax
    )
    count_norm = _full_range_norm(
        hex_joint["count_val"].to_numpy(dtype=np.float64), pct_vmax
    )
    occ_axis_title, occ_low_label, occ_high_label = _axis_legend_text(
        hex_joint["occ_val"].to_numpy(dtype=np.float64), kind="occ"
    )
    count_axis_title, count_low_label, count_high_label = _axis_legend_text(
        hex_joint["count_val"].to_numpy(dtype=np.float64), kind="count"
    )
    occ_bins = _norm_to_bins(occ_norm, n_classes)
    count_bins = _norm_to_bins(count_norm, n_classes)
    hex_joint = hex_joint.copy()
    # palette[count_bin][occ_bin]: row=count (y-axis), col=occ (x-axis)
    hex_joint["bv_color"] = [
        pal[int(cb)][int(ob)] for ob, cb in zip(occ_bins, count_bins)
    ]

    # ── Marginal histogram data ───────────────────────────────────────────
    occ_vals_raw = hex_joint["occ_val"].to_numpy(dtype=np.float64)
    count_vals_raw = hex_joint["count_val"].to_numpy(dtype=np.float64)
    # Shared x range so both histograms are on the same axis (comparable magnitude)
    hist_all = np.concatenate(
        [
            occ_vals_raw[np.isfinite(occ_vals_raw)],
            count_vals_raw[np.isfinite(count_vals_raw)],
        ]
    )
    hist_xmin = float(np.nanpercentile(hist_all, 0.5))
    hist_xmax = float(np.nanpercentile(hist_all, 99.5))
    hist_nbins = 35
    hist_edges = np.linspace(hist_xmin, hist_xmax, hist_nbins + 1)
    hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2.0
    # Per-variable normalization bounds (matching _full_range_norm)
    occ_pct_lo = float(np.nanpercentile(occ_vals_raw, 100.0 - pct_vmax))
    occ_pct_hi = float(np.nanpercentile(occ_vals_raw, pct_vmax))
    count_pct_lo = float(np.nanpercentile(count_vals_raw, 100.0 - pct_vmax))
    count_pct_hi = float(np.nanpercentile(count_vals_raw, pct_vmax))
    occ_marg = _marginal_colors(pal, n_classes, "occ")
    count_marg = _marginal_colors(pal, n_classes, "count")

    def _hist_color(v: float, lo: float, hi: float, marg: list) -> str:
        if hi <= lo:
            return marg[n_classes // 2]
        nv = np.clip(2.0 * (v - lo) / (hi - lo) - 1.0, -1.0, 1.0)
        idx = int(
            np.clip(int(np.floor((nv + 1.0) * 0.5 * n_classes)), 0, n_classes - 1)
        )
        return marg[idx]

    occ_counts, _ = np.histogram(
        occ_vals_raw[np.isfinite(occ_vals_raw)], bins=hist_edges
    )
    count_counts, _ = np.histogram(
        count_vals_raw[np.isfinite(count_vals_raw)], bins=hist_edges
    )
    occ_bar_colors = [
        _hist_color(c, occ_pct_lo, occ_pct_hi, occ_marg) for c in hist_centers
    ]
    count_bar_colors = [
        _hist_color(c, count_pct_lo, count_pct_hi, count_marg) for c in hist_centers
    ]

    # ── Layout ────────────────────────────────────────────────────────────
    # Match the map panel's box aspect to the UK bounds to avoid dead space.
    margin = hex_size * 0.6
    xlim = (bounds[0] - margin, bounds[2] + margin)
    ylim = (bounds[1] - margin, bounds[3] + margin)
    map_aspect = (ylim[1] - ylim[0]) / max(xlim[1] - xlim[0], 1.0)

    side_left = 0.30
    panel_w = 4.15
    gap = 0.28
    leg_grid_in = 1.46
    leg_left_pad = 0.46
    leg_right_pad = 0.18
    side_right = 0.12
    top_margin = 0.92
    panel_h = panel_w * map_aspect
    foot_gap = 0.20
    foot_h = 0.18
    bot_margin = 0.18

    total_w = (
        side_left
        + panel_w
        + gap
        + leg_left_pad
        + leg_grid_in
        + leg_right_pad
        + side_right
    )
    total_h = top_margin + panel_h + foot_gap + foot_h + bot_margin

    fig = plt.figure(figsize=(total_w, total_h), facecolor=style.PAPER)

    # Map axes
    map_x0 = side_left / total_w
    map_y0 = (bot_margin + foot_h + foot_gap) / total_h
    ax_map = fig.add_axes([map_x0, map_y0, panel_w / total_w, panel_h / total_h])

    # Legend grid axes — physically square: leg_grid_in" × leg_grid_in"
    leg_grid_x0 = (side_left + panel_w + gap + leg_left_pad) / total_w
    leg_grid_w_frac = leg_grid_in / total_w
    leg_grid_h_frac = leg_grid_in / total_h
    map_top_y = map_y0 + panel_h / total_h
    leg_grid_y0 = map_top_y - leg_grid_h_frac - 0.10
    ax_leg = fig.add_axes([leg_grid_x0, leg_grid_y0, leg_grid_w_frac, leg_grid_h_frac])
    ax_leg.set_zorder(3)

    # Footer axes (spans full content width)
    foot_ax = fig.add_axes(
        [
            side_left / total_w,
            bot_margin / total_h,
            (total_w - side_left - side_right) / total_w,
            foot_h / total_h,
        ]
    )
    foot_ax.set_axis_off()

    # Marginal histogram axes (stacked below the legend grid, right column)
    hist_each_h_in = 0.40
    hist_spacing_in = 0.06
    hist_block_gap_in = 0.48
    leg_grid_y0_in = leg_grid_y0 * total_h
    hist_occ_top_in = leg_grid_y0_in - hist_block_gap_in
    hist_occ_bot_in = hist_occ_top_in - hist_each_h_in
    hist_count_top_in = hist_occ_bot_in - hist_spacing_in
    hist_count_bot_in = hist_count_top_in - hist_each_h_in
    hist_h_frac = hist_each_h_in / total_h
    ax_hist_occ = fig.add_axes(
        [leg_grid_x0, hist_occ_bot_in / total_h, leg_grid_w_frac, hist_h_frac]
    )
    ax_hist_count = fig.add_axes(
        [leg_grid_x0, hist_count_bot_in / total_h, leg_grid_w_frac, hist_h_frac]
    )

    # ── Map ───────────────────────────────────────────────────────────────
    ax_map.set_facecolor(style.WATER)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(False)
    ax_map.set_aspect("equal")
    ax_map.set_box_aspect(map_aspect)
    ax_map.set_xlim(*xlim)
    ax_map.set_ylim(*ylim)

    uk.plot(ax=ax_map, color=_NO_DATA, linewidth=0, zorder=1)
    hex_joint.plot(
        ax=ax_map,
        color=hex_joint["bv_color"],
        linewidth=0,
        edgecolor="none",
        alpha=1.0,
        zorder=3,
    )
    uk.boundary.plot(ax=ax_map, color=style.COAST, linewidth=0.45, zorder=5)

    # ── Bivariate legend ──────────────────────────────────────────────────
    n = n_classes
    n1 = n - 1

    colors_grid = np.array(
        [[mc.to_rgba(pal[row][col]) for col in range(n)] for row in range(n)],
        dtype=np.float32,
    )
    ax_leg.set_facecolor(style.PAPER)
    ax_leg.imshow(
        colors_grid,
        origin="lower",
        interpolation="nearest",
        extent=[-0.5, n - 0.5, -0.5, n - 0.5],
        aspect="auto",
        zorder=2,
    )

    # Fine cell separators (thin)
    for k in np.arange(0.5, n - 0.5):
        ax_leg.axhline(k, color="white", linewidth=0.6, zorder=3)
        ax_leg.axvline(k, color="white", linewidth=0.6, zorder=3)

    ax_leg.set_xlim(-0.5, n - 0.5)
    ax_leg.set_ylim(-0.5, n - 0.5)

    # Axis ticks are hidden; all explanatory text is placed outside the grid.
    ax_leg.set_xticks([])
    ax_leg.set_yticks([])
    ax_leg.tick_params(length=0)
    for sp in ax_leg.spines.values():
        sp.set_visible(False)

    leg_left = leg_grid_x0
    leg_right = leg_grid_x0 + leg_grid_w_frac
    leg_bottom = leg_grid_y0
    leg_top = leg_grid_y0 + leg_grid_h_frac
    leg_cx = (leg_left + leg_right) / 2.0
    leg_cy = (leg_bottom + leg_top) / 2.0

    # No-data swatch placed in legend block (right column, above colour grid)
    nodata_swatch_x = leg_left
    nodata_swatch_y = leg_top + 0.016
    swatch_w_fig = 0.018
    swatch_h_fig = 0.010
    fig.add_artist(
        Rectangle(
            (nodata_swatch_x, nodata_swatch_y),
            swatch_w_fig,
            swatch_h_fig,
            transform=fig.transFigure,
            facecolor=_NO_DATA,
            edgecolor="none",
            zorder=4,
        )
    )
    fig.text(
        nodata_swatch_x + swatch_w_fig + 0.006,
        nodata_swatch_y + swatch_h_fig / 2.0,
        "No survey coverage",
        ha="left",
        va="center",
        fontsize=6.8,
        color=style.TEXT_MUTED,
        fontfamily=style.FONT_FAMILY,
    )

    # Endpoint labels aligned to grid edges
    fig.text(
        leg_left,
        leg_bottom - 0.006,
        occ_low_label,
        ha="left",
        va="top",
        fontsize=6.2,
        color=style.TEXT_MUTED,
        fontfamily=style.FONT_FAMILY,
        linespacing=1.15,
    )
    fig.text(
        leg_right,
        leg_bottom - 0.006,
        occ_high_label,
        ha="right",
        va="top",
        fontsize=6.2,
        color=style.TEXT_MUTED,
        fontfamily=style.FONT_FAMILY,
        linespacing=1.15,
    )
    fig.text(
        leg_left - 0.010,
        leg_bottom,
        count_low_label,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=style.TEXT_MUTED,
        fontfamily=style.FONT_FAMILY,
        rotation=90,
        linespacing=1.1,
    )
    fig.text(
        leg_left - 0.010,
        leg_top,
        count_high_label,
        ha="right",
        va="top",
        fontsize=6.2,
        color=style.TEXT_MUTED,
        fontfamily=style.FONT_FAMILY,
        rotation=90,
        linespacing=1.1,
    )

    # Axis titles
    fig.text(
        leg_cx,
        leg_bottom - 0.036,
        f"{occ_axis_title}  ({TAU_OCC_LABEL})",
        ha="center",
        va="top",
        fontsize=8.0,
        color=style.TEXT_MUTED,
        fontfamily=style.FONT_FAMILY,
    )
    fig.text(
        leg_left - 0.052,
        leg_cy,
        f"{count_axis_title}  ({TAU_COUNT_LABEL})",
        ha="center",
        va="center",
        fontsize=8.0,
        color=style.TEXT_MUTED,
        fontfamily=style.FONT_FAMILY,
        rotation=90,
    )

    # ── Marginal histograms ───────────────────────────────────────────────
    bar_w = (hist_edges[1] - hist_edges[0]) * 0.92
    for ax_h, counts, bar_colors, var_label, show_xlabel in [
        (ax_hist_occ, occ_counts, occ_bar_colors, TAU_OCC_LABEL, False),
        (ax_hist_count, count_counts, count_bar_colors, TAU_COUNT_LABEL, True),
    ]:
        ax_h.set_facecolor(style.PAPER)
        ax_h.bar(
            hist_centers, counts, width=bar_w, color=bar_colors, linewidth=0, zorder=2
        )
        ax_h.axvline(0, color=style.GRIDLINE, linewidth=0.7, zorder=3)
        ax_h.set_xlim(hist_xmin, hist_xmax)
        ax_h.set_ylim(0, None)
        ax_h.set_yticks([])
        for sp in ["top", "right", "left"]:
            ax_h.spines[sp].set_visible(False)
        ax_h.spines["bottom"].set_color(style.GRIDLINE)
        ax_h.spines["bottom"].set_linewidth(0.6)
        if show_xlabel:
            ax_h.tick_params(
                axis="x",
                length=2.5,
                width=0.5,
                labelsize=6.5,
                colors=style.TEXT_MUTED,
                pad=1.5,
            )
            ax_h.set_xlabel(
                "% per year",
                fontsize=7.0,
                color=style.TEXT_MUTED,
                fontfamily=style.FONT_FAMILY,
                labelpad=3,
            )
        else:
            ax_h.tick_params(
                axis="x",
                length=2.5,
                width=0.5,
                labelbottom=False,
                colors=style.TEXT_MUTED,
            )
        ax_h.text(
            0.02,
            0.88,
            var_label,
            transform=ax_h.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            color=style.TEXT_MUTED,
            fontfamily=style.FONT_FAMILY,
            fontstyle="italic",
        )

    # ── Title block ────────────────────────────────────────────────────────
    sp_label = species_label or title.replace("_", " ").title()
    parts = sp_label.split("  —  ", 1)
    common_name = parts[0].strip()
    sci_name = parts[1].strip() if len(parts) > 1 else ""
    title_y = 0.976
    t_common = fig.text(
        0.04,
        title_y,
        common_name,
        ha="left",
        va="top",
        fontsize=18.8,
        fontweight="700",
        color=style.TEXT,
        fontfamily=style.FONT_FAMILY,
    )
    if sci_name:
        # Measure common name width after a renderer pass so we can place
        # sci name on the same baseline, right after.
        fig.canvas.draw()
        bb = t_common.get_window_extent(renderer=fig.canvas.get_renderer())
        x_sci = fig.transFigure.inverted().transform((bb.x1 + 8, bb.y1))[0]
        fig.text(
            x_sci,
            title_y - 0.0045,
            sci_name,
            ha="left",
            va="top",
            fontsize=11.0,
            fontstyle="italic",
            fontweight="normal",
            color=style.TEXT_MUTED,
            fontfamily=style.FONT_FAMILY,
        )
    fig.text(
        0.04,
        0.926,
        f"Spatial coupling between occurrence and count trends"
        f"  ·  {hex_size / 1_000:.0f} km hexagonal grid",
        ha="left",
        va="top",
        fontsize=10.2,
        color=style.TEXT_MUTED,
        fontfamily=style.FONT_FAMILY,
    )
    # ── Footer ─────────────────────────────────────────────────────────────
    med_occ = float(hex_joint["occ_val"].median())
    med_count = float(hex_joint["count_val"].median())
    same_sign = float(
        (np.sign(hex_joint["occ_val"]) == np.sign(hex_joint["count_val"])).mean() * 100
    )
    med_abs_delta = float(
        np.abs(hex_joint["occ_val"] - hex_joint["count_val"]).median()
    )
    foot_ax.text(
        0.5,
        0.46,
        f"median {TAU_OCC_LABEL} {med_occ:+.2f}% per year  ·  "
        f"median {TAU_COUNT_LABEL} {med_count:+.2f}% per year  ·  "
        f"median |\u0394| {med_abs_delta:.2f}% per year  ·  "
        f"same-sign hexes {same_sign:.0f}%",
        ha="center",
        va="center",
        **style.FONT_ANNOTATION,
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, facecolor=style.PAPER, bbox_inches="tight")
        logger.info("Saved → %s", output_path)
    return fig


# ── CLI ────────────────────────────────────────────────────────────────────


def _load_config_defaults(config_path: str | None) -> dict:
    if config_path is None:
        return {}
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    out: dict = {}
    if "run" in raw and "output_dir" in raw["run"]:
        out["results"] = raw["run"]["output_dir"]
    if "paths" in raw and "uk_shp" in raw["paths"]:
        out["uk_shp"] = raw["paths"]["uk_shp"]
    return out


def _parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    cfg = _load_config_defaults(pre_args.config)

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        default=None,
        help="pipeline YAML config file; sets --results and --uk-shp defaults",
    )
    p.add_argument(
        "--results",
        default=cfg.get("results", "output/prototype"),
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
        help="hex circumradius in metres (default: 3000 = 3 km)",
    )
    p.add_argument(
        "--min-cells",
        type=int,
        default=3,
        dest="min_cells",
        help="minimum 1 km cells per hex to render it (default: 3)",
    )
    p.add_argument(
        "--pct-vmax",
        type=float,
        default=97.5,
        dest="pct_vmax",
        help="percentile anchoring the diverging colour scale ends (default: 97.5)",
    )
    p.add_argument(
        "--palette",
        default=_DEFAULT_PALETTE,
        choices=list(_BIVARIATE_PALETTES),
        help=(
            "bivariate colour palette for the decoupling map "
            f"(default: {_DEFAULT_PALETTE})"
        ),
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--uk-shp",
        default=cfg.get("uk_shp"),
        help="path to UK boundary shapefile; read from --config if omitted",
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
        logger.info("[%s] trend panels → %s", sp, out_dir / "hex_trends.png")
        plot_trend_panels(
            df,
            title=sp,
            species_label=label,
            hex_size=args.hex_size,
            min_cells=args.min_cells,
            pct_vmax=args.pct_vmax,
            output_path=out_dir / "hex_trends.png",
            uk_shp=shp_path,
            dpi=args.dpi,
        )
        plt.close("all")
        logger.info("[%s] bivariate map → %s", sp, out_dir / "hex_decoupling.png")
        plot_bivariate_map(
            df,
            title=sp,
            species_label=label,
            hex_size=args.hex_size,
            min_cells=args.min_cells,
            pct_vmax=args.pct_vmax,
            palette=args.palette,
            output_path=out_dir / "hex_decoupling.png",
            uk_shp=shp_path,
            dpi=args.dpi,
        )
        plt.close("all")

    logger.info("Done.")


if __name__ == "__main__":
    main()
