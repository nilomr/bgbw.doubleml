"""Poster-quality map figures for BGBW DML trend and SDM outputs.

Figures use:
  - Off-white background (#EEEAE3) for sea and figure surround.
  - Dark charcoal (#282828) for land with no model coverage.
  - Clean diverging / sequential colormaps; near-zero values visually
    recede into the background.
  - No axis tick marks or grid lines; minimal chrome.

Typical usage::

    from bgbw_trends.viz.maps import plot_trend, plot_sdm
    fig = plot_trend(df, title="Blackbird")
    fig.savefig("trend.png", dpi=300, bbox_inches="tight")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import geopandas as gpd
import matplotlib as mpl
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bgbw_trends.viz import style
from bgbw_trends.viz._bng import refs_to_en, to_raster

logger = logging.getLogger(__name__)

# ── Default UK shapefile path ──────────────────────────────────────────────
_DEFAULT_UK_SHP = Path(
    "/home/nilomr/projects/bgbw.metadata.pipeline"
    "/data/metadata/uk-shapefile/UK-shapefile.shp"
)

# ── Significance threshold ─────────────────────────────────────────────────
_Z80 = 1.2816  # one-sided 90th pct of N(0,1) → 80% CI half-width


def _load_uk(shp_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shp_path)
    if gdf.crs.to_epsg() != 27700:
        gdf = gdf.to_crs(epsg=27700)
    return gdf


def _draw_land_bg(ax: Axes, uk: gpd.GeoDataFrame, extent: tuple) -> None:
    e_min, e_max, n_min, n_max = extent
    ax.set_xlim(e_min, e_max)
    ax.set_ylim(n_min, n_max)
    uk.plot(ax=ax, color=style.LAND, linewidth=0, zorder=1)


def _draw_coast(ax: Axes, uk: gpd.GeoDataFrame) -> None:
    uk.boundary.plot(ax=ax, color=style.COAST, linewidth=0.4, zorder=5)


def _style_ax(ax: Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")
    ax.set_facecolor(style.WATER)


def _add_colorbar(
    fig: Figure,
    cax: Axes,
    norm: mc.Normalize,
    cmap: mc.Colormap,
    label: str,
    fmt: str = "%.1f",
    orientation: Literal["horizontal", "vertical"] = "horizontal",
) -> None:
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation=orientation)
    cb.outline.set_visible(False)
    cb.ax.tick_params(
        length=2.5, width=0.5, labelsize=7.5, colors=style.TEXT_MUTED, pad=2
    )
    axis = cb.ax.yaxis if orientation == "vertical" else cb.ax.xaxis
    axis.set_major_formatter(mpl.ticker.FormatStrFormatter(fmt))
    if orientation == "vertical":
        cb.ax.yaxis.set_ticks_position("right")
        cb.ax.set_ylabel(
            label, **style.FONT_LABEL, labelpad=8, rotation=270, va="bottom"
        )
    else:
        cb.ax.xaxis.set_ticks_position("bottom")
        cb.ax.set_xlabel(label, **style.FONT_LABEL, labelpad=4)
    cb.locator = mpl.ticker.MaxNLocator(
        nbins=5, symmetric=isinstance(norm, mc.TwoSlopeNorm)
    )
    cb.update_ticks()
    cax.set_facecolor(style.PAPER)
    for spine in cax.spines.values():
        spine.set_visible(False)


def _panel_label(ax: Axes, letter: str) -> None:
    ax.text(
        0.02,
        0.98,
        letter,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        fontweight="700",
        color=style.TEXT_MUTED,
    )


def _panel_title(ax: Axes, text: str) -> None:
    ax.text(
        0.02,
        0.98,
        text,
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


def _caption(ax: Axes, text: str) -> None:
    ax.text(
        0.02,
        0.02,
        text,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.24,rounding_size=0.15",
            "facecolor": style.PAPER,
            "edgecolor": "none",
            "alpha": 0.92,
        },
        **style.FONT_ANNOTATION,
    )


def _raster_rgba(
    raster: np.ndarray,
    norm: mc.Normalize,
    cmap: mc.Colormap,
) -> np.ndarray:
    """RGBA image with NaN cells fully transparent."""
    rgba = np.array(cmap(norm(np.nan_to_num(raster, nan=0.0))), dtype=np.float32)
    rgba[np.isnan(raster), 3] = 0.0
    return rgba


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float64)
    return np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float64)
    return np.where(
        arr <= 0.0031308, 12.92 * arr, 1.055 * np.power(arr, 1.0 / 2.4) - 0.055
    )


def _rgb_to_oklab(rgb: tuple[float, float, float]) -> np.ndarray:
    r_lin, g_lin, b_lin = _srgb_to_linear(np.asarray(rgb, dtype=np.float64))
    l = 0.4122214708 * r_lin + 0.5363325363 * g_lin + 0.0514459929 * b_lin
    m = 0.2119034982 * r_lin + 0.6806995451 * g_lin + 0.1073969566 * b_lin
    s = 0.0883024619 * r_lin + 0.2817188376 * g_lin + 0.6299787005 * b_lin

    l_cbrt = np.cbrt(l)
    m_cbrt = np.cbrt(m)
    s_cbrt = np.cbrt(s)
    return np.array(
        [
            0.2104542553 * l_cbrt + 0.7936177850 * m_cbrt - 0.0040720468 * s_cbrt,
            1.9779984951 * l_cbrt - 2.4285922050 * m_cbrt + 0.4505937099 * s_cbrt,
            0.0259040371 * l_cbrt + 0.7827717662 * m_cbrt - 0.8086757660 * s_cbrt,
        ],
        dtype=np.float64,
    )


def _oklab_to_rgb(lab: np.ndarray) -> tuple[float, float, float]:
    L, a, b = np.asarray(lab, dtype=np.float64)
    l_cbrt = L + 0.3963377774 * a + 0.2158037573 * b
    m_cbrt = L - 0.1055613458 * a - 0.0638541728 * b
    s_cbrt = L - 0.0894841775 * a - 1.2914855480 * b

    l = l_cbrt**3
    m = m_cbrt**3
    s = s_cbrt**3

    rgb_lin = np.array(
        [
            4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
        ],
        dtype=np.float64,
    )
    rgb = _linear_to_srgb(np.clip(rgb_lin, 0.0, None))
    return tuple(np.clip(rgb, 0.0, 1.0))


def _blend_oklab(
    abundance_rgb: tuple[float, float, float],
    trend_rgb: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Blend abundance and trend colours in Oklab.

    The abundance palette primarily controls lightness, while the trend palette
    carries most of the hue/chroma information so the sign of change remains
    visually legible.
    """
    abd_lab = _rgb_to_oklab(abundance_rgb)
    trend_lab = _rgb_to_oklab(trend_rgb)
    mixed_lab = np.array(
        [
            0.60 * abd_lab[0] + 0.40 * trend_lab[0],
            0.50 * abd_lab[1] + 0.50 * trend_lab[1],
            0.50 * abd_lab[2] + 0.50 * trend_lab[2],
        ],
        dtype=np.float64,
    )
    return _oklab_to_rgb(mixed_lab)


def _make_bivariate_palette(
    abd_breaks: np.ndarray,
    tau_breaks: np.ndarray,
) -> list[list[str]]:
    """Build an ``n × n`` bivariate palette via 4-corner bilinear Oklab interpolation.

    Four anchor colours define the corners of the 2-D colour space:

    * (low abd, min trend)  → warm sand   #F0D9A0
    * (low abd, max trend)  → pale sky    #A8CDDE
    * (high abd, min trend) → burnt amber #B53A16
    * (high abd, max trend) → forest teal #1B5C44

    Bin fractions are uniform — bin *i* → ``(i + 0.5) / n`` — so all *n*
    bins always span the full colour space regardless of where the data
    range sits relative to zero.  This avoids the compression artefact
    produced by ``TwoSlopeNorm`` when all trend values share the same sign.
    """
    n_abd = len(abd_breaks) - 1
    n_tau = len(tau_breaks) - 1

    c00 = _rgb_to_oklab(mc.to_rgb("#EB8423"))  # low abd, min trend
    c01 = _rgb_to_oklab(mc.to_rgb("#7FE298"))  # low abd, max trend
    c10 = _rgb_to_oklab(mc.to_rgb("#EC3818"))  # high abd, min trend
    c11 = _rgb_to_oklab(mc.to_rgb("#21AEF0"))  # high abd, max trend

    palette: list[list[str]] = []
    for i in range(n_abd):
        a = (i + 0.5) / n_abd  # 0 = lowest abundance bin, 1 = highest
        row: list[str] = []
        for j in range(n_tau):
            t = (j + 0.5) / n_tau  # 0 = min-trend bin, 1 = max-trend bin
            lab = (
                (1 - a) * (1 - t) * c00
                + (1 - a) * t * c01
                + a * (1 - t) * c10
                + a * t * c11
            )
            row.append(mc.to_hex(_oklab_to_rgb(lab)))
        palette.append(row)
    return palette


def _bin_edges(arr: np.ndarray, n_classes: int) -> np.ndarray:
    """Quantile bin edges with monotonic guards for tied values."""
    edges = np.nanquantile(arr, np.linspace(0.0, 1.0, n_classes + 1)).astype(np.float64)
    for index in range(1, len(edges)):
        if edges[index] <= edges[index - 1]:
            edges[index] = np.nextafter(edges[index - 1], np.inf)
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def _format_bin_label(left: float, right: float, *, scale: float = 1.0) -> str:
    """Compact label for a closed-open bin interval."""
    left_scaled = left * scale
    right_scaled = right * scale

    def _fmt(value: float) -> str:
        sign = "+" if value > 0 else ""
        if abs(value) >= 10:
            return f"{sign}{value:.0f}"
        if abs(value) >= 1:
            return f"{sign}{value:.1f}"
        return f"{sign}{value:.2f}"

    return f"{_fmt(left_scaled)}–{_fmt(right_scaled)}"


# ─────────────────────────────────────────────────────────────────────────
#  Figure layout helper
# ─────────────────────────────────────────────────────────────────────────


def _two_panel_layout(
    ncols: int = 2,
    panel_w: float = 5.4,
    panel_h: float = 7.95,
    gap: float = 0.44,
    top_margin: float = 1.08,
    footer_h: float = 0.22,
    footer_gap: float = 0.10,
    legend_h: float = 0.22,
    legend_gap: float = 0.16,
    bottom_margin: float = 0.34,
    side_margin: float = 0.34,
) -> tuple[Figure, list[Axes], list[Axes], Axes]:
    """Return a figure, map axes, legend axes, and a dedicated footer axis."""
    total_w = side_margin + ncols * panel_w + (ncols - 1) * gap + side_margin
    total_h = (
        top_margin
        + panel_h
        + footer_gap
        + footer_h
        + legend_gap
        + legend_h
        + bottom_margin
    )

    fig = plt.figure(figsize=(total_w, total_h), facecolor=style.PAPER)

    map_axes, cbar_axes = [], []
    for i in range(ncols):
        x0 = (side_margin + i * (panel_w + gap)) / total_w
        y0 = (bottom_margin + legend_h + legend_gap + footer_h + footer_gap) / total_h
        w = panel_w / total_w
        h = panel_h / total_h

        ax = fig.add_axes([x0, y0, w, h])
        map_axes.append(ax)

        cx0 = x0 + 0.04 * w
        cw = 0.92 * w
        cy0 = bottom_margin / total_h
        ch = legend_h / total_h
        cax = fig.add_axes([cx0, cy0, cw, ch])
        cbar_axes.append(cax)

    footer_ax = fig.add_axes(
        [
            side_margin / total_w,
            (bottom_margin + legend_h + legend_gap) / total_h,
            (total_w - 2 * side_margin) / total_w,
            footer_h / total_h,
        ]
    )
    footer_ax.set_axis_off()

    return fig, map_axes, cbar_axes, footer_ax


def _header_block(
    fig: Figure,
    *,
    title: str,
    subtitle: str,
) -> None:
    fig.text(0.06, 0.972, title, ha="left", va="top", **style.FONT_TITLE)
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


def _footer_block(footer_ax: Axes, text: str) -> None:
    footer_ax.text(1.0, 0.5, text, ha="right", va="center", **style.FONT_ANNOTATION)


def _single_panel_layout(
    panel_w: float = 6.8,
    panel_h: float = 7.9,
    top_margin: float = 1.08,
    legend_w: float = 2.05,
    legend_h: float = 1.75,
    legend_gap: float = 0.18,
    footer_h: float = 0.22,
    footer_gap: float = 0.26,
    bottom_margin: float = 0.38,
    side_margin: float = 0.34,
) -> tuple[Figure, Axes, Axes, Axes]:
    """Return a figure, one map axis, one legend axis, and a footer axis."""
    total_w = side_margin + panel_w + side_margin
    total_h = (
        top_margin
        + panel_h
        + legend_gap
        + legend_h
        + footer_gap
        + footer_h
        + bottom_margin
    )

    fig = plt.figure(figsize=(total_w, total_h), facecolor=style.PAPER)

    x0 = side_margin / total_w
    y0 = (bottom_margin + footer_h + footer_gap + legend_h + legend_gap) / total_h
    ax = fig.add_axes([x0, y0, panel_w / total_w, panel_h / total_h])

    legend_x0 = ((total_w - legend_w) / 2.0) / total_w
    legend_y0 = (bottom_margin + footer_h + footer_gap) / total_h
    legend_ax = fig.add_axes(
        [legend_x0, legend_y0, legend_w / total_w, legend_h / total_h]
    )
    legend_ax.set_facecolor(style.PAPER)

    footer_ax = fig.add_axes(
        [x0, bottom_margin / total_h, panel_w / total_w, footer_h / total_h]
    )
    footer_ax.set_axis_off()

    return fig, ax, legend_ax, footer_ax


def _draw_bivariate_legend(
    legend_ax: Axes,
    palette: list[list[str]],
    *,
    tau_breaks: np.ndarray,
    abd_breaks: np.ndarray,
    n_classes: int,
    metric_unit: str = "% yr⁻¹",
) -> None:
    """Draw a discrete bivariate legend in a dedicated legend axis."""
    colors = np.array(
        [
            [mc.to_rgba(palette[row][col]) for col in range(n_classes)]
            for row in range(n_classes)
        ],
        dtype=np.float32,
    )

    legend_ax.imshow(
        colors,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        extent=[-0.5, n_classes - 0.5, -0.5, n_classes - 0.5],
    )

    for coord in np.arange(0.5, n_classes - 0.5, 1.0):
        legend_ax.axhline(coord, color=style.PAPER, linewidth=1.0, alpha=0.95)
        legend_ax.axvline(coord, color=style.PAPER, linewidth=1.0, alpha=0.95)

    legend_ax.set_xticks(range(n_classes))
    legend_ax.set_xticklabels(
        [
            _format_bin_label(tau_breaks[idx], tau_breaks[idx + 1], scale=100.0)
            for idx in range(n_classes)
        ],
        fontsize=6.0,
        color=style.TEXT_MUTED,
        rotation=28,
        ha="right",
    )
    legend_ax.set_yticks(range(n_classes))
    legend_ax.set_yticklabels(
        [
            _format_bin_label(abd_breaks[idx], abd_breaks[idx + 1], scale=1.0)
            for idx in range(n_classes)
        ],
        fontsize=6.0,
        color=style.TEXT_MUTED,
    )
    legend_ax.tick_params(length=0, pad=2.0)

    legend_ax.set_xlabel(
        f"Trend ({metric_unit})",
        fontsize=7.0,
        color=style.TEXT_MUTED,
        labelpad=7,
    )
    legend_ax.set_ylabel(
        "Expected abundance",
        fontsize=7.0,
        color=style.TEXT_MUTED,
        labelpad=8,
    )
    legend_ax.set_title("Binned interaction", fontsize=7.2, color=style.TEXT, pad=4)

    legend_ax.scatter(
        -0.5, -0.5, marker="^", color=style.TEXT, s=20, clip_on=False, zorder=10
    )
    legend_ax.scatter(
        n_classes - 0.5,
        n_classes - 0.5,
        marker=">",
        color=style.TEXT,
        s=20,
        clip_on=False,
        zorder=10,
    )

    for spine in legend_ax.spines.values():
        spine.set_visible(False)


# ─────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────


def plot_trend(
    df: pd.DataFrame,
    *,
    title: str,
    species_label: str | None = None,
    metric: Literal["abd_ppy", "occ_ppy", "count_ppy"] = "abd_ppy",
    sigma_col: str | None = None,
    strong_threshold_pct: float = 1.0,
    pct_vmax: float = 97.5,
    output_path: Path | str | None = None,
    uk_shp: Path | str = _DEFAULT_UK_SHP,
    dpi: int = 300,
) -> Figure:
    """Two-panel trend map.

    **Panel A** — continuous trend estimate across all grid cells.
    **Panel B** — cells with 80%-CI excluding zero where uncertainty is
    available, otherwise cells with |trend| ≥ ``strong_threshold_pct`` % yr⁻¹.

    Trend values are multiplied by 100 and displayed as % yr⁻¹.

    Parameters
    ----------
    df
        DataFrame with ``bng_ref`` and the trend metric column.
    title
        Short species identifier (column name or code).
    species_label
        Human-readable species name for the figure title.
    metric
        Which per-year rate to plot.
    sigma_col
        Column holding the posterior SD of *metric*; when provided the
        significance mask uses 80%-CI non-zero instead of the threshold.
    strong_threshold_pct
        Absolute threshold in % yr⁻¹; used when ``sigma_col`` is absent.
    pct_vmax
        Symmetric colour scale extends to this percentile of |trend|.
    output_path
        Save path; parent directory is created automatically.
    uk_shp
        Path to the UK boundary shapefile.
    dpi
        Output resolution.
    """
    style.apply()
    uk = _load_uk(Path(uk_shp))

    # Trends are stored as ratios; display as percent-per-year
    tau_raw = df[metric].to_numpy(dtype=np.float64)
    tau_pct = tau_raw * 100.0

    east, north = refs_to_en(df["bng_ref"].to_numpy())
    valid = np.isfinite(east) & np.isfinite(north) & np.isfinite(tau_pct)
    tau_pct = tau_pct[valid]
    east, north = east[valid], north[valid]

    vmax = max(float(np.nanpercentile(np.abs(tau_pct), pct_vmax)), 0.01)
    norm = mc.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    raster_all, extent = to_raster(east, north, tau_pct)

    # Mask: 80%-CI non-zero whenever a usable sigma exists.
    sigma_pct: np.ndarray | None = None
    if sigma_col and sigma_col in df.columns:
        sigma_pct = df[sigma_col].to_numpy(dtype=np.float64)[valid] * 100.0
        panel_b_cap = "80% CI excludes zero"
    elif metric == "abd_ppy" and {"occ_sigma", "count_sigma"}.issubset(df.columns):
        occ_sigma_pct = df["occ_sigma"].to_numpy(dtype=np.float64)[valid] * 100.0
        count_sigma_pct = df["count_sigma"].to_numpy(dtype=np.float64)[valid] * 100.0
        # Approximate abundance-stage uncertainty via variance additivity.
        sigma_pct = np.sqrt(occ_sigma_pct**2 + count_sigma_pct**2)
        panel_b_cap = "80% CI excludes zero"
    else:
        panel_b_cap = f"|trend| ≥ {strong_threshold_pct:.0f}% yr⁻¹"

    if sigma_pct is not None:
        sig_mask = (np.abs(tau_pct) - _Z80 * sigma_pct) > 0
    else:
        sig_mask = np.abs(tau_pct) >= strong_threshold_pct

    tau_masked = np.where(sig_mask, tau_pct, np.nan)
    raster_sig, _ = to_raster(east, north, tau_masked)

    # ── Layout ─────────────────────────────────────────────────────────
    fig, (ax_l, ax_r), (cax_l, cax_r), footer_ax = _two_panel_layout()
    # Share one colorbar scale; put it only on the right panel
    cax_l.set_visible(False)

    for ax, rast, panel_title in [
        (ax_l, raster_all, "All estimates"),
        (ax_r, raster_sig, panel_b_cap),
    ]:
        _style_ax(ax)
        _draw_land_bg(ax, uk, extent)
        ax.imshow(
            _raster_rgba(rast, norm, style.TREND_CMAP),
            origin="lower",
            extent=extent,
            interpolation="none",
            aspect="equal",
            zorder=3,
        )
        _draw_coast(ax, uk)
        _panel_title(ax, panel_title)

    _add_colorbar(
        fig,
        cax_r,
        norm,
        style.TREND_CMAP,
        label="Annual change (% yr⁻¹)",
        fmt="%.1f",
        orientation="horizontal",
    )
    cax_l.text(0.0, 0.70, "Colour scale", ha="left", va="center", **style.FONT_SECTION)
    cax_l.text(
        0.0,
        0.18,
        "Decrease shown in red, increase in blue.",
        ha="left",
        va="center",
        **style.FONT_ANNOTATION,
    )
    cax_l.set_axis_off()

    # ── Title block ────────────────────────────────────────────────────
    sp_label = species_label or title.replace("_", " ").title()
    metric_names = {
        "abd_ppy": "Abundance trend",
        "occ_ppy": "Occurrence trend",
        "count_ppy": "Count trend",
    }
    _header_block(
        fig,
        title=sp_label,
        subtitle=f"{metric_names.get(metric, metric)} across 1 km grid cells",
    )

    mean_tau = float(np.nanmean(tau_pct))
    n_sig = int(sig_mask.sum())
    pct_sig = 100.0 * n_sig / max(len(sig_mask), 1)
    _footer_block(
        footer_ax,
        f"Grid mean {mean_tau:+.2f}% yr⁻¹  •  detectable change in {pct_sig:.0f}% of cells",
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, facecolor=style.PAPER)
        logger.info("Saved trend map → %s", output_path)

    return fig


def plot_sdm(
    df: pd.DataFrame,
    *,
    title: str,
    species_label: str | None = None,
    occ_col: str = "p_occurrence",
    abd_col: str = "e_abundance",
    output_path: Path | str | None = None,
    uk_shp: Path | str = _DEFAULT_UK_SHP,
    dpi: int = 300,
) -> Figure:
    """Two-panel SDM map: P(occurrence) and E[abundance].

    Parameters
    ----------
    df
        DataFrame with ``bng_ref``, occurrence and abundance columns.
    title
        Short species identifier.
    species_label
        Human-readable species name for the figure title.
    output_path
        Save path; parent directory is created automatically.
    uk_shp
        Path to the UK boundary shapefile.
    dpi
        Output resolution.
    """
    style.apply()
    uk = _load_uk(Path(uk_shp))

    east, north = refs_to_en(df["bng_ref"].to_numpy())
    valid = np.isfinite(east) & np.isfinite(north)

    occ = df[occ_col].to_numpy(dtype=np.float64)[valid]
    abd = df[abd_col].to_numpy(dtype=np.float64)[valid]
    east, north = east[valid], north[valid]

    raster_occ, extent = to_raster(east, north, occ)
    raster_abd, _ = to_raster(east, north, abd)

    norm_occ = mc.Normalize(vmin=0.0, vmax=1.0)
    abd_max = max(float(np.nanpercentile(abd, 99)), 1e-6)
    norm_abd = mc.Normalize(vmin=0.0, vmax=abd_max)

    # ── Layout ─────────────────────────────────────────────────────────
    fig, (ax_l, ax_r), (cax_l, cax_r), footer_ax = _two_panel_layout()

    for ax, rast, norm, cmap, cax, panel_title, cbar_lbl in [
        (
            ax_l,
            raster_occ,
            norm_occ,
            style.OCC_CMAP,
            cax_l,
            "Probability of occurrence",
            "P(occurrence)",
        ),
        (
            ax_r,
            raster_abd,
            norm_abd,
            style.ABD_CMAP,
            cax_r,
            "Expected abundance",
            "E[count | occ.]",
        ),
    ]:
        _style_ax(ax)
        _draw_land_bg(ax, uk, extent)
        ax.imshow(
            _raster_rgba(rast, norm, cmap),
            origin="lower",
            extent=extent,
            interpolation="none",
            aspect="equal",
            zorder=3,
        )
        _draw_coast(ax, uk)
        _panel_title(ax, panel_title)
        _add_colorbar(
            fig,
            cax,
            norm,
            cmap,
            label=cbar_lbl,
            fmt="%.2f",
            orientation="horizontal",
        )

    # ── Title block ────────────────────────────────────────────────────
    sp_label = species_label or title.replace("_", " ").title()
    _header_block(
        fig,
        title=sp_label,
        subtitle="Effort-adjusted species distribution model",
    )

    mean_occ = float(np.nanmean(occ))
    mean_abd = float(np.nanmean(abd))
    _footer_block(
        footer_ax,
        f"Mean P(occ) {mean_occ:.3f}  •  Mean E[abundance] {mean_abd:.2f}",
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, facecolor=style.PAPER)
        logger.info("Saved SDM map → %s", output_path)

    return fig


def plot_bivariate(
    df: pd.DataFrame,
    *,
    title: str,
    species_label: str | None = None,
    metric: Literal["abd_ppy", "occ_ppy", "count_ppy"] = "abd_ppy",
    abd_col: str = "e_abundance",
    n_classes: int = 5,
    pct_vmax: float = 97.5,
    output_path: Path | str | None = None,
    uk_shp: Path | str = _DEFAULT_UK_SHP,
    dpi: int = 300,
) -> Figure:
    """Single-panel bivariate choropleth: binned abundance × binned trend.

    Grid cells are classified into *n_classes* × *n_classes* bins (tertiles
    by default) along two dimensions — expected abundance and the chosen
    trend metric — and each cell is assigned the colour from the corresponding
    square of a bivariate palette obtained by blending the abundance and trend
    palettes across the two-dimensional bin grid.

    Parameters
    ----------
    df
        DataFrame with ``bng_ref``, the abundance column, and the trend
        metric column.
    title
        Short species identifier.
    species_label
        Human-readable species name for the figure title.
    metric
        Trend metric column to use for the x-dimension.
    abd_col
        Abundance column to use for the y-dimension.
    n_classes
        Number of bins per dimension (3 produces a 3×3 scheme).
    pct_vmax
        Only used to report the colourbar range in the footer.
    output_path
        Save path; parent directory is created automatically.
    uk_shp
        Path to the UK boundary shapefile.
    dpi
        Output resolution.
    """
    style.apply()
    uk = _load_uk(Path(uk_shp))

    tau_raw = df[metric].to_numpy(dtype=np.float64)
    abd_raw = df[abd_col].to_numpy(dtype=np.float64)
    east, north = refs_to_en(df["bng_ref"].to_numpy())
    valid = (
        np.isfinite(east)
        & np.isfinite(north)
        & np.isfinite(tau_raw)
        & np.isfinite(abd_raw)
    )
    tau, abd = tau_raw[valid], abd_raw[valid]
    east, north = east[valid], north[valid]

    # Quantile bins follow the Parkes reference pattern: bin each variable
    # independently, then map the bin-pair onto a bivariate colour square.
    abd_breaks = _bin_edges(abd, n_classes)
    tau_breaks = _bin_edges(tau, n_classes)

    abd_idx = np.clip(np.digitize(abd, abd_breaks) - 1, 0, n_classes - 1)
    tau_idx = np.clip(np.digitize(tau, tau_breaks) - 1, 0, n_classes - 1)

    # Discrete n x n palette: sequential abundance vertically, diverging trend
    # horizontally, with each square sampled from the bin centres and blended
    # perceptually.
    palette = _make_bivariate_palette(abd_breaks, tau_breaks)
    cell_colors = np.array(
        [mc.to_rgba(palette[abd_idx[i]][tau_idx[i]]) for i in range(len(tau))],
        dtype=np.float32,
    )

    # Build RGBA raster at 1km cell resolution
    cell_m = 1_000
    dummy = np.zeros(len(tau), dtype=np.float64)
    raster_dummy, extent = to_raster(east, north, dummy, cell_m=cell_m)
    H, W = raster_dummy.shape
    rgba_raster = np.zeros((H, W, 4), dtype=np.float32)
    sw_e = (np.round(east / cell_m) * cell_m).astype(np.int64)
    sw_n = (np.round(north / cell_m) * cell_m).astype(np.int64)
    col_idx = ((sw_e - int(extent[0])) // cell_m).astype(int)
    row_idx = ((sw_n - int(extent[2])) // cell_m).astype(int)
    for i in range(len(tau)):
        r, c = row_idx[i], col_idx[i]
        if 0 <= r < H and 0 <= c < W:
            rgba_raster[r, c] = cell_colors[i]

    # ── Layout ─────────────────────────────────────────────────────────
    fig, ax, legend_ax, footer_ax = _single_panel_layout()
    _style_ax(ax)
    _draw_land_bg(ax, uk, extent)
    ax.imshow(
        rgba_raster,
        origin="lower",
        extent=extent,
        interpolation="none",
        aspect="equal",
        zorder=3,
    )
    _draw_coast(ax, uk)

    metric_names = {
        "abd_ppy": "abundance",
        "occ_ppy": "occurrence",
        "count_ppy": "count",
    }
    _panel_title(
        ax, f"Expected abundance  ×  annual {metric_names.get(metric, metric)} trend"
    )
    _draw_bivariate_legend(
        legend_ax,
        palette,
        tau_breaks=tau_breaks,
        abd_breaks=abd_breaks,
        n_classes=n_classes,
        metric_unit="% yr⁻¹",
    )

    # ── Title block ────────────────────────────────────────────────────
    sp_label = species_label or title.replace("_", " ").title()
    metric_long = {
        "abd_ppy": "Abundance",
        "occ_ppy": "Occurrence",
        "count_ppy": "Count",
    }
    _header_block(
        fig,
        title=sp_label,
        subtitle=f"{metric_long.get(metric, metric)} × Trend bivariate map",
    )

    mean_tau = float(np.nanmean(tau * 100.0))
    mean_abd = float(np.nanmean(abd))
    _footer_block(
        footer_ax,
        f"Mean {metric_names.get(metric, metric)} {mean_abd:.2f}  •  "
        f"Grid mean trend {mean_tau:+.2f}% yr⁻¹",
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, facecolor=style.PAPER)
        logger.info("Saved bivariate map → %s", output_path)

    return fig
