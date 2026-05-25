#!/usr/bin/env python3
"""Species trend forest plot  —  UK medians and spatial spread.

Each row shows a species for three metrics (occupancy, count, abundance):

  dot      UK-wide median of per-cell %/yr trend estimates
  bar      spatial IQR (Q25–Q75 across ~80k 1 km cells)
  whisker  spatial P10–P90

All spread is SPATIAL (cell-to-cell variation), not trend uncertainty.
Species are sorted ascending by median abundance trend.

Usage
-----
    uv run python scripts/products/make_trends_forest_plot.py \\
        --config configs/local.yaml

    uv run python scripts/products/make_trends_forest_plot.py \\
        --results-dir /mnt/w/projects/BGBW/results/cluster
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from _common import (
    load_config_defaults,
    panel_chip,
    species_label,
)
from bgbw_trends.viz import style

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

_METRICS = [
    ("occ_ppy",   "Occupancy"),
    ("count_ppy", "Count"),
    ("abd_ppy",   "Abundance"),
]

# Percentile to clip the shared x-axis (suppresses extreme outliers)
_CLIP_PCT = 98.0
# Inner bar (spatial IQR) and outer whisker percentile bounds
_INNER_PCT = [25, 75]
_OUTER_PCT = [10, 90]


def _parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    cfg = load_config_defaults(pre_args.config)

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default=None, help="pipeline YAML config file")
    p.add_argument(
        "--results-dir", default=cfg.get("results_dir", "output"),
        help="pipeline output directory",
    )
    p.add_argument("--species", nargs="+", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def _draw_panel(
    ax: plt.Axes,
    species_order: list[str],
    data_by_species: dict[str, pd.DataFrame],
    metric: str,
    title: str,
    *,
    show_ylabels: bool,
    norm: mcolors.Normalize,
    cmap: mcolors.Colormap,
    x_lim: tuple[float, float],
    abd_medians: dict[str, float],
) -> None:
    n = len(species_order)

    for i, sp in enumerate(species_order):
        vals = data_by_species[sp][metric].dropna().to_numpy() * 100.0
        if len(vals) < 5:
            continue

        p10, p25, med, p75, p90 = np.percentile(vals, [10, 25, 50, 75, 90])
        # Colour always driven by abundance trend for cross-panel consistency
        color = cmap(norm(abd_medians[sp]))

        # Alternating row shading
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=style.GRIDLINE, alpha=0.07, zorder=0)

        # Clamp whisker/bar endpoints to display range
        p10d = max(p10, x_lim[0])
        p90d = min(p90, x_lim[1])
        p25d = max(p25, x_lim[0])
        p75d = min(p75, x_lim[1])

        # Thin outer whisker (P10–P90)
        ax.plot(
            [p10d, p90d], [i, i],
            color=color, linewidth=1.0, alpha=0.35, zorder=2,
            solid_capstyle="round",
        )
        # Whisker end caps
        for xv in (p10d, p90d):
            ax.plot(
                [xv, xv], [i - 0.15, i + 0.15],
                color=color, linewidth=0.8, alpha=0.40, zorder=2,
            )

        # Thick inner IQR bar (Q25–Q75)
        ax.plot(
            [p25d, p75d], [i, i],
            color=color, linewidth=4.5, alpha=0.60, zorder=3,
            solid_capstyle="round",
        )

        # Median dot with white edge
        ax.scatter(
            np.clip(med, *x_lim), i,
            s=55, color=color, zorder=4,
            edgecolors=style.PAPER, linewidths=0.9,
        )

    # Zero reference line
    ax.axvline(0, color=style.TEXT_MUTED, linewidth=0.75, linestyle=(0, (4, 3)), zorder=1)

    # Horizontal gridlines at each species row
    for i in range(n):
        ax.axhline(i, color=style.GRIDLINE, linewidth=0.35, zorder=0)

    ax.set_xlim(*x_lim)
    ax.set_ylim(-0.5, n - 0.4)
    ax.set_yticks(range(n))

    if show_ylabels:
        ax.set_yticklabels(
            [species_label(sp) for sp in species_order],
            **style.FONT_LABEL,
        )
    else:
        ax.tick_params(labelleft=False)

    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=style.TEXT_MUTED, labelsize=8.5, length=3)
    ax.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, _: f"{x:+.0f}")
    )
    ax.set_xlabel("% per year", **style.FONT_LABEL, labelpad=5)

    ax.set_facecolor(style.PAPER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(style.GRIDLINE)

    panel_chip(ax, title, style)


def main() -> None:
    args = _parse_args()
    style.apply()

    results_dir = Path(args.results_dir)
    output_path = (
        Path(args.output) if args.output
        else results_dir / "figures" / "products" / "trends_ridgeline.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ensemble_root = results_dir / "ensemble" / "species"
    found = {p.parent.name: p for p in sorted(ensemble_root.glob("*/ensemble.parquet"))}
    if not found:
        logger.error("No ensemble parquets found in %s", ensemble_root)
        sys.exit(1)

    species_list = [s for s in (args.species or sorted(found)) if s in found]
    if missing := set(args.species or []) - set(found):
        logger.warning("No ensemble data for: %s", ", ".join(sorted(missing)))
    if not species_list:
        logger.error("No valid species.")
        sys.exit(1)

    # Load all ensemble parquets
    data_by_species: dict[str, pd.DataFrame] = {}
    abd_medians: dict[str, float] = {}
    all_vals: dict[str, list[float]] = {m: [] for m, _ in _METRICS}

    for sp in species_list:
        df = pd.read_parquet(found[sp])
        data_by_species[sp] = df
        abd_medians[sp] = float(np.median(df["abd_ppy"].dropna()) * 100.0)
        for metric, _ in _METRICS:
            vals = df[metric].dropna().to_numpy() * 100.0
            all_vals[metric].extend(vals.tolist())
        logger.info("  %s  abd_ppy median = %+.2f%%", sp, abd_medians[sp])

    # Sort species by median abundance trend (worst → best, bottom to top)
    species_order = sorted(species_list, key=lambda s: abd_medians[s])
    n = len(species_order)

    # Shared x-limits per metric, clipped to suppress extreme outliers
    x_lims: dict[str, tuple[float, float]] = {}
    for metric, _ in _METRICS:
        v = np.array(all_vals[metric])
        lo = float(np.percentile(v, 100 - _CLIP_PCT))
        hi = float(np.percentile(v, _CLIP_PCT))
        pad = (hi - lo) * 0.05
        x_lims[metric] = (lo - pad, hi + pad)

    # Colour norm: symmetric, driven by abundance trend (consistent across panels)
    vext = max(abs(v) for v in abd_medians.values()) or 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-vext, vcenter=0.0, vmax=vext)
    cmap = style.TREND_CMAP

    # ── Figure ────────────────────────────────────────────────────────────────
    fig_h = max(5.0, 0.55 * n + 3.2)
    fig, axes = plt.subplots(
        1, 3, figsize=(13.5, fig_h), facecolor=style.PAPER,
    )

    for col_idx, (ax, (metric, title)) in enumerate(zip(axes, _METRICS)):
        _draw_panel(
            ax, species_order, data_by_species, metric, title,
            show_ylabels=(col_idx == 0),
            norm=norm, cmap=cmap,
            x_lim=x_lims[metric],
            abd_medians=abd_medians,
        )

    gs_left, gs_right = 0.16, 0.86
    fig.subplots_adjust(left=gs_left, right=gs_right, top=0.80, bottom=0.12, wspace=0.08)

    # Shared colourbar — placed manually so it never overlaps the panels
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar_ax = fig.add_axes([0.88, 0.22, 0.014, 0.46])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.outline.set_visible(False)
    cbar.ax.set_ylabel("Abundance trend\n(% / yr)", **style.FONT_LABEL, labelpad=7)
    cbar.ax.tick_params(labelsize=8, colors=style.TEXT_MUTED, length=3)
    formatter = mpl.ticker.FuncFormatter(lambda x, _: f"{x:+.0f}")
    cbar.ax.yaxis.set_major_formatter(formatter)

    # ── Header aligned to plot left edge ─────────────────────────────────────
    fig.text(
        gs_left, 0.975,
        "Species trends  \u2014  UK medians and spatial spread",
        ha="left", va="top", **style.FONT_TITLE,
    )
    fig.text(
        gs_left, 0.928,
        "dot = UK median  \u2022  bar = spatial IQR (Q25\u2013Q75)  \u2022  "
        "whisker = P10\u2013P90  \u2022  all spread is spatial (cell-to-cell), not trend uncertainty",
        ha="left", va="top", **style.FONT_SUBTITLE,
    )
    fig.add_artist(
        mpl.lines.Line2D(
            [gs_left, 0.96], [0.905, 0.905],
            transform=fig.transFigure, color=style.GRIDLINE, linewidth=0.8,
        )
    )
    fig.text(
        gs_right, 0.030,
        f"x-axis clipped to {_CLIP_PCT:.0f}th percentile  \u2022  "
        "sorted by median abundance trend (ascending)",
        ha="right", va="center", **style.FONT_ANNOTATION,
    )

    fig.savefig(output_path, dpi=args.dpi, facecolor=style.PAPER)
    logger.info("Saved → %s", output_path)


if __name__ == "__main__":
    main()
