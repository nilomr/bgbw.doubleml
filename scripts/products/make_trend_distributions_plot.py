#!/usr/bin/env python3
"""Paired trend distribution plot for two species.

Two vertically stacked panels, one per species, showing the spatial
distribution of per-cell percent-per-year trend estimates across ~80k
1 km cells for all three model components (occurrence, count, abundance).

Each filled step histogram shows how cell-level estimates vary across
the landscape — the width reflects spatial heterogeneity, not trend
uncertainty.  Dashed vertical lines mark the UK-wide medians.

Usage
-----
    uv run python scripts/products/make_trend_distributions_plot.py \\
        --results-dir /mnt/w/projects/BGBW/results/cluster

    uv run python scripts/products/make_trend_distributions_plot.py \\
        --results-dir /mnt/w/projects/BGBW/results/cluster \\
        --species goldfinch greenfinch
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib as mpl
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

_SPECIES_DEFAULT = ["goldfinch", "greenfinch"]

_METRICS: list[tuple[str, str, str]] = [
    ("occ_ppy",   "Occurrence", "#2C6E91"),
    ("count_ppy", "Count",      "#C5672F"),
    ("abd_ppy",   "Abundance",  "#5A7F62"),
]

# Clip x-axis to this percentile of the combined distribution
_CLIP_PCT = 99.5
# Histogram bins per panel
_N_BINS = 70


def _parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    cfg = load_config_defaults(pre_args.config)

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default=None)
    p.add_argument(
        "--results-dir", default=cfg.get("results_dir", "output"),
    )
    p.add_argument(
        "--species", nargs="+", default=_SPECIES_DEFAULT,
        metavar="SPECIES",
        help="exactly two species slugs (default: goldfinch greenfinch)",
    )
    p.add_argument("--output", default=None)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def _draw_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    sp_label: str,
    x_lim: tuple[float, float],
    show_xlabel: bool,
) -> None:
    """Draw one species panel with three overlapping step histograms."""
    for metric, label, color in _METRICS:
        vals = df[metric].dropna().to_numpy() * 100.0
        if len(vals) < 5:
            logger.warning("  %s: too few values for %s", sp_label, metric)
            continue

        clipped = vals[(vals >= x_lim[0]) & (vals <= x_lim[1])]
        med = float(np.median(vals))

        # Filled step histogram (subtle fill first, then outline on top)
        ax.hist(
            clipped, bins=_N_BINS, density=True,
            histtype="stepfilled", color=color, alpha=0.12,
            linewidth=0, zorder=2,
        )
        ax.hist(
            clipped, bins=_N_BINS, density=True,
            histtype="step", color=color, linewidth=1.8,
            alpha=0.90, zorder=3,
        )

        # Dashed median line
        ax.axvline(
            med, color=color, linewidth=1.2,
            linestyle=(0, (3, 2)), alpha=0.85, zorder=4,
        )

        # Legend proxy
        ax.plot(
            [], [], color=color, linewidth=2.2,
            label=f"{label}  {med:+.2f} %/yr",
        )

    # Zero reference
    ax.axvline(0, color=style.TEXT_MUTED, linewidth=0.7, linestyle=(0, (4, 3)), zorder=1)

    ax.set_xlim(*x_lim)
    ax.set_ylim(bottom=0)

    ax.set_ylabel("Density", **style.FONT_LABEL, labelpad=5)
    if show_xlabel:
        ax.set_xlabel("% per year", **style.FONT_LABEL, labelpad=5)

    ax.tick_params(axis="x", colors=style.TEXT_MUTED, labelsize=8.5, length=3)
    ax.tick_params(axis="y", colors=style.TEXT_MUTED, labelsize=8.0, length=3)
    ax.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, _: f"{x:+.1f}")
    )

    ax.set_facecolor(style.PAPER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(style.GRIDLINE)
    ax.spines["bottom"].set_color(style.GRIDLINE)

    ax.legend(
        frameon=False, fontsize=8.5, labelcolor=style.TEXT_MUTED,
        loc="upper left", handlelength=1.6, handletextpad=0.6,
    )

    panel_chip(ax, sp_label, style)


def main() -> None:
    args = _parse_args()
    style.apply()

    if len(args.species) != 2:
        logger.error("Exactly two species are required; got %d", len(args.species))
        sys.exit(1)

    sp_a, sp_b = args.species
    results_dir = Path(args.results_dir)
    ensemble_root = results_dir / "ensemble" / "species"

    # ── Load data ─────────────────────────────────────────────────────────────
    data: dict[str, pd.DataFrame] = {}
    for sp in (sp_a, sp_b):
        path = ensemble_root / sp / "ensemble.parquet"
        if not path.exists():
            logger.error("No ensemble parquet for %s at %s", sp, path)
            sys.exit(1)
        data[sp] = pd.read_parquet(path)
        for metric, _, _ in _METRICS:
            med = np.median(data[sp][metric].dropna()) * 100
            logger.info("  %s  %s  median = %+.2f%%", sp, metric, med)

    # ── Shared x-axis range (combined P(1-clip) – P(clip)) ───────────────────
    all_vals: list[float] = []
    for sp in (sp_a, sp_b):
        for metric, _, _ in _METRICS:
            all_vals.extend((data[sp][metric].dropna().to_numpy() * 100).tolist())
    arr = np.asarray(all_vals)
    lo = float(np.percentile(arr, 100 - _CLIP_PCT))
    hi = float(np.percentile(arr, _CLIP_PCT))
    pad = (hi - lo) * 0.04
    x_lim = (lo - pad, hi + pad)

    # ── Output path ───────────────────────────────────────────────────────────
    slug = "_".join(args.species)
    output_path = (
        Path(args.output) if args.output
        else results_dir / "figures" / "products" / f"trend_distributions_{slug}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig_w  = 10.5
    fig_h  = 8.0
    gs_left, gs_right = 0.09, 0.96

    fig, axes = plt.subplots(
        2, 1, figsize=(fig_w, fig_h),
        sharex=True, facecolor=style.PAPER,
    )
    fig.subplots_adjust(
        left=gs_left, right=gs_right,
        top=0.77, bottom=0.11,
        hspace=0.12,
    )

    for i, (ax, sp) in enumerate(zip(axes, (sp_a, sp_b))):
        _draw_panel(
            ax, data[sp],
            sp_label=species_label(sp),
            x_lim=x_lim,
            show_xlabel=(i == 1),
        )

    # ── Header ────────────────────────────────────────────────────────────────
    sp_a_label = species_label(sp_a)
    sp_b_label = species_label(sp_b)

    fig.text(
        gs_left, 0.975,
        f"Trend distributions  \u2014  {sp_a_label} and {sp_b_label}",
        ha="left", va="top", **style.FONT_TITLE,
    )
    fig.text(
        gs_left, 0.928,
        "histogram = per-cell %/yr estimates across ~80k 1 km cells  "
        "\u2022  dashed line = UK median",
        ha="left", va="top", **style.FONT_SUBTITLE,
    )
    fig.text(
        gs_left, 0.900,
        "spread is spatial (cell-to-cell variation), not trend uncertainty",
        ha="left", va="top", **style.FONT_SUBTITLE,
    )
    fig.add_artist(
        mpl.lines.Line2D(
            [gs_left, gs_right], [0.872, 0.872],
            transform=fig.transFigure, color=style.GRIDLINE, linewidth=0.8,
        )
    )
    fig.text(
        gs_right, 0.030,
        f"x-axis clipped to {_CLIP_PCT:.4g}th percentile of combined distribution",
        ha="right", va="center", **style.FONT_ANNOTATION,
    )

    fig.savefig(output_path, dpi=args.dpi, facecolor=style.PAPER)
    logger.info("Saved \u2192 %s", output_path)


if __name__ == "__main__":
    main()
