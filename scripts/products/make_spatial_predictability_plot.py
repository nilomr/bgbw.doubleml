#!/usr/bin/env python3
"""Species trends plot — abundance trend and spatial heterogeneity.

Two panels, species sorted by median abundance trend (ascending).

  Panel 1  Abundance trend — median per-cell abd_ppy.
           Lollipop from zero. Colour encodes direction and magnitude.

  Panel 2  Spatial heterogeneity — IQR (Q75−Q25) of per-cell abd_ppy across the UK.
           Horizontal bar from zero; wide = heterogeneous across the UK.
           Dot marks the UK-wide median.

Usage
-----
    uv run python scripts/products/make_spatial_predictability_plot.py \\
        --config configs/local.yaml

    uv run python scripts/products/make_spatial_predictability_plot.py \\
        --results-dir /mnt/w/projects/BGBW/results/cluster
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from _common import (
    footer_note,
    header_block,
    load_config_defaults,
    panel_chip,
    species_label,
)
from bgbw_trends.viz import style

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _saturated_cmap(base: mcolors.Colormap, skip: float = 0.13) -> mcolors.Colormap:
    """Remove the pale centre band of a diverging colormap."""
    pts = np.concatenate([
        np.linspace(0.0, 0.5 - skip, 128),
        np.linspace(0.5 + skip, 1.0, 128),
    ])
    return mcolors.LinearSegmentedColormap.from_list("_sat", base(pts))


def _parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    cfg = load_config_defaults(pre_args.config)

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default=None)
    p.add_argument("--results-dir", default=cfg.get("results_dir", "output"))
    p.add_argument("--species", nargs="+", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    style.apply()

    results_dir = Path(args.results_dir)
    output_path = (
        Path(args.output) if args.output
        else results_dir / "figures" / "products" / "species_trends.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ensemble_root = results_dir / "ensemble" / "species"
    ens_files = {p.parent.name: p for p in sorted(ensemble_root.glob("*/ensemble.parquet"))}
    if not ens_files:
        logger.error("No ensemble parquets found in %s", ensemble_root)
        sys.exit(1)

    species_list = [s for s in (args.species or sorted(ens_files)) if s in ens_files]
    if not species_list:
        logger.error("No valid species.")
        sys.exit(1)

    rows: list[dict] = []
    for sp in species_list:
        ens = pd.read_parquet(ens_files[sp])
        abd     = ens["abd_ppy"].dropna() * 100.0
        abd_med = float(np.median(abd))
        q25     = float(abd.quantile(0.25))
        q75     = float(abd.quantile(0.75))
        iqr     = q75 - q25
        rows.append({
            "species":  sp,
            "abd_med":  abd_med,
            "q25":      q25,
            "q75":      q75,
            "iqr":      iqr,
        })
        logger.info(
            "  %-24s  med=%+6.2f%%  IQR=%.2f%%",
            sp, abd_med, iqr,
        )

    data = pd.DataFrame(rows).dropna()
    if data.empty:
        logger.error("No complete records.")
        sys.exit(1)

    species_order = data.sort_values("abd_med")["species"].tolist()
    n = len(species_order)

    # Colourmap: cap range at 75th-pct of |trend| so mid-range species are
    # clearly coloured; remove pale centre band so near-zero isn't invisible.
    vext = float(max(data["abd_med"].abs().quantile(0.80), 2.0))
    norm = mcolors.TwoSlopeNorm(vmin=-vext, vcenter=0.0, vmax=vext)
    cmap = _saturated_cmap(style.TREND_CMAP)

    fig_h = max(4.0, 0.42 * n + 2.2)
    fig, axes = plt.subplots(
        1, 2, figsize=(12.0, fig_h),
        sharey=True, facecolor=style.PAPER,
        gridspec_kw={"width_ratios": [1.6, 1]},
    )
    fig.subplots_adjust(left=0.20, right=0.88, top=0.80, bottom=0.12, wspace=0.08)

    # ── Panel 1: abundance trend + CI ────────────────────────────────────────
    ax = axes[0]
    for i, sp in enumerate(species_order):
        row  = data.loc[data["species"] == sp].iloc[0]
        med  = float(row["abd_med"])
        color = cmap(norm(np.clip(med, -vext, vext)))

        # Lollipop stem
        ax.plot([0, med], [i, i], color=color, linewidth=2.0, alpha=0.80,
                solid_capstyle="round", zorder=2)
        # Endpoint dot
        ax.scatter(med, i, s=52, color=color, zorder=3, linewidths=0)

    for i in range(n):
        ax.axhline(i, color=style.GRIDLINE, linewidth=0.45, zorder=0)
    ax.axvline(0, color=style.TEXT_MUTED, linewidth=0.75, linestyle=(0, (4, 3)), zorder=1)

    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(range(n))
    ax.set_yticklabels([species_label(sp) for sp in species_order], **style.FONT_LABEL)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=style.TEXT_MUTED, labelsize=8.5, length=3)
    ax.xaxis.set_major_formatter(
        plt.matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:+.0f}")
    )
    x_lo = float(data["abd_med"].min())
    x_hi = float(data["abd_med"].max())
    pad = (x_hi - x_lo) * 0.12 or 0.5
    axes[0].set_xlim(x_lo - pad, x_hi + pad)

    ax.set_xlabel("% per year", **style.FONT_LABEL, labelpad=5)
    ax.set_facecolor(style.PAPER)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(style.GRIDLINE)
    panel_chip(ax, "Abundance trend  (UK median)", style)

    # ── Panel 2: spatial IQR ──────────────────────────────────────────────────
    ax = axes[1]
    for i, sp in enumerate(species_order):
        row   = data.loc[data["species"] == sp].iloc[0]
        iqr   = float(row["iqr"])
        med   = float(row["abd_med"])
        color = cmap(norm(np.clip(med, -vext, vext)))
        # IQR bar from zero
        ax.plot([0, iqr], [i, i], color=color, linewidth=2.5,
                solid_capstyle="round", alpha=0.80, zorder=2)
        # Dot at IQR value
        ax.scatter(iqr, i, s=52, color=color, zorder=3, linewidths=0)

    for i in range(n):
        ax.axhline(i, color=style.GRIDLINE, linewidth=0.45, zorder=0)

    iqr_hi = float(data["iqr"].max())
    pad_iqr = iqr_hi * 0.08
    ax.set_xlim(0, iqr_hi + pad_iqr)

    ax.tick_params(labelleft=False, axis="y", length=0)
    ax.tick_params(axis="x", colors=style.TEXT_MUTED, labelsize=8.5, length=3)
    ax.xaxis.set_major_formatter(
        plt.matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.1f}")
    )
    ax.set_xlabel("% per year", **style.FONT_LABEL, labelpad=5)
    ax.set_facecolor(style.PAPER)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(style.GRIDLINE)
    panel_chip(ax, "Spatial heterogeneity  (IQR of cell trends)", style)

    # ── Colourbar ─────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar_ax = fig.add_axes([0.905, 0.22, 0.018, 0.46])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.outline.set_visible(False)
    cbar.ax.set_ylabel("Abundance trend\n(% / yr)", **style.FONT_LABEL, labelpad=7)
    cbar.ax.tick_params(labelsize=8, colors=style.TEXT_MUTED, length=3)
    formatter = plt.matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:+.0f}")
    cbar.ax.yaxis.set_major_formatter(formatter)

    header_block(
        fig,
        "Species trends  —  direction and spatial heterogeneity",
        "sorted by UK-wide median abundance trend (ascending)  •  "
        "IQR = Q75 − Q25 of per-cell abd_ppy",
        style,
    )
    footer_note(
        fig,
        "Abundance trend = UK-wide median of per-cell abd_ppy  •  "
        "IQR = Q75 − Q25 of per-cell abd_ppy across ~80k 1 km cells",
        style,
    )

    fig.savefig(output_path, dpi=args.dpi, facecolor=style.PAPER)
    logger.info("Saved → %s", output_path)


if __name__ == "__main__":
    main()
