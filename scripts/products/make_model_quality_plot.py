#!/usr/bin/env python3
"""Model quality dot plot — occurrence AUC and count R² by species.

Two panels, species sorted by median abundance trend (ascending).

Usage
-----
    uv run python scripts/products/make_model_quality_plot.py \\
        --config configs/local.yaml

    uv run python scripts/products/make_model_quality_plot.py \\
        --results-dir /mnt/w/projects/BGBW/results/cluster
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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


def _draw_dot_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    species_order: list[str],
    key: str,
    *,
    show_labels: bool,
    ref_line: float | None = None,
) -> None:
    n = len(species_order)

    for i, sp in enumerate(species_order):
        val = float(data.loc[data["species"] == sp, key].iloc[0])
        ax.scatter(val, i, s=55, color=style.TEXT_MUTED, zorder=3, linewidths=0)

    for i in range(n):
        ax.axhline(i, color=style.GRIDLINE, linewidth=0.45, zorder=0)

    if ref_line is not None:
        ax.axvline(ref_line, color=style.GRIDLINE, linewidth=0.8,
                   linestyle=(0, (4, 3)), zorder=1)

    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(range(n))
    if show_labels:
        ax.set_yticklabels(
            [species_label(sp) for sp in species_order],
            **style.FONT_LABEL,
        )
    else:
        ax.tick_params(labelleft=False)

    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=style.TEXT_MUTED, labelsize=8.5, length=3)
    ax.set_facecolor(style.PAPER)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(style.GRIDLINE)


def main() -> None:
    args = _parse_args()
    style.apply()

    results_dir = Path(args.results_dir)
    output_path = (
        Path(args.output) if args.output
        else results_dir / "figures" / "products" / "model_quality.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    diag_path = results_dir / "diagnostics" / "pipeline_model_diagnostics.csv"
    ensemble_root = results_dir / "ensemble" / "species"

    if not diag_path.exists():
        logger.error("Diagnostics CSV not found: %s", diag_path)
        sys.exit(1)

    diag = pd.read_csv(diag_path)
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
        diag_sp = diag[diag["species"] == sp]
        if diag_sp.empty:
            logger.warning("No diagnostics for %s — skipping", sp)
            continue
        ens = pd.read_parquet(ens_files[sp])
        abd_ppy = float(np.median(ens["abd_ppy"].dropna()) * 100.0)
        rows.append({
            "species":  sp,
            "occ_auc":  float(diag_sp["occurrence_auc"].iloc[0]),
            "count_r2": float(diag_sp["count_r2_log1p"].iloc[0]),
            "abd_ppy":  abd_ppy,
        })
        logger.info("  %-24s  occ_AUC=%.3f  count_R²=%.3f", sp,
                    rows[-1]["occ_auc"], rows[-1]["count_r2"])

    data = pd.DataFrame(rows).dropna()
    if data.empty:
        logger.error("No complete records.")
        sys.exit(1)

    species_order = data.sort_values("abd_ppy")["species"].tolist()
    n = len(species_order)

    fig_h = max(4.0, 0.42 * n + 2.2)
    fig, axes = plt.subplots(
        1, 2, figsize=(9.0, fig_h),
        sharey=True, facecolor=style.PAPER,
    )
    fig.subplots_adjust(left=0.22, right=0.97, top=0.80, bottom=0.12, wspace=0.08)

    _draw_dot_panel(axes[0], data, species_order, "occ_auc",
                    show_labels=True, ref_line=None)
    axes[0].set_xlabel("AUC", **style.FONT_LABEL, labelpad=5)
    panel_chip(axes[0], "Occurrence AUC", style)

    _draw_dot_panel(axes[1], data, species_order, "count_r2",
                    show_labels=False, ref_line=None)
    axes[1].set_xlabel("R²  (log1p scale)", **style.FONT_LABEL, labelpad=5)
    panel_chip(axes[1], "Count R²  (given detection)", style)

    header_block(
        fig,
        "Model quality  —  SDM discrimination by species",
        "sorted by median abundance trend (ascending)  •  "
        "occurrence AUC and count R² from pipeline diagnostics",
        style,
    )
    footer_note(fig, "Count R² evaluated on log1p-transformed counts", style)

    fig.savefig(output_path, dpi=args.dpi, facecolor=style.PAPER)
    logger.info("Saved → %s", output_path)


if __name__ == "__main__":
    main()
