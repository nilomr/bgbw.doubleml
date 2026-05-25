#!/usr/bin/env python3
"""Effort-variable direction heatmap across species.

Shows how each observation-process variable shifts SDM predictions, for
both the occurrence and count model stages.

Each cell encodes a single directional question:
  "When this variable is higher, does the prediction tend to go up or down?"

  colour   Spearman ρ between feature value and SHAP value.
           Red = higher value raises the prediction.
           Blue = higher value lowers it.
           White / pale = no consistent directional effect.

A slim bar below each heatmap shows the stage-wide relative importance
(mean |SHAP|) so you can see which effort variables drive the model most.

Usage
-----
    uv run python scripts/products/make_effort_importance_plot.py \\
        --config configs/local.yaml

    uv run python scripts/products/make_effort_importance_plot.py \\
        --results-dir /mnt/w/projects/BGBW/results/cluster
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from _common import (
    load_config_defaults,
    panel_chip,
    species_label,
)
from bgbw_trends.config import EFFORT_FEATURES
from bgbw_trends.viz import style

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _saturated_cmap(base: mcolors.Colormap, skip: float = 0.13) -> mcolors.Colormap:
    """Remove the pale centre band of a diverging colormap."""
    pts = np.concatenate(
        [
            np.linspace(0.0, 0.5 - skip, 128),
            np.linspace(0.5 + skip, 1.0, 128),
        ]
    )
    return mcolors.LinearSegmentedColormap.from_list("_sat", base(pts))

_EFFORT_LABELS: dict[str, str] = {
    "n_surveys":           "Survey\nintensity",
    "mean_observers":      "Observers /\nsurvey",
    "proportion_adult":    "Adult\nproportion",
    "proportion_online":   "Online\nsubmission",
    "mean_temp_c":         "Weekend\ntemp.",
    "total_rain_mm":       "Weekend\nrainfall",
    "jan_mean_temp_c":     "January\ntemp.",
    "jan_total_rain_mm":   "January\nrainfall",
}

_STAGE_TITLES = {
    "sdm_occ":   "Occurrence stage",
    "sdm_count": "Count stage (given detection)",
}

_FEATURE_GROUPS = [
    ("Survey\nprocess",  ["n_surveys", "mean_observers", "proportion_adult", "proportion_online"]),
    ("Survey\nweather",  ["mean_temp_c", "total_rain_mm"]),
    ("January\nweather", ["jan_mean_temp_c", "jan_total_rain_mm"]),
]

# Colour for NaN / missing cells
_NAN_COLOUR = "#D9D5D0"


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
    p.add_argument("--species", nargs="+", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def _rho_and_magnitude(
    stage_df: pd.DataFrame, feature: str
) -> tuple[float, float]:
    """Spearman ρ (direction) and mean |SHAP| (magnitude) for one feature."""
    sub = stage_df.loc[
        stage_df["feature"] == feature,
        ["shap_value", "feature_value"],
    ].dropna()
    if len(sub) < 5:
        return float("nan"), float("nan")
    rho, _ = spearmanr(sub["feature_value"], sub["shap_value"])
    return float(rho), float(sub["shap_value"].abs().mean())


def _draw_heatmap(
    ax: plt.Axes,
    rho_mat: np.ndarray,          # shape (n_sp, n_feat)
    *,
    species_labels: list[str],
    feature_list: list[str],
    cmap: mcolors.Colormap,
    norm: mcolors.Normalize,
    title: str,
    show_ylabels: bool,
) -> None:
    n_sp, n_feat = rho_mat.shape

    # Build RGBA image; NaN cells get the neutral fill colour
    rgba = cmap(norm(np.where(np.isnan(rho_mat), 0.0, rho_mat)))
    nan_mask = np.isnan(rho_mat)
    rgba[nan_mask] = mcolors.to_rgba(_NAN_COLOUR)

    ax.imshow(
        rgba,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        extent=[-0.5, n_feat - 0.5, n_sp - 0.5, -0.5],
        zorder=2,
    )

    # Fine grid lines between cells
    for x in np.arange(-0.5, n_feat, 1):
        ax.axvline(x, color=style.PAPER, linewidth=0.7, zorder=3)
    for y in np.arange(-0.5, n_sp, 1):
        ax.axhline(y, color=style.PAPER, linewidth=0.7, zorder=3)

    # Heavier separator between feature groups
    offset = 0
    for _gname, gfeats in _FEATURE_GROUPS[:-1]:
        offset += len(gfeats)
        ax.axvline(offset - 0.5, color=style.PAPER, linewidth=2.5, zorder=4)

    ax.set_xlim(-0.5, n_feat - 0.5)
    ax.set_ylim(n_sp - 0.5, -0.5)

    ax.set_xticks(range(n_feat))
    ax.set_xticklabels([""] * n_feat)
    ax.tick_params(axis="x", length=0, pad=0)

    ax.set_yticks(range(n_sp))
    if show_ylabels:
        ax.set_yticklabels(species_labels, **style.FONT_LABEL)
    else:
        ax.tick_params(labelleft=False)
    ax.tick_params(axis="y", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    panel_chip(ax, title, style)


def _draw_importance_bar(
    ax: plt.Axes,
    imp_vals: np.ndarray,     # shape (n_feat,), stage-normalised mean |SHAP|
    *,
    feature_list: list[str],
    show_ylabel: bool,
) -> None:
    n_feat = len(feature_list)

    ax.bar(
        range(n_feat), imp_vals,
        color=style.TEXT_MUTED, width=0.6, zorder=2,
    )

    # Group separators aligned with heatmap
    offset = 0
    for _gname, gfeats in _FEATURE_GROUPS[:-1]:
        offset += len(gfeats)
        ax.axvline(offset - 0.5, color=style.GRIDLINE, linewidth=1.0, zorder=3)

    ax.set_xlim(-0.5, n_feat - 0.5)
    ax.set_ylim(0, imp_vals.max() * 1.30 if imp_vals.max() > 0 else 1.0)
    ax.set_xticks(range(n_feat))
    ax.set_xticklabels(
        [_EFFORT_LABELS.get(f, f) for f in feature_list],
        **style.FONT_LABEL,
        linespacing=1.3,
    )
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", labelsize=7, colors=style.TEXT_MUTED, length=2)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2, integer=False))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))

    if show_ylabel:
        ax.set_ylabel("mean |SHAP|", **style.FONT_LABEL, labelpad=4)

    ax.set_facecolor(style.PAPER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color(style.GRIDLINE)


def main() -> None:
    args = _parse_args()
    style.apply()

    results_dir = Path(args.results_dir)
    output_path = (
        Path(args.output) if args.output
        else results_dir / "figures" / "products" / "effort_importance.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shap_root = results_dir / "shap" / "species"
    found = {p.parent.name: p for p in sorted(shap_root.glob("*/sdm.parquet"))}
    if not found:
        logger.error("No sdm.parquet files found in %s", shap_root)
        sys.exit(1)

    requested = args.species or sorted(found)
    species_list = [s for s in requested if s in found]
    if missing := set(args.species or []) - set(found):
        logger.warning("No SHAP data for: %s", ", ".join(sorted(missing)))
    if not species_list:
        logger.error("No valid species.")
        sys.exit(1)

    if args.species is None:
        species_list = sorted(species_list, key=species_label)

    stages = list(_STAGE_TITLES.keys())
    feature_list = list(EFFORT_FEATURES)
    n_sp   = len(species_list)
    n_feat = len(feature_list)

    # ── Compute Spearman ρ and mean |SHAP| for every (species, feature, stage) ─
    records: list[dict] = []
    for sp in species_list:
        df = pd.read_parquet(found[sp])
        for stage in stages:
            stage_df = df[df["stage"] == stage]
            if stage_df.empty:
                logger.warning("  %s: stage %s not found in sdm.parquet", sp, stage)
                continue
            for feat in feature_list:
                rho, mean_abs = _rho_and_magnitude(stage_df, feat)
                records.append({"species": sp, "feature": feat,
                                 "stage": stage, "rho": rho, "mean_abs": mean_abs})
        logger.info("  Loaded %s", sp)

    summary = pd.DataFrame(records)

    # Stage-normalised importance for the bar strip
    for stage in stages:
        mask = summary["stage"] == stage
        mx = summary.loc[mask, "mean_abs"].max()
        summary.loc[mask, "imp_norm"] = (
            summary.loc[mask, "mean_abs"] / mx if mx > 0 else 0.0
        )

    sp_idx   = {sp: i for i, sp in enumerate(species_list)}
    feat_idx = {f: i  for i, f  in enumerate(feature_list)}

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig_w  = 16.5
    cell_h = max(0.28, min(0.48, 10.0 / max(n_sp, 1)))
    heat_h = n_sp * cell_h
    bar_h  = 1.05
    top_m  = 1.30
    bot_m  = 1.70
    cbar_h = 0.30
    fig_h  = top_m + heat_h + bar_h + cbar_h + bot_m

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=style.PAPER)

    gs_left, gs_right = 0.20, 0.96
    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[heat_h, bar_h],
        hspace=0.04,
        wspace=0.06,
        left=gs_left, right=gs_right,
        top=1.0 - top_m / fig_h,
        bottom=(bot_m + cbar_h) / fig_h,
    )

    cmap     = _saturated_cmap(style.TREND_CMAP)
    norm     = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    sp_labels = [species_label(s) for s in species_list]

    for col_idx, stage in enumerate(stages):
        sub = summary[summary["stage"] == stage]

        # Build ρ matrix (n_sp × n_feat)
        rho_mat = np.full((n_sp, n_feat), np.nan)
        for _, row in sub.iterrows():
            yi = sp_idx.get(row["species"])
            xi = feat_idx.get(row["feature"])
            if yi is not None and xi is not None and not np.isnan(row["rho"]):
                rho_mat[yi, xi] = row["rho"]

        # Mean importance per feature for this stage (may be NaN for missing features)
        imp_by_feat = np.array([
            sub.loc[sub["feature"] == f, "imp_norm"].mean()
            if not sub.loc[sub["feature"] == f, "imp_norm"].dropna().empty else 0.0
            for f in feature_list
        ])

        ax_heat = fig.add_subplot(gs[0, col_idx])
        ax_bar  = fig.add_subplot(gs[1, col_idx])

        _draw_heatmap(
            ax_heat, rho_mat,
            species_labels=sp_labels,
            feature_list=feature_list,
            cmap=cmap, norm=norm,
            title=_STAGE_TITLES[stage],
            show_ylabels=(col_idx == 0),
        )
        _draw_importance_bar(
            ax_bar, imp_by_feat,
            feature_list=feature_list,
            show_ylabel=(col_idx == 0),
        )

    # ── Shared colour scale ───────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar_ax = fig.add_axes([
        0.26, (bot_m * 0.45) / fig_h,
        0.48, (cbar_h * 0.55) / fig_h,
    ])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, colors=style.TEXT_MUTED, length=3)
    cbar.set_ticks([-1, 0, 1])
    cbar.ax.xaxis.set_ticklabels(
        ["fewer detections predicted", "no effect", "more detections predicted"],
    )
    fig.text(
        0.50, (bot_m * 0.45 + cbar_h * 0.55 + 0.20) / fig_h,
        "Spearman ρ  (feature value vs SHAP value)",
        ha="center", va="bottom", **style.FONT_LABEL,
    )

    # ── Header aligned to plot left edge ─────────────────────────────────────
    fig.text(
        gs_left, 0.975,
        "Observation effort  —  direction and importance of effects",
        ha="left", va="top", **style.FONT_TITLE,
    )
    fig.text(
        gs_left, 0.928,
        "colour = Spearman ρ (feature value vs SHAP value)  "
        "•  bar = mean |SHAP| (stage-normalised)",
        ha="left", va="top", **style.FONT_SUBTITLE,
    )
    fig.add_artist(
        mpl.lines.Line2D(
            [gs_left, gs_right], [0.905, 0.905],
            transform=fig.transFigure, color=style.GRIDLINE, linewidth=0.8,
        )
    )
    fig.text(
        gs_right, 0.030,
        "SHAP values from marginal SDM forests  "
        "•  ρ computed across SHAP background samples per species",
        ha="right", va="center", **style.FONT_ANNOTATION,
    )

    fig.savefig(output_path, dpi=args.dpi, facecolor=style.PAPER)
    logger.info("Saved → %s", output_path)


if __name__ == "__main__":
    main()
