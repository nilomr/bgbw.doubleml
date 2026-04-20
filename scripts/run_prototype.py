#!/usr/bin/env python3
"""Prototype analysis: two-part DML + SDM for BGBW data.

Runs ``bgbw_trends.pipeline.train_predict_species`` per species, which in
one pass produces:

* the effort-adjusted SDM (distribution + relative abundance maps), and
* the spatially varying trend τ(W) for occurrence, count and the
  chain-rule abundance rate (Fink §3.2).

Optional diagnostics: null-trend simulation for residual confounding and
a subsample ensemble for grid-level CATE uncertainty.

Usage:
    uv run python scripts/run_prototype.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from bgbw_trends.config import RunConfig
from bgbw_trends.console import (
    console,
    make_progress,
    make_spinner,
    print_config_panel,
    print_rc_summary,
    print_sdm_table,
    section,
    setup_rich_logging,
    timed_step,
)
from bgbw_trends.data import (
    assemble_analysis_data,
    assemble_harmonized_data,
    load_embeddings,
)
from bgbw_trends.features.effort import compute_reference_effort
from bgbw_trends.features.habitat import reduce_embeddings_pca
from bgbw_trends.pipeline import train_predict_species
from bgbw_trends.plotting import (
    plot_raw_vs_fitted,
    plot_sdm_prediction_maps,
    plot_trend_map,
    plot_yearly_trend,
)
from bgbw_trends.trends import (
    diagnose_residual_confounding,
    fit_ensemble_trend,
)

setup_rich_logging(logging.INFO)
logger = logging.getLogger(__name__)


def _build_grid(df: pd.DataFrame) -> pd.DataFrame:
    """One row per bng_ref using its most recent (year) habitat snapshot."""
    return (
        df.sort_values("year")
        .drop_duplicates(subset=["bng_ref"], keep="last")
        .reset_index(drop=True)
    )


def main() -> None:
    # ── Configuration ───────────────────────────────────────────────────
    run_ensemble = True  # slow — set True for uncertainty quantification

    cfg = RunConfig(
        species=["chaffinch"],
        years=[2015, 2025],
        embedding_year=2019,
        output_dir=Path("output/prototype"),
        n_folds=3,
        n_rep=3,
        n_pca_components=5,
        n_estimators=300,
        cf_n_estimators=500,
        cf_min_samples_leaf=50,
    )
    out = cfg.output_paths()

    console.print()
    print_config_panel(cfg, run_ensemble)
    console.print()

    # ── Data assembly ────────────────────────────────────────────────────
    section("Data preparation")
    with timed_step("Loading and aggregating surveys + embeddings"):
        df = assemble_analysis_data(cfg)

    console.print(
        f"  [dim]rows:[/dim] [cyan]{len(df):,}[/cyan]"
        f"  [dim]cells:[/dim] [cyan]{df['bng_ref'].nunique():,}[/cyan]"
        f"  [dim]years:[/dim] [cyan]{df['year'].nunique()}[/cyan]"
    )

    # ── Embedding PCA ────────────────────────────────────────────────────
    embedding_pca = None
    if cfg.n_pca_components:
        with timed_step(
            f"PCA: 128-dim embeddings -> {cfg.n_pca_components} components per stat"
        ):
            df, embedding_pca = reduce_embeddings_pca(df, cfg.n_pca_components)

    n_emb = sum(1 for c in df.columns if c.startswith("emb_"))
    console.print(f"  [dim]embedding features after PCA:[/dim] [cyan]{n_emb}[/cyan]")

    for sp in cfg.species:
        det_rate = (df[sp] > 0).mean()
        mean_count = df.loc[df[sp] > 0, sp].mean()
        console.print(
            f"  [dim]{sp}:[/dim]"
            f"  detection rate [cyan]{det_rate * 100:.1f}%[/cyan]"
            f"  mean count (given detection) [cyan]{mean_count:.2f}[/cyan]"
        )

    # ── Reference effort & grid ─────────────────────────────────────────
    cfg.reference_effort = compute_reference_effort(df)
    grid = _build_grid(df)
    console.print(
        f"  [dim]grid:[/dim] [cyan]{len(grid):,}[/cyan] cells  "
        f"[dim]reference effort:[/dim] n_surveys=[cyan]"
        f"{cfg.reference_effort.n_surveys:.2f}[/cyan]"
    )

    # ── Two-part DML trend + SDM (Fink §3.2) ────────────────────────────
    section("Two-part DML trend + SDM")
    console.print()
    predictions = {}
    with make_progress() as prog:
        task = prog.add_task("train_predict_species", total=len(cfg.species))
        for sp in cfg.species:
            prog.update(task, description=f"train_predict_species  [{sp}]")
            pred = train_predict_species(
                df,
                df,
                species=sp,
                cfg=cfg,
                grid=grid,
                reference_effort=cfg.reference_effort,
            )
            predictions[sp] = pred
            prog.advance(task)

    for sp, pred in predictions.items():
        console.print(
            f"  [dim]{sp}:[/dim]"
            f"  mean P(occur)=[cyan]{np.nanmean(pred.p_occurrence):.3f}[/cyan]"
            f"  mean E[abund]=[cyan]{np.nanmean(pred.e_abundance):.3f}[/cyan]"
            f"  abd_ppy=[cyan]{np.nanmean(pred.abd_ppy) * 100:+.2f}%[/cyan]"
        )

    sdm_rows = []
    diagnostics_rows = []
    for sp, pred in predictions.items():
        diag = pred.diagnostics
        sdm_rows.extend(
            [
                (sp, "occurrence RF", "Brier", diag.occurrence_brier),
                (sp, "occurrence RF", "ROC AUC", diag.occurrence_auc),
                (sp, "occurrence RF", "PR AUC", diag.occurrence_pr_auc),
                (sp, "count RF", "R2 log1p", diag.count_r2),
                (sp, "count RF", "RMSE log1p", diag.count_rmse),
                (sp, "count RF", "MAE log1p", diag.count_mae),
                (sp, "year RF (occ)", "R2", diag.propensity_occ_r2),
                (sp, "year RF (occ)", "RMSE", diag.propensity_occ_rmse),
                (sp, "year RF (count)", "R2", diag.propensity_count_r2),
                (sp, "year RF (count)", "RMSE", diag.propensity_count_rmse),
            ]
        )
        diagnostics_rows.append(
            {
                "species": sp,
                "occurrence_brier": diag.occurrence_brier,
                "occurrence_auc": diag.occurrence_auc,
                "occurrence_pr_auc": diag.occurrence_pr_auc,
                "count_r2_log1p": diag.count_r2,
                "count_rmse_log1p": diag.count_rmse,
                "count_mae_log1p": diag.count_mae,
                "propensity_occ_r2": diag.propensity_occ_r2,
                "propensity_occ_rmse": diag.propensity_occ_rmse,
                "propensity_count_r2": diag.propensity_count_r2,
                "propensity_count_rmse": diag.propensity_count_rmse,
                "n_train_occ": pred.n_train_occ,
                "n_train_count": pred.n_train_count,
            }
        )

    console.print()
    print_sdm_table(sdm_rows, title="Pipeline RF diagnostics")
    pd.DataFrame(diagnostics_rows).to_csv(
        out.diagnostics_dir / "pipeline_model_diagnostics.csv", index=False
    )

    for sp, pred in predictions.items():
        out_df = pd.DataFrame(
            {
                "bng_ref": pred.bng_ref,
                "p_occurrence": pred.p_occurrence,
                "e_count_given_occ": pred.e_count_given_occ,
                "e_abundance": pred.e_abundance,
                "occ_ppy": pred.occ_ppy,
                "count_ppy": pred.count_ppy,
                "abd_ppy": pred.abd_ppy,
                "occ_sigma": pred.occ_sigma,
                "count_sigma": pred.count_sigma,
            }
        )
        species_pred_path = out.species_prediction_file(sp)
        species_pred_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(species_pred_path)

        species_fig_dir = out.species_figures_dir(sp)
        species_fig_dir.mkdir(parents=True, exist_ok=True)

        plot_sdm_prediction_maps(pred, output_path=out.species_sdm_figure(sp))
        plot_trend_map(
            pred,
            metric="abd_ppy",
            output_path=out.species_trend_figure(sp, "abd_ppy"),
        )
        plot_trend_map(
            pred,
            metric="occ_ppy",
            output_path=out.species_trend_figure(sp, "occ_ppy"),
        )
        plot_yearly_trend(pred, output_path=out.species_yearly_trend_figure(sp))
        plot_raw_vs_fitted(pred, output_path=out.species_raw_vs_fitted_figure(sp))

    # ── Residual confounding diagnostic ─────────────────────────────────
    section("Residual confounding diagnostic")
    console.print()
    n_rc = 5
    with make_spinner() as spin:
        spin.add_task(
            f"Null-trend simulation [{cfg.species[0]}]  (n_simulations={n_rc})",
            total=None,
        )
        rc = diagnose_residual_confounding(
            df,
            species=cfg.species[0],
            cfg=cfg,
            grid=grid,
            reference_effort=cfg.reference_effort,
            n_simulations=n_rc,
        )
    print_rc_summary(rc.bias_ate_occ, rc.bias_ate_count, rc.n_successful, n_rc)

    # ── Ensemble (Johnston §3.4) ────────────────────────────────────────
    section("Ensemble trends")
    if not run_ensemble:
        console.print("  [dim]skipped (run_ensemble=False)[/dim]")
    else:
        n_ensemble = 10
        console.print(
            f"  [dim]{n_ensemble} members  •  subsample fraction 0.75  •  "
            f"independent occ/count draws[/dim]"
        )
        console.print()
        with make_spinner() as spin:
            spin.add_task(
                f"Ensemble fitting [{cfg.species[0]}]  ({n_ensemble} members)",
                total=None,
            )
            harmonized = assemble_harmonized_data(cfg)
            embeddings = load_embeddings(cfg.embedding_year)
            ens = fit_ensemble_trend(
                harmonized=harmonized,
                embeddings=embeddings,
                cfg=cfg,
                species=cfg.species[0],
                grid=grid,
                reference_effort=cfg.reference_effort,
                n_ensemble=n_ensemble,
                subsample_fraction=0.75,
                pca_model=embedding_pca,
            )
        ensemble_path = out.species_ensemble_file(cfg.species[0])
        ensemble_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "bng_ref": ens.bng_ref,
                "abd_ppy": ens.abd_ppy,
                "abd_ppy_lower": ens.abd_ppy_lower,
                "abd_ppy_upper": ens.abd_ppy_upper,
                "occ_ppy": ens.occ_ppy,
                "count_ppy": ens.count_ppy,
                "p_occurrence": ens.p_occurrence,
                "e_abundance": ens.e_abundance,
            }
        ).to_parquet(ensemble_path)
        console.print(
            f"  [dim]{ens.n_successful}/{ens.n_ensemble} members successful[/dim]"
        )

    console.print()
    console.print(f"[dim]All results written to[/dim] [cyan]{out.root}[/cyan]")
    console.print()


if __name__ == "__main__":
    main()
