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
    uv run python scripts/pipeline/run_prototype.py
"""

from __future__ import annotations

import argparse
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
from bgbw_trends.data import assemble_analysis_data
from bgbw_trends.features.effort import compute_reference_effort
from bgbw_trends.features.habitat import reduce_embeddings_pca
from bgbw_trends.interpret import (
    explain_cate_surrogate,
    explain_tree_model,
    plot_shap_panel,
)
from bgbw_trends.pipeline import train_predict_species
from bgbw_trends.tuning import tune_nuisance_for_species
from bgbw_trends.plotting import (
    plot_prediction_distribution_summary,
    plot_raw_vs_fitted,
    plot_sdm_prediction_maps,
    plot_trend_ppy_panels,
    plot_yearly_trend,
)
from bgbw_trends.trends import (
    diagnose_residual_confounding,
    fit_ensemble_trend,
    split_cell_years,
)

setup_rich_logging(logging.INFO)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse CLI args so -h/--help exits safely without running the pipeline."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--species",
        nargs="+",
        metavar="SPECIES",
        default=None,
        help=(
            "One or more species to process (e.g. --species blackcap robin). "
            "Defaults to the list hard-coded in this script."
        ),
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        metavar="YEAR",
        default=None,
        help=("Years to include (e.g. --years 2017 2018 2019). Defaults to 2017–2024."),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Root output directory. Defaults to output/prototype.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Outer process-level workers for the ensemble and residual-"
            "confounding loops. 1 = sequential. On a cluster node, set to "
            "the number of pipeline replicas you can run concurrently."
        ),
    )
    parser.add_argument(
        "--inner-n-jobs",
        type=int,
        default=-1,
        metavar="N",
        help=(
            "Threads per individual RF / CausalForest fit. -1 = all cores "
            "(default). When --n-workers > 1, this is auto-capped to "
            "cores // n_workers to avoid CPU oversubscription unless an "
            "explicit positive value is given."
        ),
    )
    parser.add_argument(
        "--n-ensemble",
        type=int,
        default=10,
        metavar="N",
        help=(
            "Ensemble members for trend uncertainty. Default 10 — tractable "
            "on a single machine, but only an exploratory CI. Fink et al. "
            "(2023) and Johnston et al. (2025) both use 100; raise to ≥25 "
            "for a usable percentile CI and to 100 with --n-workers on a "
            "cluster for publication-grade uncertainty."
        ),
    )
    parser.add_argument(
        "--n-rc-sims",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Null-trend simulations for residual confounding. Default 5 — "
            "a smoke check. The trendswf vignette uses 100; ≥25 gives a "
            "stable bias estimate. Each sim fits the full pipeline."
        ),
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Skip the ensemble step (uncertainty quantification).",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help=(
            "Tune min_samples_leaf per nuisance forest (OOB R²) once per "
            "species, then reuse those hyperparameters for the main fit, "
            "ensemble, and residual-confounding sims. Costs ~16 extra "
            "nuisance fits per species (4 candidates × 4 forests)."
        ),
    )
    return parser.parse_args()


def _build_grid(df: pd.DataFrame) -> pd.DataFrame:
    """One row per bng_ref using its most recent (year) habitat snapshot."""
    return (
        df.sort_values("year")
        .drop_duplicates(subset=["bng_ref"], keep="last")
        .reset_index(drop=True)
    )


def main() -> None:
    args = _parse_args()

    # ── Configuration ───────────────────────────────────────────────────
    run_ensemble = not args.no_ensemble

    cfg = RunConfig(
        species=args.species or ["blackbird"],
        years=args.years or list(range(2017, 2025, 2)),
        output_dir=args.output_dir or Path("output/prototype"),
        n_pca_components=15,
        n_estimators=300,
        cf_n_estimators=300,
        cf_min_samples_leaf=50,
        n_workers=args.n_workers,
        inner_n_jobs=args.inner_n_jobs,
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

    # ── Per-species hyperparameter tuning ───────────────────────────────
    # When --tune is set, run the OOB-R² tuner once per species on the full
    # df. The resulting params flow through the main fit, ensemble, and
    # residual-confounding sims, so every replicate uses the same leaf
    # sizes (matches the "tune once, freeze, reuse" pattern in the dev
    # guide; cluster-style per-replicate tuning is not implemented here).
    tuned_params: dict[str, object] = {}
    if args.tune:
        section("Hyperparameter tuning (nuisance forests)")
        console.print()
        for sp in cfg.species:
            with timed_step(f"  tune  [{sp}]"):
                tuned_params[sp] = tune_nuisance_for_species(
                    df, df, species=sp, cfg=cfg
                )
            tp = tuned_params[sp]
            console.print(
                f"  [dim]{sp}:[/dim]"
                f"  occ marg leaf=[cyan]{tp.occ_marginal}[/cyan]"
                f" mf=[cyan]{tp.occ_marginal_max_features}[/cyan]"
                f" (R²=[cyan]{tp.occ_marginal_oob_r2:+.3f}[/cyan])"
                f"  occ prop leaf=[cyan]{tp.occ_propensity}[/cyan]"
                f" mf=[cyan]{tp.occ_propensity_max_features}[/cyan]"
                f" (R²=[cyan]{tp.occ_propensity_oob_r2:+.3f}[/cyan])"
            )
            console.print(
                f"  [dim]{' ' * len(sp)} [/dim]"
                f"  count marg leaf=[cyan]{tp.count_marginal}[/cyan]"
                f" mf=[cyan]{tp.count_marginal_max_features}[/cyan]"
                f" (R²=[cyan]{tp.count_marginal_oob_r2:+.3f}[/cyan])"
                f"  count prop leaf=[cyan]{tp.count_propensity}[/cyan]"
                f" mf=[cyan]{tp.count_propensity_max_features}[/cyan]"
                f" (R²=[cyan]{tp.count_propensity_oob_r2:+.3f}[/cyan])"
            )

    # ── Two-part DML trend + SDM (Fink §3.2) ────────────────────────────
    # Prototype path — Use fit_ensemble_trend for final trend inference.
    section("Two-part DML trend + SDM")
    console.print()
    predictions = {}
    trained_models = {}
    rng = np.random.default_rng(cfg.__dict__.get("seed", 42))
    with make_progress() as prog:
        task = prog.add_task("train_predict_species", total=len(cfg.species))
        for sp in cfg.species:
            prog.update(task, description=f"train_predict_species  [{sp}]")
            df_occ, df_count = split_cell_years(df, rng)
            pred, models = train_predict_species(
                df_occ,
                df_count,
                species=sp,
                cfg=cfg,
                grid=grid,
                reference_effort=cfg.reference_effort,
                return_models=True,
                nuisance_params=tuned_params.get(sp),
            )
            predictions[sp] = pred
            trained_models[sp] = models
            prog.advance(task)

    for sp, pred in predictions.items():
        console.print(
            f"  [dim]{sp}:[/dim]"
            f"  median P(occur)=[cyan]{np.nanmedian(pred.p_occurrence):.3f}[/cyan]"
            f"  median E[abund]=[cyan]{np.nanmedian(pred.e_abundance):.3f}[/cyan]"
            f"  abd_ppy=[cyan]{np.nanmedian(pred.abd_ppy) * 100:+.2f}%[/cyan]"
        )

    sdm_rows = []
    diagnostics_rows = []
    for sp, pred in predictions.items():
        diag = pred.diagnostics
        sdm_rows.extend(
            [
                # marginal m̂(X): predicts Y from X, OOB preds become Y.hat
                (sp, "marginal · occurrence", "Brier", diag.occurrence_brier),
                (sp, "marginal · occurrence", "ROC AUC", diag.occurrence_auc),
                (sp, "marginal · occurrence", "PR AUC", diag.occurrence_pr_auc),
                (sp, "marginal · count | occ", "R²", diag.count_r2),
                (sp, "marginal · count | occ", "RMSE", diag.count_rmse),
                # propensity ŝ(X): predicts year from X, OOB preds become T.hat
                # High R² = strong temporal confounding; DML is doing real work.
                # R² > 0.85 may indicate a year-proxy feature leaking into X.
                (sp, "propensity · occurrence", "R²", diag.propensity_occ_r2),
                (sp, "propensity · count | occ", "R²", diag.propensity_count_r2),
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
        cols = {
            "bng_ref": pred.bng_ref,
            "p_occurrence": pred.p_occurrence,
            "e_count_given_occ": pred.e_count_given_occ,
            "e_abundance": pred.e_abundance,
            "occ_ppy": pred.occ_ppy,
            "count_ppy": pred.count_ppy,
            "abd_ppy": pred.abd_ppy,
        }
        if pred.occ_sigma.size == len(pred.bng_ref):
            cols["occ_sigma"] = pred.occ_sigma
        if pred.count_sigma.size == len(pred.bng_ref):
            cols["count_sigma"] = pred.count_sigma
        out_df = pd.DataFrame(cols)
        species_pred_path = out.species_prediction_file(sp)
        species_pred_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(species_pred_path)

        species_fig_dir = out.species_figures_dir(sp)
        species_fig_dir.mkdir(parents=True, exist_ok=True)

        plot_sdm_prediction_maps(pred, output_path=out.species_sdm_figure(sp))
        plot_trend_ppy_panels(
            pred,
            output_path=out.species_trend_ppy_panels_figure(sp),
        )
        plot_prediction_distribution_summary(
            pred,
            output_path=out.species_distribution_summary_figure(sp),
        )
        plot_yearly_trend(pred, output_path=out.species_yearly_trend_figure(sp))
        plot_raw_vs_fitted(pred, output_path=out.species_raw_vs_fitted_figure(sp))

    # ── SHAP interpretability ───────────────────────────────────────────
    # SDM drivers: exact TreeExplainer on the marginal forests.
    # τ(W) drivers: surrogate RandomForest on τ̂(W), then TreeExplainer —
    # econml.grf.CausalForest has no native SHAP hook (unlike CausalForestDML).
    import matplotlib.pyplot as plt

    section("SHAP interpretability")
    console.print()
    for sp, models in trained_models.items():
        sp_label = sp.replace("_", " ").title()
        console.print(f"  [dim]{sp}[/dim]  SDM marginal forests…")
        with timed_step(f"  SDM occ SHAP  (approx, n=500 rows)"):
            occ_expl = explain_tree_model(models.y_bin_forest, models.X_marg_occ)
        with timed_step(f"  SDM count SHAP  (approx, n=500 rows)"):
            count_expl = explain_tree_model(models.y_count_forest, models.X_marg_count)
        console.print(f"  [dim]{sp}[/dim]  CATE surrogate forests…")
        with timed_step(
            f"  CATE occ  surrogate fit → SHAP (n={len(models.tau_occ)} rows)"
        ):
            cate_occ_expl = explain_cate_surrogate(
                models.bin_cf,
                models.W_occ,
                models.W_feature_names,
                tau_hat=models.tau_occ,
            )
        with timed_step(
            f"  CATE count  surrogate fit → SHAP (n={len(models.tau_count)} rows)"
        ):
            cate_count_expl = explain_cate_surrogate(
                models.count_cf,
                models.W_count,
                models.W_feature_names,
                tau_hat=models.tau_count,
            )

        fig = plot_shap_panel(
            occ_expl,
            count_expl,
            title_left="Occurrence stage",
            title_right="Count stage (given detection)",
            suptitle=f"{sp_label} — SDM feature attribution",
            footer="SHAP values from marginal response forests  •  colour = feature value",
            output_path=out.species_shap_figure(sp, "sdm"),
        )
        plt.close(fig)

        fig = plot_shap_panel(
            cate_occ_expl,
            cate_count_expl,
            title_left="Occurrence CATE  τ_occ(W)",
            title_right="Count CATE  τ_count(W)",
            suptitle=f"{sp_label} — What drives spatial variation in trend?",
            footer="SHAP attribution via surrogate RandomForest on τ̂(W)",
            output_path=out.species_shap_figure(sp, "cate"),
        )
        plt.close(fig)
        console.print(f"  [dim]{sp}:[/dim] SDM → shap_sdm.png  •  CATE → shap_cate.png")

    # ── Residual confounding diagnostic ─────────────────────────────────
    section("Residual confounding diagnostic")
    console.print()
    n_rc = args.n_rc_sims
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
            nuisance_params=tuned_params.get(cfg.species[0]),
        )
    print_rc_summary(rc.bias_ate_occ, rc.bias_ate_count, rc.n_successful, n_rc)

    # ── Ensemble (Johnston §3.4) ────────────────────────────────────────
    section("Ensemble trends")
    if not run_ensemble:
        console.print("  [dim]skipped (run_ensemble=False)[/dim]")
    else:
        n_ensemble = args.n_ensemble
        console.print(
            f"  [dim]{n_ensemble} members  •  50/50 cell-year split  •  "
            f"independent occ/count draws[/dim]"
        )
        console.print()
        with make_spinner() as spin:
            spin.add_task(
                f"Ensemble fitting [{cfg.species[0]}]  ({n_ensemble} members)",
                total=None,
            )
            ens = fit_ensemble_trend(
                df=df,
                cfg=cfg,
                species=cfg.species[0],
                grid=grid,
                reference_effort=cfg.reference_effort,
                n_ensemble=n_ensemble,
                nuisance_params=tuned_params.get(cfg.species[0]),
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
        plot_prediction_distribution_summary(
            predictions[cfg.species[0]],
            ensemble=ens,
            output_path=out.species_distribution_summary_figure(cfg.species[0]),
        )
        console.print(
            f"  [dim]{ens.n_successful}/{ens.n_ensemble} members successful[/dim]"
        )

    console.print()
    console.print(f"[dim]All results written to[/dim] [cyan]{out.root}[/cyan]")
    console.print()


if __name__ == "__main__":
    main()
