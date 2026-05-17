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
    uv run python scripts/pipeline/run_prototype.py configs/local.yaml
    uv run python scripts/pipeline/run_prototype.py configs/cluster.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import yaml

# ── Bootstrap ────────────────────────────────────────────────────────────────
# config.py and data.py resolve path constants at import time, so data-path
# env vars must be set before any bgbw_trends import happens.

_PATH_ENV_KEYS = {
    "harmonized_dir": "BGBW_HARMONIZED_DIR",
    "aef_dir": "BGBW_AEF_DIR",
    "weather_path": "BGBW_WEATHER_PATH",
    "january_weather_path": "BGBW_JANUARY_WEATHER_PATH",
}


def _bootstrap() -> tuple[Path, dict]:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "config", type=Path, help="Path to a YAML config file (see configs/)."
    )
    args = p.parse_args()
    if not args.config.exists():
        p.error(f"Config file not found: {args.config}")
    with open(args.config, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    for key, env_var in _PATH_ENV_KEYS.items():
        if key in raw.get("paths", {}):
            os.environ[env_var] = str(raw["paths"][key])
    return args.config, raw


_config_path, _raw_cfg = _bootstrap()

logger = logging.getLogger(__name__)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    # bgbw_trends imports live here so they happen after _bootstrap() has set
    # the data-path env vars that config.py and data.py consume at import time.
    import numpy as np
    import pandas as pd

    from bgbw_trends.config import RunConfig, SPECIES_COLUMNS
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
    from bgbw_trends.plotting import (
        plot_prediction_distribution_summary,
        plot_raw_vs_fitted,
        plot_sdm_prediction_maps,
        plot_trend_ppy_panels,
        plot_yearly_trend,
    )
    import json
    from bgbw_trends.trends import (
        diagnose_residual_confounding,
        fit_ensemble_trend,
        split_cell_years,
    )
    import matplotlib.pyplot as plt

    def _build_run_config(raw: dict) -> tuple[RunConfig, bool, int, int]:
        run = raw.get("run", {})
        model = raw.get("model", {})
        compute = raw.get("compute", {})
        ens_cfg = raw.get("ensemble", {})
        species_raw = run.get("species", ["blackbird"])
        species = SPECIES_COLUMNS if species_raw == "all" else list(species_raw)
        cfg = RunConfig(
            species=species,
            years=run.get("years", list(range(2017, 2025, 2))),
            output_dir=Path(run.get("output_dir", "output/prototype")),
            n_pca_components=model.get("n_pca_components", 15),
            n_estimators=model.get("n_estimators", 300),
            cf_n_estimators=model.get("cf_n_estimators", 300),
            rf_min_samples_leaf=model.get("rf_min_samples_leaf", 10),
            cf_min_samples_leaf=model.get("cf_min_samples_leaf", 50),
            n_workers=compute.get("n_workers", 1),
        )
        return (
            cfg,
            ens_cfg.get("run", True),
            ens_cfg.get("n_ensemble", 10),
            ens_cfg.get("n_rc_sims", 5),
        )

    def _build_grid(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.sort_values("year")
            .drop_duplicates(subset=["bng_ref"], keep="last")
            .reset_index(drop=True)
        )

    raw = _raw_cfg

    # ── Logging ──────────────────────────────────────────────────────────
    setup_rich_logging(logging.INFO)
    if log_file := raw.get("log_file"):
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
        )
        logging.getLogger().addHandler(fh)
        sys.stderr.write(f"Logging to {log_path}\n")

    # ── Configuration ────────────────────────────────────────────────────
    cfg, run_ensemble, n_ensemble, n_rc_sims = _build_run_config(raw)
    out = cfg.output_paths()

    shutil.copy(_config_path, out.root / "config.yaml")

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
            df, _ = reduce_embeddings_pca(df, cfg.n_pca_components)

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

    # ── Reference effort & grid ──────────────────────────────────────────
    cfg.reference_effort = compute_reference_effort(df)
    grid = _build_grid(df)
    console.print(
        f"  [dim]grid:[/dim] [cyan]{len(grid):,}[/cyan] cells  "
        f"[dim]reference effort:[/dim] n_surveys=[cyan]"
        f"{cfg.reference_effort.n_surveys:.2f}[/cyan]"
    )

    # ── Two-part DML trend + SDM (Fink §3.2) ─────────────────────────────
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
                n_jobs=cfg.effective_n_jobs(1),
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
                (sp, "marginal · occurrence", "Brier", diag.occurrence_brier),
                (sp, "marginal · occurrence", "ROC AUC", diag.occurrence_auc),
                (sp, "marginal · occurrence", "PR AUC", diag.occurrence_pr_auc),
                (sp, "marginal · count | occ", "R²", diag.count_r2),
                (sp, "marginal · count | occ", "RMSE", diag.count_rmse),
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
        plot_trend_ppy_panels(pred, output_path=out.species_trend_ppy_panels_figure(sp))
        plot_prediction_distribution_summary(
            pred, output_path=out.species_distribution_summary_figure(sp)
        )
        plot_yearly_trend(pred, output_path=out.species_yearly_trend_figure(sp))
        plot_raw_vs_fitted(pred, output_path=out.species_raw_vs_fitted_figure(sp))

    # ── SHAP interpretability ────────────────────────────────────────────
    section("SHAP interpretability")
    console.print()
    for sp, models in trained_models.items():
        sp_label = sp.replace("_", " ").title()
        console.print(f"  [dim]{sp}[/dim]  SDM marginal forests…")
        with timed_step("  SDM occ SHAP  (approx, n=500 rows)"):
            occ_expl = explain_tree_model(models.y_bin_forest, models.X_marg_occ)
        with timed_step("  SDM count SHAP  (approx, n=500 rows)"):
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

        # ── SHAP data (long format for downstream comparative analysis) ──
        shap_sdm_path = out.species_shap_data(sp, "sdm")
        shap_sdm_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(
            [occ_expl.to_frame("sdm_occ"), count_expl.to_frame("sdm_count")],
            ignore_index=True,
        ).to_parquet(shap_sdm_path)

        shap_cate_path = out.species_shap_data(sp, "cate")
        pd.concat(
            [
                cate_occ_expl.to_frame("cate_occ"),
                cate_count_expl.to_frame("cate_count"),
            ],
            ignore_index=True,
        ).to_parquet(shap_cate_path)

        # GRF-native heterogeneity importances (unsigned, surrogate-free)
        grf_frames = []
        for stage, expl in [
            ("cate_occ", cate_occ_expl),
            ("cate_count", cate_count_expl),
        ]:
            if expl.grf_importances is not None:
                grf_df = expl.grf_importances.copy()
                grf_df.insert(0, "stage", stage)
                grf_frames.append(grf_df)
        if grf_frames:
            grf_path = out.species_shap_data(sp, "cate_grf_importances")
            grf_path.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(grf_frames, ignore_index=True).to_parquet(grf_path)

        console.print(
            f"  [dim]{sp}:[/dim] SDM → shap_sdm.png + shap_sdm.parquet"
            f"  •  CATE → shap_cate.png + shap_cate.parquet"
        )

    # ── Residual confounding calibration (Fink §3.4) ─────────────────────
    section("Residual confounding diagnostic (Fink §2.2)")
    console.print()
    rc_results: dict[str, object] = {}
    for sp in cfg.species:
        with make_spinner() as spin:
            spin.add_task(
                f"RC null sims [{sp}]  ({n_rc_sims} simulation(s))",
                total=None,
            )
            rc = diagnose_residual_confounding(
                df,
                species=sp,
                cfg=cfg,
                grid=grid,
                reference_effort=cfg.reference_effort,
                n_simulations=n_rc_sims,
            )
        rc_results[sp] = rc
        print_rc_summary(rc)

        # Save null-bias summary (JSON) and per-sim per-cell data (parquet)
        rc_dir = out.species_rc_dir(sp)
        rc_dir.mkdir(parents=True, exist_ok=True)

        adj = {
            "species": sp,
            "bias_ate_occ": rc.bias_ate_occ,
            "bias_ate_count": rc.bias_ate_count,
            "bias_ate_abd": rc.bias_ate_abd,
            "n_successful": rc.n_successful,
            "n_simulations": n_rc_sims,
        }
        with open(out.species_rc_adjustment_file(sp), "w") as f:
            json.dump(adj, f, indent=2)

        if not rc.sim_df.empty:
            rc.sim_df.to_parquet(out.species_rc_simulations_file(sp), index=False)

        console.print(
            f"  [dim]{sp}: bias_abd={rc.bias_ate_abd:+.4f}"
            f"  n={rc.n_successful}/{n_rc_sims}[/dim]"
        )

    # ── Ensemble (Fink §3.2 + §3.4 correction) ───────────────────────────
    section("Ensemble trends")
    if not run_ensemble:
        console.print("  [dim]skipped (ensemble.run: false)[/dim]")
    else:
        console.print(
            f"  [dim]{n_ensemble} members  •  50/50 cell-year split  •  "
            f"independent occ/count draws  •  RC correction in adjustment.json[/dim]"
        )
        console.print()
        for sp in cfg.species:
            with make_spinner() as spin:
                spin.add_task(
                    f"Ensemble fitting [{sp}]  ({n_ensemble} members)", total=None
                )
                ens = fit_ensemble_trend(
                    df=df,
                    cfg=cfg,
                    species=sp,
                    grid=grid,
                    reference_effort=cfg.reference_effort,
                    n_ensemble=n_ensemble,
                )
            ensemble_path = out.species_ensemble_file(sp)
            ensemble_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "bng_ref": ens.bng_ref,
                    "p_occurrence": ens.p_occurrence,
                    "p_occ_lower": ens.p_occ_lower,
                    "p_occ_upper": ens.p_occ_upper,
                    "e_abundance": ens.e_abundance,
                    "e_abund_lower": ens.e_abund_lower,
                    "e_abund_upper": ens.e_abund_upper,
                    "occ_ppy": ens.occ_ppy,
                    "occ_ppy_lower": ens.occ_ppy_lower,
                    "occ_ppy_upper": ens.occ_ppy_upper,
                    "count_ppy": ens.count_ppy,
                    "count_ppy_lower": ens.count_ppy_lower,
                    "count_ppy_upper": ens.count_ppy_upper,
                    "abd_ppy": ens.abd_ppy,
                    "abd_ppy_lower": ens.abd_ppy_lower,
                    "abd_ppy_upper": ens.abd_ppy_upper,
                }
            ).to_parquet(ensemble_path)
            plot_prediction_distribution_summary(
                predictions[sp],
                ensemble=ens,
                output_path=out.species_distribution_summary_figure(sp),
            )
            console.print(
                f"  [dim]{sp}: {ens.n_successful}/{ens.n_ensemble} members successful[/dim]"
            )

    console.print()
    console.print(f"[dim]All results written to[/dim] [cyan]{out.root}[/cyan]")
    console.print()


if __name__ == "__main__":
    main()
