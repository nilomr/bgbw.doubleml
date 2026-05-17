"""Residual-confounding calibration (Fink 2023 §2.2 / §3.4).

Two-stage workflow matching the trendswf reference implementation
(trendswf/workflow/summarize.R, trendswf/R/summarize.R, trendswf/R/utils.R):

Stage 1 — Simulation
    For each of several known trend magnitudes (including zero), the species
    response is regenerated under a controlled DGP, the full pipeline is
    refit, and the per-cell ppy estimates are recorded.

    Trend injection follows Fink: the null DGP (§2.2) generates zero-trend
    responses via stratified resampling + conditional-mean bootstrap; a
    known trend is then injected by multiplying detected counts by
    ``exp(trend_ppy × (year − ȳear))``, so simulated_abd_ppy = trend_ppy.

Stage 2 — Calibration regression (trendswf utils.R ``lm_parameters``)
    Pooling all (cell, scenario, simulation) triples, fit

        simulated_abd_ppy = β₀ + β₁ × estimated_abd_ppy

    β₀ = directional bias at zero trend (intercept).
    β₁ = scaling factor (slope).

    Correction applied per cell:

        corrected_abd_ppy = β₀ + β₁ × raw_abd_ppy

Outputs
-------
``ResidualConfoundingResult`` carries the coefficients plus a
``sim_df`` DataFrame (per-sim per-cell) for downstream diagnostics and
plotting.  The caller (run_prototype.py) is responsible for saving both
to disk (adjustment.json and simulations.parquet).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy import stats

from bgbw_trends.config import ReferenceEffort, RunConfig
from bgbw_trends.pipeline import (
    _assemble_X,
    _fit_nuisance,
    _predict,
    train_predict_species,
)
from bgbw_trends.trends.ensemble import _resolve_inner_n_jobs, split_cell_years

logger = logging.getLogger(__name__)

# ── Default scenario grid ─────────────────────────────────────────────────────
# Trend magnitudes in ratios/yr (multiply × 100 for %/yr).
# Covers zero (null), moderate (≈IUCN 3.3%/yr) and strong (≈IUCN 6.7%/yr)
# in both directions — matching Fink's five-scenario training grid (§3.5).
DEFAULT_RC_SCENARIOS: list[float] = [-0.067, -0.033, 0.0, 0.033, 0.067]

_NULL_DGP_MAX_TREES = 100


# ── Result container ──────────────────────────────────────────────────────────


@dataclass
class ResidualConfoundingResult:
    """Calibration regression coefficients and per-simulation diagnostics.

    Coefficients
    ------------
    beta0, beta1
        From ``simulated_abd_ppy = β₀ + β₁ × estimated_abd_ppy`` fitted
        across all (cell, scenario, simulation) combinations.
        Apply correction as: ``corrected = beta0 + beta1 * raw_estimated``.
    beta0_se, beta1_se
        Standard errors of the regression coefficients.
    rsquared
        R² of the calibration regression (how well the DML recovers the
        injected trend signal).
    rank_cor
        Spearman rank correlation between simulated and estimated abd_ppy.

    Diagnostics
    -----------
    sim_df
        Long-format DataFrame with one row per (simulation, cell):
        ``scenario_trend``, ``sim_idx``, ``bng_ref``,
        ``occ_ppy``, ``count_ppy``, ``abd_ppy`` (all estimated values).
    bias_ate_occ, bias_ate_count
        Mean ATE from null-only simulations (legacy diagnostic; in
        probability/yr and log-count/yr respectively, not in %/yr).
    n_successful
        Total simulations that completed without error.
    """

    species: str
    # Calibration regression (simulated ~ estimated)
    beta0: float
    beta1: float
    beta0_se: float
    beta1_se: float
    rsquared: float
    rank_cor: float
    # Per-simulation per-cell data for plotting
    sim_df: pd.DataFrame
    # ATE from null simulations (for backward-compatible console summary)
    bias_ate_occ: float
    bias_ate_count: float
    n_successful: int
    n_scenarios: int = 0
    n_sims_per_scenario: int = 0
    sim_ate_occ: np.ndarray = field(default_factory=lambda: np.array([]))
    sim_ate_count: np.ndarray = field(default_factory=lambda: np.array([]))


# ── Null DGP ──────────────────────────────────────────────────────────────────


def _null_species_values(
    df: pd.DataFrame,
    species: str,
    cfg: RunConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Regenerate the species column under a zero-trend DGP.

    Follows trendswf vignette §2.2 / motivating-example.Rmd L312-320:
    fit a forest on the resampled features (year excluded by default in
    _assemble_X), use OOB predictions as the conditional mean, add
    parametric-bootstrap noise.  Adapted to the hurdle: Bernoulli draw
    for occurrence, log-normal noise for log-counts on detected rows.
    """
    X = _assemble_X(df, None)
    null_cfg = replace(cfg, n_estimators=min(cfg.n_estimators, _NULL_DGP_MAX_TREES))

    y_bin = (df[species] > 0).to_numpy(dtype=np.float64)
    _occ_forest, p_hat = _fit_nuisance(
        X, y_bin, null_cfg, seed=int(rng.integers(0, 2**31))
    )
    p_hat = np.clip(p_hat, 0.0, 1.0)
    y_bin_null = rng.binomial(1, p_hat).astype(np.float64)

    count_null = np.zeros(len(df), dtype=np.float64)
    orig_detected = y_bin > 0
    if orig_detected.sum() < 50:
        return count_null

    Y_log = np.log1p(df.loc[orig_detected, species].to_numpy(dtype=np.float64))
    X_det = X.iloc[orig_detected.nonzero()[0]].reset_index(drop=True)
    _count_forest, m_count_oof = _fit_nuisance(
        X_det, Y_log, null_cfg, seed=int(rng.integers(0, 2**31)),
    )
    resid_sd = float(np.std(Y_log - m_count_oof, ddof=0))

    m_count_all = _predict(_count_forest, X)
    noise = rng.normal(0.0, resid_sd, size=len(df))
    log_count_null = m_count_all + noise
    count_null = np.where(
        y_bin_null > 0, np.maximum(np.expm1(log_count_null), 0.0), 0.0
    )
    return count_null


def _inject_count_trend(
    null_df: pd.DataFrame,
    species: str,
    trend_ppy: float,
) -> pd.DataFrame:
    """Multiply detected counts by exp(trend_ppy × (year − ȳear)).

    Injects a known count-stage trend while leaving occurrence unchanged.
    Simulated abd_ppy = 0 (occ) + trend_ppy (count) = trend_ppy.
    """
    if trend_ppy == 0.0:
        return null_df
    years = null_df["year"].to_numpy(dtype=float)
    year_mean = float(years.mean())
    factor = np.exp(trend_ppy * (years - year_mean))
    df_out = null_df.copy()
    counts = df_out[species].to_numpy(dtype=float)
    df_out[species] = np.where(counts > 0, counts * factor, 0.0)
    return df_out


# ── Per-simulation worker ─────────────────────────────────────────────────────


def _one_rc_simulation(
    df: pd.DataFrame,
    species: str,
    cfg: RunConfig,
    grid: pd.DataFrame,
    reference_effort: ReferenceEffort,
    sim_idx: int,
    sim_seed: int,
    trend_ppy: float,
) -> dict | None:
    """One residual-confounding simulation at a given trend magnitude.

    Returns a dict with keys:
      bng_ref, occ_ppy, count_ppy, abd_ppy  (per-cell arrays)
      ate_occ, ate_count                     (scalar ATEs for diagnostics)
    or None on failure.
    """
    rng = np.random.default_rng(sim_seed)
    years = df["year"].to_numpy()
    unique_years = np.unique(years)

    # Step 1 — stratified bootstrap (preserves interannual confounding)
    pieces = [
        rng.choice(np.where(years == y)[0], size=int((years == y).sum()), replace=True)
        for y in unique_years
    ]
    boot_idx = np.concatenate(pieces)
    null_df = df.iloc[boot_idx].reset_index(drop=True)

    # Step 2 — zero-trend DGP
    null_df[species] = _null_species_values(null_df, species, cfg, rng)

    # Step 3 — inject known trend into count
    null_df = _inject_count_trend(null_df, species, trend_ppy)

    # Step 4 — 50/50 split matching ensemble, refit pipeline
    occ_df, count_df = split_cell_years(null_df, rng)

    try:
        pred = train_predict_species(
            occ_df,
            count_df,
            species=species,
            cfg=cfg,
            grid=grid,
            reference_effort=reference_effort,
            seed=int(rng.integers(0, 2**31)),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "RC simulation %d (trend=%.3f) failed: %s", sim_idx, trend_ppy, e
        )
        return None

    return {
        "bng_ref": pred.bng_ref,
        "occ_ppy": pred.occ_ppy,
        "count_ppy": pred.count_ppy,
        "abd_ppy": pred.abd_ppy,
        "ate_occ": float(pred.ate_occ),
        "ate_count": float(pred.ate_count),
    }


# ── Calibration regression ────────────────────────────────────────────────────


def _fit_calibration_regression(
    simulated: np.ndarray,
    estimated: np.ndarray,
) -> dict[str, float]:
    """Fit simulated_abd_ppy = β₀ + β₁ × estimated_abd_ppy.

    Mirrors trendswf/R/utils.R ``lm_parameters(x=estimated, y=simulated)``.
    Both inputs are ratios/yr (not percent).
    """
    mask = np.isfinite(simulated) & np.isfinite(estimated)
    s, e = simulated[mask], estimated[mask]
    if len(s) < 10:
        logger.warning("Too few finite RC data points (%d) to fit calibration", len(s))
        return {
            "beta0": 0.0, "beta0_se": np.nan,
            "beta1": 1.0, "beta1_se": np.nan,
            "rsquared": np.nan, "rank_cor": np.nan,
        }
    result = stats.linregress(e, s)
    n = len(e)
    x_mean = np.mean(e)
    ss_x = np.sum((e - x_mean) ** 2)
    mse = np.sum((s - (result.intercept + result.slope * e)) ** 2) / (n - 2)
    se_slope = np.sqrt(mse / ss_x) if ss_x > 0 else np.nan
    se_intercept = np.sqrt(mse * (1.0 / n + x_mean**2 / ss_x)) if ss_x > 0 else np.nan
    rank_cor = float(stats.spearmanr(e, s).statistic)
    return {
        "beta0": float(result.intercept),
        "beta0_se": float(se_intercept),
        "beta1": float(result.slope),
        "beta1_se": float(se_slope),
        "rsquared": float(result.rvalue**2),
        "rank_cor": rank_cor,
    }


# ── Public API ────────────────────────────────────────────────────────────────


def diagnose_residual_confounding(
    df: pd.DataFrame,
    *,
    species: str,
    cfg: RunConfig,
    grid: pd.DataFrame,
    reference_effort: ReferenceEffort,
    n_simulations: int = 10,
    trend_scenarios: list[float] | None = None,
    seed: int = 0,
) -> ResidualConfoundingResult:
    """Run multi-scenario RC simulations and fit the calibration regression.

    Parameters
    ----------
    df
        Training frame — same as passed to ``train_predict_species``.
    species
        Species column name.
    cfg, grid, reference_effort
        Passed through to ``train_predict_species``.
    n_simulations
        *Total* simulations across all scenarios.  Distributed evenly:
        ``sims_per_scenario = max(1, n_simulations // len(trend_scenarios))``.
        With the default 5 scenarios, ``n_simulations=5`` gives 1 sim/scenario;
        ``n_simulations=10`` gives 2, etc.
    trend_scenarios
        Trend magnitudes (ratios/yr) to simulate.  ``None`` → DEFAULT_RC_SCENARIOS.
        Must include 0.0 for the null-only diagnostics (bias_ate_*).
    seed
        Base RNG seed.
    """
    if trend_scenarios is None:
        trend_scenarios = DEFAULT_RC_SCENARIOS

    n_scen = len(trend_scenarios)
    sims_per_scenario = max(1, n_simulations // n_scen)

    logger.info(
        "Residual confounding [%s]: %d scenario(s) × %d sim(s) = %d total",
        species, n_scen, sims_per_scenario, n_scen * sims_per_scenario,
    )

    # Pre-generate seeds
    seed_seq = np.random.SeedSequence(seed)
    total_sims = n_scen * sims_per_scenario
    child_seeds = seed_seq.spawn(total_sims)

    worker_cfg = replace(cfg, inner_n_jobs=_resolve_inner_n_jobs(cfg))

    # Build job list: (scenario_trend, sim_idx_global, sim_seed)
    jobs = []
    global_idx = 0
    for trend in trend_scenarios:
        for _s in range(sims_per_scenario):
            jobs.append((
                df, species, worker_cfg, grid, reference_effort,
                global_idx,
                int(child_seeds[global_idx].generate_state(1)[0]),
                trend,
            ))
            global_idx += 1

    n_workers = max(1, cfg.n_workers)
    if n_workers == 1:
        results_raw = [_one_rc_simulation(*args) for args in jobs]
    else:
        from joblib import Parallel, delayed
        logger.info(
            "RC sims: %d parallel workers  •  inner_n_jobs=%d",
            n_workers, worker_cfg.inner_n_jobs,
        )
        results_raw = Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
            delayed(_one_rc_simulation)(*args) for args in jobs
        )

    # Collect results alongside their scenario trends
    sim_records: list[dict] = []
    null_ate_occ: list[float] = []
    null_ate_count: list[float] = []
    n_ok = 0

    for (args, result) in zip(jobs, results_raw):
        trend = args[-1]  # last element of args tuple
        if result is None:
            continue
        n_ok += 1
        sim_idx = args[5]
        bng_ref = result["bng_ref"]
        n_cells = len(bng_ref)
        sim_records.append(pd.DataFrame({
            "scenario_trend": np.full(n_cells, trend, dtype=np.float64),
            "sim_idx": np.full(n_cells, sim_idx, dtype=np.int32),
            "bng_ref": bng_ref,
            "occ_ppy": result["occ_ppy"],
            "count_ppy": result["count_ppy"],
            "abd_ppy": result["abd_ppy"],
        }))
        if trend == 0.0:
            null_ate_occ.append(result["ate_occ"])
            null_ate_count.append(result["ate_count"])

    if n_ok == 0:
        logger.warning("All RC simulations failed")
        return ResidualConfoundingResult(
            species=species,
            beta0=0.0, beta1=1.0,
            beta0_se=np.nan, beta1_se=np.nan,
            rsquared=np.nan, rank_cor=np.nan,
            sim_df=pd.DataFrame(),
            bias_ate_occ=float("nan"),
            bias_ate_count=float("nan"),
            n_successful=0,
            n_scenarios=n_scen,
            n_sims_per_scenario=sims_per_scenario,
        )

    sim_df = pd.concat(sim_records, ignore_index=True)

    # Calibration regression: simulated (= scenario_trend) ~ estimated (= abd_ppy)
    reg = _fit_calibration_regression(
        simulated=sim_df["scenario_trend"].to_numpy(),
        estimated=sim_df["abd_ppy"].to_numpy(),
    )

    bias_ate_occ = float(np.mean(null_ate_occ)) if null_ate_occ else float("nan")
    bias_ate_count = float(np.mean(null_ate_count)) if null_ate_count else float("nan")
    sim_ate_occ = np.asarray(null_ate_occ, dtype=np.float64)
    sim_ate_count = np.asarray(null_ate_count, dtype=np.float64)

    logger.info(
        "RC calibration [%s]: β₀=%+.4f (SE=%.4f) β₁=%.4f (SE=%.4f) "
        "R²=%.3f  ρ=%.3f  n_sims=%d",
        species,
        reg["beta0"], reg["beta0_se"],
        reg["beta1"], reg["beta1_se"],
        reg["rsquared"], reg["rank_cor"],
        n_ok,
    )
    if null_ate_occ:
        logger.info(
            "RC null-bias [%s]: ate_occ=%+.4f (sd=%.4f)  ate_count=%+.4f (sd=%.4f)",
            species,
            bias_ate_occ,
            float(np.std(sim_ate_occ)) if len(sim_ate_occ) > 1 else 0.0,
            bias_ate_count,
            float(np.std(sim_ate_count)) if len(sim_ate_count) > 1 else 0.0,
        )

    return ResidualConfoundingResult(
        species=species,
        beta0=reg["beta0"],
        beta1=reg["beta1"],
        beta0_se=reg["beta0_se"],
        beta1_se=reg["beta1_se"],
        rsquared=reg["rsquared"],
        rank_cor=reg["rank_cor"],
        sim_df=sim_df,
        bias_ate_occ=bias_ate_occ,
        bias_ate_count=bias_ate_count,
        n_successful=n_ok,
        n_scenarios=n_scen,
        n_sims_per_scenario=sims_per_scenario,
        sim_ate_occ=sim_ate_occ,
        sim_ate_count=sim_ate_count,
    )
