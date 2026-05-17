"""Residual-confounding diagnostic (Fink 2023 §2.2, vignette §2.2).

Direct port of the ``generate_null_trend_model_data`` / ``train_cf``
pair in ``docs/refrence/trendswf/vignettes/motivating-example.Rmd``,
adapted for the BGBW two-part pipeline:

1. Resample training rows **stratified by year**, with replacement.
   This preserves interannual drift in the covariates — the confounding
   structure we want to test against — while breaking any direct trend
   in the outcome.
2. Fit a conditional-mean model ``m̂(X)`` (a zero-trend model by
   construction; X excludes year).
3. Regenerate the species response from ``m̂(X*)`` + noise. For the
   hurdle model we regenerate *both* components: occurrence via a
   Bernoulli draw from the fitted occurrence probabilities, and
   per-detection log-counts via a parametric bootstrap with residual
   SD (vignette L319-320).
4. Split the null data 50/50 stratified by year (matching the ensemble
   exactly) and refit ``train_predict_species``. Any non-zero
   ``ate_occ`` / ``ate_count`` is residual confounding the propensity
   model did not absorb.

This matches exactly the vignette's §2.2 workflow: same diagnostic,
same null generator, but applied to both stages of the hurdle and
driven through the *actual* production pipeline (so what we test is
what we ship).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from bgbw_trends.config import ReferenceEffort, RunConfig
from bgbw_trends.pipeline import (
    _assemble_X,
    _fit_nuisance,
    _predict,
    train_predict_species,
)
from bgbw_trends.trends.ensemble import _resolve_inner_n_jobs, split_cell_years

logger = logging.getLogger(__name__)


@dataclass
class ResidualConfoundingResult:
    """Null-simulation biases for the two-part DML pipeline.

    Both biases are the average ``ate_*`` obtained by running
    ``train_predict_species`` on data where the true trend is known
    to be zero. Values close to zero indicate the propensity model is
    absorbing the interannual confounding; values materially distinct
    from zero signal residual confounding and can be subtracted from
    the corresponding ``ate_*`` on real data as a first-order
    bias correction (Fink §2.2 Eq. 6, vignette §2.2).
    """

    species: str
    bias_ate_occ: float
    bias_ate_count: float
    sim_ate_occ: np.ndarray
    sim_ate_count: np.ndarray
    n_successful: int


_NULL_DGP_MAX_TREES = 100


def _null_species_values(
    df: pd.DataFrame,
    species: str,
    cfg: RunConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Regenerate the species column under a zero-trend DGP.

    Follows vignette L312-320: fit a forest on the resampled features,
    use its predictions as the conditional mean, add parametric-bootstrap
    noise. Adapted to the hurdle: Bernoulli draw for occurrence, then
    log-normal noise for log-counts on detected rows.
    """
    # X excludes year (year is in df but not in marginal_covariates default).
    X = _assemble_X(df, None)

    # The null DGP forests only need a rough conditional mean for the
    # Bernoulli draw / parametric bootstrap — they are not the diagnostic
    # model and their quality does not affect the bias estimate. Cap at
    # _NULL_DGP_MAX_TREES regardless of cfg.n_estimators; this makes RC
    # sims materially faster at high n_estimators (cluster runs).
    null_cfg = replace(cfg, n_estimators=min(cfg.n_estimators, _NULL_DGP_MAX_TREES))

    # Occurrence null — Bernoulli(p̂(X))
    y_bin = (df[species] > 0).to_numpy(dtype=np.float64)
    _occ_forest, p_hat = _fit_nuisance(
        X, y_bin, null_cfg, seed=int(rng.integers(0, 2**31))
    )
    p_hat = np.clip(p_hat, 0.0, 1.0)
    y_bin_null = rng.binomial(1, p_hat).astype(np.float64)

    # Count null — log1p mean via OOF predictions on detected rows, plus
    # parametric bootstrap with empirical residual SD (vignette L319-320).
    count_null = np.zeros(len(df), dtype=np.float64)
    orig_detected = y_bin > 0
    if orig_detected.sum() < 50:
        # Not enough detections to fit a count-null generator; leave zeros.
        # (train_predict_species will then fail on this stage — caller
        # should treat this as an inconclusive diagnostic.)
        return count_null

    Y_log = np.log1p(df.loc[orig_detected, species].to_numpy(dtype=np.float64))
    X_det = X.iloc[orig_detected.nonzero()[0]].reset_index(drop=True)
    _count_forest, m_count_oof = _fit_nuisance(
        X_det, Y_log, null_cfg, seed=int(rng.integers(0, 2**31)),
    )
    resid_sd = float(np.std(Y_log - m_count_oof, ddof=0))

    # Predict the mean at every row — we only need values where y_bin_null=1
    m_count_all = _predict(_count_forest, X)
    noise = rng.normal(0.0, resid_sd, size=len(df))
    log_count_null = m_count_all + noise
    count_null = np.where(
        y_bin_null > 0, np.maximum(np.expm1(log_count_null), 0.0), 0.0
    )
    return count_null


def _one_null_simulation(
    df: pd.DataFrame,
    species: str,
    cfg: RunConfig,
    grid: pd.DataFrame,
    reference_effort: ReferenceEffort,
    sim_idx: int,
    sim_seed: int,
) -> tuple[float, float] | None:
    """Run one null-trend simulation. Top-level so workers can pickle it."""
    rng = np.random.default_rng(sim_seed)
    years = df["year"].to_numpy()
    unique_years = np.unique(years)

    # Step 1 — resample stratified by year, with replacement. Preserves the
    # joint distribution of features within each year, including the
    # interannual drift that drives the confounding.
    pieces = [
        rng.choice(np.where(years == y)[0], size=int((years == y).sum()), replace=True)
        for y in unique_years
    ]
    boot_idx = np.concatenate(pieces)
    null_df = df.iloc[boot_idx].reset_index(drop=True)

    # Steps 2–3 — regenerate the species column under a zero-trend DGP.
    null_df[species] = _null_species_values(null_df, species, cfg, rng)

    # Step 4 — split 50/50 stratified by year, matching the ensemble exactly.
    # This keeps RC methodologically consistent with what the ensemble does
    # and halves the RF training cost per sim.
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
    except Exception as e:  # noqa: BLE001 — isolate failing sims
        logger.warning("Null simulation %d failed: %s", sim_idx, e)
        return None
    return float(pred.ate_occ), float(pred.ate_count)


def diagnose_residual_confounding(
    df: pd.DataFrame,
    *,
    species: str,
    cfg: RunConfig,
    grid: pd.DataFrame,
    reference_effort: ReferenceEffort,
    n_simulations: int = 10,
    seed: int = 0,
) -> ResidualConfoundingResult:
    """Run the null-trend simulation (Fink vignette §2.2).

    Simulations run in parallel when ``cfg.n_workers > 1`` (one process per
    worker, inner-forest threading throttled to keep CPU use balanced).

    Parameters
    ----------
    df : pd.DataFrame
        The training frame — same shape as what you pass to
        ``train_predict_species`` (cell-year rows with species counts,
        effort, habitat features).
    species : str
        Species column to diagnose.
    cfg, grid, reference_effort
        Passed through to ``train_predict_species``.
    n_simulations : int
        Number of null-trend simulations. Each fits the full two-part
        pipeline, so cost scales linearly (and divides by ``n_workers``
        when run in parallel).
    seed : int
        RNG seed.
    """
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(n_simulations)

    worker_cfg = replace(cfg, inner_n_jobs=_resolve_inner_n_jobs(cfg))

    jobs = [
        (
            df,
            species,
            worker_cfg,
            grid,
            reference_effort,
            i,
            int(child_seeds[i].generate_state(1)[0]),
        )
        for i in range(n_simulations)
    ]

    n_workers = max(1, cfg.n_workers)
    if n_workers == 1:
        results = [_one_null_simulation(*args) for args in jobs]
    else:
        from joblib import Parallel, delayed

        logger.info(
            "Residual-confounding sims: %d  •  %d parallel workers  •  "
            "inner_n_jobs=%d",
            n_simulations,
            n_workers,
            worker_cfg.inner_n_jobs,
        )
        results = Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
            delayed(_one_null_simulation)(*args) for args in jobs
        )

    sim_occ = [r[0] for r in results if r is not None]
    sim_count = [r[1] for r in results if r is not None]

    sim_occ_arr = np.asarray(sim_occ, dtype=np.float64)
    sim_count_arr = np.asarray(sim_count, dtype=np.float64)
    n_ok = len(sim_occ_arr)

    if n_ok == 0:
        logger.warning("All residual-confounding simulations failed")
        return ResidualConfoundingResult(
            species=species,
            bias_ate_occ=float("nan"),
            bias_ate_count=float("nan"),
            sim_ate_occ=sim_occ_arr,
            sim_ate_count=sim_count_arr,
            n_successful=0,
        )

    bias_occ = float(np.mean(sim_occ_arr))
    bias_count = float(np.mean(sim_count_arr))

    logger.info(
        "Residual confounding [%s]: bias_ate_occ=%+.4f (sd=%.4f) "
        "bias_ate_count=%+.4f (sd=%.4f) n=%d",
        species,
        bias_occ,
        float(np.std(sim_occ_arr)),
        bias_count,
        float(np.std(sim_count_arr)),
        n_ok,
    )

    return ResidualConfoundingResult(
        species=species,
        bias_ate_occ=bias_occ,
        bias_ate_count=bias_count,
        sim_ate_occ=sim_occ_arr,
        sim_ate_count=sim_count_arr,
        n_successful=n_ok,
    )
