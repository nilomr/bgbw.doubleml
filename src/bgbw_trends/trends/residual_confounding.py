"""Residual-confounding null diagnostic (Fink 2023 §2.2).

Fully data-driven simulation diagnostic: generate zero-trend data via
stratified bootstrap + conditional-mean forest, refit the full pipeline,
and measure the bias in the estimated trend.

Algorithm (§2.2):
  1. Resample surveys with replacement, stratified by year, to maintain
     the interannual confounding in the original features.
  2. Regenerate the species response from the conditional mean model
     (zero-trend DGP) with parametric-bootstrap noise.
  3. Refit the full DML pipeline on the null data.
  4. Record the estimated occ_ppy, count_ppy, abd_ppy.

Interpretation:
  Bias far from zero indicates the propensity score model has not fully
  absorbed the interannual confounding.  The null ATE provides a simple
  directional-bias correction (Fink vignette):

      corrected_range_wide_trend ≈ raw_trend − bias_ate_abd

This is intentionally the §2.2 null-only diagnostic, NOT the extended
multi-scenario calibration regression from §3.4/§3.5.  See docs/refrence/
fink_2023.txt §3.4 for why that extension requires a population-dynamics
DGP with spatially varying trends (50+ pipeline fits per species).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

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

_NULL_DGP_MAX_TREES = 100


# ── Result container ──────────────────────────────────────────────────────────


@dataclass
class ResidualConfoundingResult:
    """Null-simulation diagnostics for one species (Fink §2.2).

    bias_ate_occ, bias_ate_count
        Mean range-wide ATE across null simulations for the occurrence and
        count stages.  Units: probability/yr and log-count/yr respectively.
        Values far from zero indicate residual confounding that the
        propensity score model did not absorb.
    bias_ate_abd
        Mean range-wide median abd_ppy across null simulations (%/yr as a
        ratio).  Interpretable headline: how much trend the model estimates
        when the true trend is zero.
        Directional correction: corrected = raw_abd_ppy − bias_ate_abd.
    sim_df
        Long-format DataFrame (one row per simulation × cell):
        ``sim_idx``, ``bng_ref``, ``occ_ppy``, ``count_ppy``, ``abd_ppy``.
        Saved to simulations.parquet for spatial bias diagnostics.
    n_successful
        Number of null simulations that completed without error.
    """

    species: str
    bias_ate_occ: float
    bias_ate_count: float
    bias_ate_abd: float
    n_successful: int
    sim_df: pd.DataFrame
    sim_ate_occ: np.ndarray = field(default_factory=lambda: np.array([]))
    sim_ate_count: np.ndarray = field(default_factory=lambda: np.array([]))
    sim_ate_abd: np.ndarray = field(default_factory=lambda: np.array([]))


# ── Null DGP ──────────────────────────────────────────────────────────────────


def _null_species_values(
    df: pd.DataFrame,
    species: str,
    cfg: RunConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Regenerate the species column under a zero-trend DGP.

    Follows Fink §2.2 / trendswf vignette motivating-example.Rmd:
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


# ── Per-simulation worker ─────────────────────────────────────────────────────


def _one_null_simulation(
    df: pd.DataFrame,
    species: str,
    cfg: RunConfig,
    grid: pd.DataFrame,
    reference_effort: ReferenceEffort,
    sim_idx: int,
    sim_seed: int,
) -> dict | None:
    """One null-trend simulation (§2.2).

    Returns a dict with keys:
      bng_ref, occ_ppy, count_ppy, abd_ppy  (per-cell arrays)
      ate_occ, ate_count                     (scalar ATEs for diagnostics)
      abd_median                             (range-wide median abd_ppy)
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

    # Step 3 — 50/50 split matching ensemble, refit pipeline
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
        logger.warning("Null sim %d failed: %s", sim_idx, e)
        return None

    return {
        "bng_ref": pred.bng_ref,
        "occ_ppy": pred.occ_ppy,
        "count_ppy": pred.count_ppy,
        "abd_ppy": pred.abd_ppy,
        "ate_occ": float(pred.ate_occ),
        "ate_count": float(pred.ate_count),
        "abd_median": float(np.nanmedian(pred.abd_ppy)),
    }


# ── Public API ────────────────────────────────────────────────────────────────


def diagnose_residual_confounding(
    df: pd.DataFrame,
    *,
    species: str,
    cfg: RunConfig,
    grid: pd.DataFrame,
    reference_effort: ReferenceEffort,
    n_simulations: int = 5,
    seed: int = 0,
) -> ResidualConfoundingResult:
    """Run null-trend simulations and report the residual confounding bias.

    Parameters
    ----------
    df
        Training frame — same as passed to ``train_predict_species``.
    species
        Species column name.
    cfg, grid, reference_effort
        Passed through to ``train_predict_species``.
    n_simulations
        Number of independent null simulations.  Each is one full pipeline
        fit on bootstrapped null data.  5–10 is sufficient for diagnostics;
        more gives a more stable bias estimate.
    seed
        Base RNG seed.
    """
    logger.info(
        "Residual confounding null diagnostic [%s]: %d simulation(s)",
        species, n_simulations,
    )

    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(n_simulations)

    worker_cfg = replace(cfg, inner_n_jobs=_resolve_inner_n_jobs(cfg))

    jobs = [
        (
            df, species, worker_cfg, grid, reference_effort,
            i,
            int(child_seeds[i].generate_state(1)[0]),
        )
        for i in range(n_simulations)
    ]

    n_workers = max(1, cfg.n_workers)
    if n_workers == 1:
        results_raw = [_one_null_simulation(*args) for args in jobs]
    else:
        from joblib import Parallel, delayed
        logger.info(
            "RC null sims: %d parallel workers  •  inner_n_jobs=%d",
            n_workers, worker_cfg.inner_n_jobs,
        )
        results_raw = Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
            delayed(_one_null_simulation)(*args) for args in jobs
        )

    sim_records: list[pd.DataFrame] = []
    ate_occ_list: list[float] = []
    ate_count_list: list[float] = []
    abd_median_list: list[float] = []
    n_ok = 0

    for sim_idx, result in enumerate(results_raw):
        if result is None:
            continue
        n_ok += 1
        bng_ref = result["bng_ref"]
        n_cells = len(bng_ref)
        sim_records.append(pd.DataFrame({
            "sim_idx": np.full(n_cells, sim_idx, dtype=np.int32),
            "bng_ref": bng_ref,
            "occ_ppy": result["occ_ppy"],
            "count_ppy": result["count_ppy"],
            "abd_ppy": result["abd_ppy"],
        }))
        ate_occ_list.append(result["ate_occ"])
        ate_count_list.append(result["ate_count"])
        abd_median_list.append(result["abd_median"])

    if n_ok == 0:
        logger.warning("All null RC simulations failed")
        return ResidualConfoundingResult(
            species=species,
            bias_ate_occ=float("nan"),
            bias_ate_count=float("nan"),
            bias_ate_abd=float("nan"),
            n_successful=0,
            sim_df=pd.DataFrame(),
        )

    sim_df = pd.concat(sim_records, ignore_index=True)
    sim_ate_occ = np.asarray(ate_occ_list)
    sim_ate_count = np.asarray(ate_count_list)
    sim_ate_abd = np.asarray(abd_median_list)

    bias_ate_occ = float(np.mean(sim_ate_occ))
    bias_ate_count = float(np.mean(sim_ate_count))
    bias_ate_abd = float(np.mean(sim_ate_abd))

    logger.info(
        "RC null-bias [%s]: occ=%+.4f (sd=%.4f)  count=%+.4f (sd=%.4f)"
        "  abd=%+.4f (sd=%.4f)  n=%d/%d",
        species,
        bias_ate_occ,
        float(np.std(sim_ate_occ)) if n_ok > 1 else 0.0,
        bias_ate_count,
        float(np.std(sim_ate_count)) if n_ok > 1 else 0.0,
        bias_ate_abd,
        float(np.std(sim_ate_abd)) if n_ok > 1 else 0.0,
        n_ok, n_simulations,
    )

    return ResidualConfoundingResult(
        species=species,
        bias_ate_occ=bias_ate_occ,
        bias_ate_count=bias_ate_count,
        bias_ate_abd=bias_ate_abd,
        n_successful=n_ok,
        sim_df=sim_df,
        sim_ate_occ=sim_ate_occ,
        sim_ate_count=sim_ate_count,
        sim_ate_abd=sim_ate_abd,
    )
