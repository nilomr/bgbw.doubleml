"""Grid-level ensemble of the two-part DML trend (Johnston 2025, §3.4).

For each ensemble member the assembled cell-year data are split 50/50
(stratified by year) into non-overlapping halves — one for the occurrence
stage, one for the count stage — via ``split_cell_years``. Each stage
trains on its own set of cells so the chain-rule sum
``occ_ppy + count_ppy`` is unbiased.

This is the correct split level for BGBW: the median cell has only 2
surveys per year, so splitting at the individual-survey level would reduce most cells to single-observation
statistics. Splitting whole cell-year rows preserves each cell's
aggregated statistics while still guaranteeing hurdle independence for the DML assumptions.

Grid-cell point estimates are medians across members; 80% CIs use the
10th/90th percentiles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bgbw_trends.config import ReferenceEffort, RunConfig
from bgbw_trends.features.effort import compute_reference_effort
from bgbw_trends.pipeline import SpeciesPrediction, train_predict_species

logger = logging.getLogger(__name__)


@dataclass
class EnsembleTrendResult:
    """Grid-cell ensemble aggregates for one species.

    ``bng_ref`` aligns all arrays. Per-year rates are *ratios* (multiply by
    100 for percent-per-year). Point estimates are the across-member median;
    ``lower``/``upper`` are 10th/90th percentiles.
    """

    species: str
    bng_ref: np.ndarray
    # Point estimates (median across members)
    p_occurrence: np.ndarray
    e_abundance: np.ndarray
    occ_ppy: np.ndarray
    count_ppy: np.ndarray
    abd_ppy: np.ndarray
    # 80% CIs on abd_ppy (10th/90th percentile across members)
    abd_ppy_lower: np.ndarray
    abd_ppy_upper: np.ndarray
    # Full draws, shape (n_successful, n_cells)
    abd_ppy_draws: np.ndarray
    # Metadata
    n_ensemble: int
    n_successful: int
    subsample_fraction: float


def split_cell_years(
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split assembled cell-year rows 50/50, stratified by year, with zero overlap.

    Rows are shuffled within each year then divided at the midpoint. Each
    cell-year row goes to exactly one stage, so the occurrence and count halves
    are mutually exclusive and the chain-rule sum is unbiased.

    Splitting at the cell-year level (rather than at the individual-survey
    level) preserves each cell's aggregated statistics. For BGBW, where the
    median cell has 2 surveys per year, survey-level splitting would reduce
    most cells to single binary observations.
    """
    occ_frames, count_frames = [], []
    for _year, group in df.groupby("year"):
        idx = rng.permutation(group.index)
        mid = len(idx) // 2
        occ_frames.append(group.loc[idx[:mid]])
        count_frames.append(group.loc[idx[mid:]])
    return (
        pd.concat(occ_frames, ignore_index=True),
        pd.concat(count_frames, ignore_index=True),
    )


def fit_ensemble_trend(
    df: pd.DataFrame,
    cfg: RunConfig,
    species: str,
    grid: pd.DataFrame,
    reference_effort: ReferenceEffort | None = None,
    n_ensemble: int = 25,
    seed: int = 42,
) -> EnsembleTrendResult:
    """Run an ensemble of two-part DML trend fits over the assembled data.

    Parameters
    ----------
    df
        Assembled analysis DataFrame (one row per cell-year) as returned by
        ``assemble_analysis_data``, with embeddings and weather already
        joined. PCA should already have been applied if ``cfg.n_pca_components``
        is set.
    cfg
        Run configuration (RF hyperparameters).
    species
        Species column name.
    grid
        Prediction grid (one row per cell) with habitat features. Effort
        columns are ignored — reference effort is broadcast instead.
    reference_effort
        Fixed effort values for SDM prediction. If ``None``, computed from
        ``df``.
    n_ensemble
        Number of ensemble members.
    seed
        Base random seed; member ``i`` uses ``seed + i``.
    """
    rng = np.random.default_rng(seed)

    if reference_effort is None:
        reference_effort = compute_reference_effort(df)

    draws_occ: list[np.ndarray] = []
    draws_count: list[np.ndarray] = []
    draws_abd: list[np.ndarray] = []
    p_occ_stack: list[np.ndarray] = []
    e_abund_stack: list[np.ndarray] = []

    prediction: SpeciesPrediction | None = None
    for i in range(n_ensemble):
        logger.info("Ensemble member %d/%d", i + 1, n_ensemble)
        try:
            occ_df, count_df = split_cell_years(df, rng)

            result = train_predict_species(
                occ_df,
                count_df,
                species=species,
                cfg=cfg,
                grid=grid,
                reference_effort=reference_effort,
                seed=seed + i,
            )
            # train_predict_species returns a plain SpeciesPrediction when
            # return_models is False (the default).
            assert isinstance(result, SpeciesPrediction)
            prediction = result

            draws_occ.append(prediction.occ_ppy)
            draws_count.append(prediction.count_ppy)
            draws_abd.append(prediction.abd_ppy)
            p_occ_stack.append(prediction.p_occurrence)
            e_abund_stack.append(prediction.e_abundance)
        except Exception as e:  # noqa: BLE001 — isolate failing members
            logger.warning("Ensemble member %d failed: %s", i + 1, e)
            continue

    n_ok = len(draws_abd)
    if n_ok < 3:
        raise RuntimeError(f"Only {n_ok}/{n_ensemble} ensemble members succeeded")
    if prediction is None:
        raise RuntimeError("No ensemble member succeeded")

    occ_arr = np.vstack(draws_occ)
    count_arr = np.vstack(draws_count)
    abd_arr = np.vstack(draws_abd)
    p_occ_arr = np.vstack(p_occ_stack)
    e_abund_arr = np.vstack(e_abund_stack)

    logger.info(
        "Ensemble complete: %d/%d members. grid-median abd_ppy=%+.3f%% "
        "(10th/90th: %+.3f%%, %+.3f%%)",
        n_ok,
        n_ensemble,
        float(np.nanmedian(np.nanmedian(abd_arr, axis=0))) * 100,
        float(np.nanmedian(np.nanpercentile(abd_arr, 10, axis=0))) * 100,
        float(np.nanmedian(np.nanpercentile(abd_arr, 90, axis=0))) * 100,
    )

    return EnsembleTrendResult(
        species=species,
        bng_ref=prediction.bng_ref,
        p_occurrence=np.nanmedian(p_occ_arr, axis=0),
        e_abundance=np.nanmedian(e_abund_arr, axis=0),
        occ_ppy=np.nanmedian(occ_arr, axis=0),
        count_ppy=np.nanmedian(count_arr, axis=0),
        abd_ppy=np.nanmedian(abd_arr, axis=0),
        abd_ppy_lower=np.nanpercentile(abd_arr, 10, axis=0),
        abd_ppy_upper=np.nanpercentile(abd_arr, 90, axis=0),
        abd_ppy_draws=abd_arr,
        n_ensemble=n_ensemble,
        n_successful=n_ok,
        subsample_fraction=0.5,
    )
