"""Grid-level ensemble of the two-part DML trend (Johnston 2025, §3.4).

For each ensemble member we draw two *independent* 75% subsamples of the
harmonized surveys (stratified by year) — one for the occurrence stage,
one for the count stage — and run ``train_predict_species`` on the pair.
The two halves of the hurdle model stay genuinely independent, so the
chain-rule sum ``occ_ppy + count_ppy`` is unbiased (Fink §3.2).

Grid-cell point estimates are medians across members; 80% CIs use the
10th/90th percentiles (Fink §3.3 Step 5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bgbw_trends.aggregate import aggregate_surveys
from bgbw_trends.config import ReferenceEffort, RunConfig
from bgbw_trends.features.effort import compute_reference_effort
from bgbw_trends.features.habitat import apply_embedding_pca, reduce_embeddings_pca
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


def _subsample_surveys(
    harmonized: pd.DataFrame,
    fraction: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample ``fraction`` of surveys, stratified by year."""
    frames = []
    for _year, group in harmonized.groupby("year"):
        n_sample = max(1, int(len(group) * fraction))
        idx = rng.choice(group.index, size=n_sample, replace=False)
        frames.append(group.loc[idx])
    return pd.concat(frames, ignore_index=True)


def _build_analysis_df(
    surveys: pd.DataFrame,
    embeddings: pd.DataFrame,
    pca_model: dict | None = None,
) -> pd.DataFrame:
    """Aggregate subsampled surveys to BNG cells and join with embeddings."""
    agg = aggregate_surveys(surveys)
    merged = agg.merge(embeddings, left_on="bng_ref", right_index=True, how="inner")
    if pca_model is not None:
        merged = apply_embedding_pca(merged, pca_model)
    return merged


def fit_ensemble_trend(
    harmonized: pd.DataFrame,
    embeddings: pd.DataFrame,
    cfg: RunConfig,
    species: str,
    grid: pd.DataFrame,
    reference_effort: ReferenceEffort | None = None,
    n_ensemble: int = 25,
    subsample_fraction: float = 0.75,
    pca_model: dict | None = None,
    seed: int = 42,
) -> EnsembleTrendResult:
    """Run an ensemble of two-part DML trend fits and aggregate at the grid.

    Parameters
    ----------
    harmonized
        Individual-level survey data (all years stacked).
    embeddings
        Geospatial embeddings indexed by ``bng_ref``.
    cfg
        Run configuration (RF hyperparameters, PCA dimensionality).
    species
        Species column name.
    grid
        Prediction grid (one row per cell) with habitat features. Effort
        columns are ignored — reference effort is broadcast instead.
    reference_effort
        Fixed effort values for SDM prediction. If ``None``, computed from
        the first successful ensemble subsample's aggregation.
    n_ensemble
        Number of ensemble members.
    subsample_fraction
        Fraction of surveys per subsample.
    pca_model
        Fitted embedding PCA (as returned by ``reduce_embeddings_pca``). If
        ``None`` and ``cfg.n_pca_components`` is set, a PCA is fit on the
        first successful member and reused for all subsequent members.
    seed
        Random seed.
    """
    rng = np.random.default_rng(seed)

    draws_occ: list[np.ndarray] = []
    draws_count: list[np.ndarray] = []
    draws_abd: list[np.ndarray] = []
    p_occ_stack: list[np.ndarray] = []
    e_abund_stack: list[np.ndarray] = []

    prediction: SpeciesPrediction | None = None
    for i in range(n_ensemble):
        logger.info("Ensemble member %d/%d", i + 1, n_ensemble)
        try:
            occ_surveys = _subsample_surveys(harmonized, subsample_fraction, rng)
            count_surveys = _subsample_surveys(harmonized, subsample_fraction, rng)

            occ_df = _build_analysis_df(occ_surveys, embeddings, pca_model)
            count_df = _build_analysis_df(count_surveys, embeddings, pca_model)

            # Fit PCA on first successful member if not supplied
            if pca_model is None and cfg.n_pca_components:
                occ_df, pca_model = reduce_embeddings_pca(
                    occ_df, cfg.n_pca_components
                )
                count_df = _build_analysis_df(count_surveys, embeddings, pca_model)

            if reference_effort is None:
                reference_effort = compute_reference_effort(occ_df)

            prediction = train_predict_species(
                occ_df,
                count_df,
                species=species,
                cfg=cfg,
                grid=grid,
                reference_effort=reference_effort,
                seed=seed + i,
            )

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
        raise RuntimeError(
            f"Only {n_ok}/{n_ensemble} ensemble members succeeded"
        )
    if prediction is None:
        raise RuntimeError("No ensemble member succeeded")

    occ_arr = np.vstack(draws_occ)
    count_arr = np.vstack(draws_count)
    abd_arr = np.vstack(draws_abd)
    p_occ_arr = np.vstack(p_occ_stack)
    e_abund_arr = np.vstack(e_abund_stack)

    logger.info(
        "Ensemble complete: %d/%d members. grid-mean abd_ppy=%+.3f%% "
        "(10th/90th: %+.3f%%, %+.3f%%)",
        n_ok,
        n_ensemble,
        float(np.nanmean(np.nanmedian(abd_arr, axis=0))) * 100,
        float(np.nanmean(np.nanpercentile(abd_arr, 10, axis=0))) * 100,
        float(np.nanmean(np.nanpercentile(abd_arr, 90, axis=0))) * 100,
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
        subsample_fraction=subsample_fraction,
    )
