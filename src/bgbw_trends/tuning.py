"""Per-species hyperparameter tuning for the four nuisance forests.

Mirrors the spirit of ``grf::regression_forest(..., tune.parameters="all")``
in trendswf, but on a smaller search space and with the result *frozen* once
per species. Rationale:

- ``min_samples_leaf`` (grf: ``min.node.size``) and ``max_features`` (grf:
  ``mtry``) are tuned jointly on a small grid evaluated by OOB R². These two
  are the highest-leverage knobs for bias-variance in bagged regression
  forests (Probst et al. 2019). ``n_estimators`` and other parameters are
  kept at their cfg / sklearn defaults.
- Each of the four nuisance stages (occurrence × marginal/propensity,
  count × marginal/propensity) is tuned independently — they have
  different X distributions and outcomes.
- Tuning runs once per species at the start of a run; the resulting
  ``TunedNuisanceParams`` is then passed through every ensemble member
  and residual-confounding simulation unchanged ("tune once, freeze, reuse").

The causal forests are NOT tuned here — econml's CausalForest has no
native OOB score, and their performance is dominated by the quality
of the residualised inputs (which the nuisance tuning improves).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from bgbw_trends.config import RunConfig
from bgbw_trends.pipeline import _assemble_X, _make_rf

logger = logging.getLogger(__name__)


DEFAULT_LEAF_GRID: tuple[int, ...] = (5, 20, 50)
DEFAULT_MAX_FEATURES_GRID: tuple = ("sqrt", 0.3, 0.5)


@dataclass
class TunedNuisanceParams:
    """Per-stage tuned ``min_samples_leaf`` and ``max_features`` for the four
    nuisance forests.

    All values default to the cfg-level / sklearn fallbacks so a partially-
    populated instance is safe to pass through.
    """

    occ_marginal: int
    occ_marginal_max_features: str | float
    occ_propensity: int
    occ_propensity_max_features: str | float
    count_marginal: int
    count_marginal_max_features: str | float
    count_propensity: int
    count_propensity_max_features: str | float

    # OOB R² achieved at the chosen params — kept for diagnostics / reporting.
    occ_marginal_oob_r2: float
    occ_propensity_oob_r2: float
    count_marginal_oob_r2: float
    count_propensity_oob_r2: float


def _tune_forest(
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: RunConfig,
    *,
    label: str,
    leaf_grid: Sequence[int] = DEFAULT_LEAF_GRID,
    max_features_grid: Sequence[str | float] = DEFAULT_MAX_FEATURES_GRID,
) -> tuple[int, str | float, float]:
    """Pick ``(min_samples_leaf, max_features)`` maximising OOB R² on (X, y).

    Exhaustive search over the Cartesian product of ``leaf_grid`` ×
    ``max_features_grid``. Each candidate fits a full
    ``cfg.n_estimators``-tree forest (OOB R² is biased at small
    ``n_estimators``). Returns ``(best_leaf, best_max_features, best_score)``.
    """
    best_leaf: int = leaf_grid[0]
    best_mf: str | float = max_features_grid[0]
    best_score = -np.inf

    for leaf in leaf_grid:
        for mf in max_features_grid:
            rf = _make_rf(cfg, min_samples_leaf=leaf, max_features=mf)
            rf.fit(X, y)
            score = float(rf.oob_score_)
            if score > best_score:
                best_score = score
                best_leaf = leaf
                best_mf = mf

    logger.info(
        "tune[%s]: best (leaf=%d, max_features=%s) OOB R²=%+.4f  "
        "[grid %d×%d]",
        label,
        best_leaf,
        best_mf,
        best_score,
        len(leaf_grid),
        len(max_features_grid),
    )
    return best_leaf, best_mf, best_score


def tune_nuisance_for_species(
    df_occ: pd.DataFrame,
    df_count: pd.DataFrame,
    *,
    species: str,
    cfg: RunConfig,
    propensity_covariates: Sequence[str] | None = None,
    marginal_covariates: Sequence[str] | None = None,
    leaf_grid: Sequence[int] = DEFAULT_LEAF_GRID,
    max_features_grid: Sequence[str | float] = DEFAULT_MAX_FEATURES_GRID,
) -> TunedNuisanceParams:
    """Tune ``min_samples_leaf`` and ``max_features`` for each nuisance forest.

    Mirrors the feature-assembly logic of ``train_predict_species`` so the
    tuned hyperparameters are valid for the exact matrices the pipeline
    will fit later. The propensity stage's outcome is ``year`` (the
    treatment), the marginal stage's outcome is the species response.

    Parameters
    ----------
    df_occ, df_count
        Same frames you would pass to ``train_predict_species``. The count
        side is filtered to detected rows (matching the pipeline's count
        stage). If fewer than 50 detections are present, the count-stage
        tuners fall back to ``cfg.rf_min_samples_leaf`` / ``"sqrt"`` rather
        than risking an unstable fit.
    propensity_covariates, marginal_covariates
        Match the eventual pipeline call. ``None`` means default
        (X_hab + X_effort).
    leaf_grid
        Candidate ``min_samples_leaf`` values. Default ``(5, 20, 50)``.
    max_features_grid
        Candidate ``max_features`` values. Default ``("sqrt", 0.3, 0.5)``.

    Returns
    -------
    TunedNuisanceParams
        Pass to ``train_predict_species(..., nuisance_params=...)`` and
        on to ``fit_ensemble_trend`` / ``diagnose_residual_confounding``.
    """
    # ── Occurrence stage ──────────────────────────────────────────────
    Y_bin = (df_occ[species] > 0).to_numpy(dtype=np.float64)
    T_occ = df_occ["year"].to_numpy(dtype=np.float64)
    X_marg_occ = _assemble_X(df_occ, marginal_covariates)
    if propensity_covariates is marginal_covariates or (
        propensity_covariates is None and marginal_covariates is None
    ):
        X_prop_occ = X_marg_occ
    else:
        X_prop_occ = _assemble_X(df_occ, propensity_covariates)

    occ_marg_leaf, occ_marg_mf, occ_marg_r2 = _tune_forest(
        X_marg_occ, Y_bin, cfg,
        label=f"{species}/occ-marginal",
        leaf_grid=leaf_grid, max_features_grid=max_features_grid,
    )
    occ_prop_leaf, occ_prop_mf, occ_prop_r2 = _tune_forest(
        X_prop_occ, T_occ, cfg,
        label=f"{species}/occ-propensity",
        leaf_grid=leaf_grid, max_features_grid=max_features_grid,
    )

    # ── Count stage (detected rows only, log1p outcome) ───────────────
    detected = (df_count[species] > 0).to_numpy(dtype=bool)
    dc = df_count.loc[detected].reset_index(drop=True)
    if len(dc) < 50:
        logger.warning(
            "tune[%s]: only %d detections — skipping count-stage tuning, "
            "falling back to cfg.rf_min_samples_leaf=%d, max_features='sqrt'",
            species,
            len(dc),
            cfg.rf_min_samples_leaf,
        )
        return TunedNuisanceParams(
            occ_marginal=occ_marg_leaf,
            occ_marginal_max_features=occ_marg_mf,
            occ_propensity=occ_prop_leaf,
            occ_propensity_max_features=occ_prop_mf,
            count_marginal=cfg.rf_min_samples_leaf,
            count_marginal_max_features="sqrt",
            count_propensity=cfg.rf_min_samples_leaf,
            count_propensity_max_features="sqrt",
            occ_marginal_oob_r2=occ_marg_r2,
            occ_propensity_oob_r2=occ_prop_r2,
            count_marginal_oob_r2=float("nan"),
            count_propensity_oob_r2=float("nan"),
        )

    Y_log = np.log1p(dc[species].to_numpy(dtype=np.float64))
    T_count = dc["year"].to_numpy(dtype=np.float64)
    X_marg_count = _assemble_X(dc, marginal_covariates)
    if propensity_covariates is marginal_covariates or (
        propensity_covariates is None and marginal_covariates is None
    ):
        X_prop_count = X_marg_count
    else:
        X_prop_count = _assemble_X(dc, propensity_covariates)

    count_marg_leaf, count_marg_mf, count_marg_r2 = _tune_forest(
        X_marg_count, Y_log, cfg,
        label=f"{species}/count-marginal",
        leaf_grid=leaf_grid, max_features_grid=max_features_grid,
    )
    count_prop_leaf, count_prop_mf, count_prop_r2 = _tune_forest(
        X_prop_count, T_count, cfg,
        label=f"{species}/count-propensity",
        leaf_grid=leaf_grid, max_features_grid=max_features_grid,
    )

    return TunedNuisanceParams(
        occ_marginal=occ_marg_leaf,
        occ_marginal_max_features=occ_marg_mf,
        occ_propensity=occ_prop_leaf,
        occ_propensity_max_features=occ_prop_mf,
        count_marginal=count_marg_leaf,
        count_marginal_max_features=count_marg_mf,
        count_propensity=count_prop_leaf,
        count_propensity_max_features=count_prop_mf,
        occ_marginal_oob_r2=occ_marg_r2,
        occ_propensity_oob_r2=occ_prop_r2,
        count_marginal_oob_r2=count_marg_r2,
        count_propensity_oob_r2=count_prop_r2,
    )
