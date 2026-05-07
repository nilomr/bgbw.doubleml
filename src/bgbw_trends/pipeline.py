"""Unified two-part DML trend + SDM pipeline for a single species.

Direct Python port of the Fink et al. (2023) R workflow in
``trendswf::train_predict_trend`` (docs/refrence/trendswf/R/train-predict.R).
For each species we fit exactly four nuisance random forests (propensity +
marginal × occurrence + count) and two causal forests, reusing the
nuisance fits for both:

1. **SDM**: distribution and relative abundance maps, obtained by predicting
   from the marginal-response forests at the prediction grid with effort
   held at reference values (Fink §3.2, Johnston §3.1).
2. **Trend**: τ(W) via a causal forest that consumes the marginal and
   propensity forests' out-of-bag predictions as ``Y.hat`` and ``W.hat``
   (Fink §2.1, Eq. 4 — the residual-on-residual regression).

OOB predictions come for free from the bagging process (``oob_score=True``),
matching ``grf::regression_forest``'s internal behaviour — 1 forest fit per
nuisance. SHAP's TreeExplainer supports sklearn forests natively.

Chain rule for the two-part abundance trend (Fink §3.2 / train-predict.R L197):

    occ_ppy  = τ_occ_CATE(W) / P(occurrence | W, reference effort)
    abd_ppy  = occ_ppy + τ_count_CATE(W)

``τ_occ_CATE`` is an absolute change in occurrence probability per year; the
division by the marginal probability converts it to a relative rate, so the
sum with the log-count slope is a proper per-year log change in abundance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as _RF
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from bgbw_trends.config import ReferenceEffort, RunConfig
from bgbw_trends.features.effort import make_effort_features
from bgbw_trends.features.habitat import make_habitat_features, make_spatial_features_W

logger = logging.getLogger(__name__)


# ── Diagnostics ───────────────────────────────────────────────────────────


@dataclass
class SpeciesModelDiagnostics:
    """Out-of-fold predictive diagnostics for the nuisance random forests."""

    occurrence_brier: float
    occurrence_auc: float
    occurrence_pr_auc: float
    count_r2: float
    count_rmse: float
    count_mae: float
    propensity_occ_r2: float
    propensity_occ_rmse: float
    propensity_count_r2: float
    propensity_count_rmse: float


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _r2_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    return (
        float(r2_score(y_true, y_pred)),
        float(np.sqrt(mean_squared_error(y_true, y_pred))),
    )


# ── RF factory + OOB fit ──────────────────────────────────────────────────


def _make_rf(cfg: RunConfig):
    """Random Forest for nuisance models (marginal response + propensity).

    Mirrors ``grf::regression_forest`` with ``tune.parameters="all"`` in
    Fink's R code — a regressor used also for binary {0, 1} outcomes
    (following grf, which fits the 0/1 indicator as regression to get
    calibrated probability estimates rather than using a classifier).

    Hyperparameters follow grf defaults: unconstrained depth, leaves
    sized by ``min_samples_leaf`` (configurable on ``RunConfig``). OOB
    predictions are enabled so the full-data fit immediately exposes
    out-of-bag estimates, matching grf's native OOB behaviour — 1 fit
    per nuisance, no separate K-fold cross-fit required.
    """
    return _RF(
        n_estimators=cfg.n_estimators,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
    )


def _predict(model, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Robust predict -> 1-D float64 numpy array."""
    return np.asarray(model.predict(X), dtype=np.float64).ravel()


def _fit_nuisance(
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: RunConfig,
    *,
    seed: int,
):
    """Fit a nuisance RF and return ``(model, oof_predictions)``.

    One forest fit; OOB predictions come from bagging, exactly mirroring
    ``grf::regression_forest`` (1 fit total, no separate K-fold cross-fit).
    The ``seed`` argument is accepted for API symmetry across the four
    nuisance fits but is not used — sklearn RF is seeded via ``RunConfig``
    or left as the global numpy RNG.
    """
    model = _make_rf(cfg)
    model.fit(X, y)
    oob = np.asarray(model.oob_prediction_, dtype=np.float64).ravel()
    bad = ~np.isfinite(oob)
    if bad.any():
        oob[bad] = _predict(model, X.iloc[bad.nonzero()[0]])
    return model, oob


# ── Prediction-unit construction ───────────────────────────────────────────


def _build_prediction_effort(
    training_effort: pd.DataFrame, reference: ReferenceEffort, n: int
) -> pd.DataFrame:
    """Broadcast reference-effort values across ``n`` grid cells.

    Implements Fink's ``prediction_unit`` (workflow/train.R L108-125): freeze
    effort covariates to representative values so the SDM reflects habitat,
    not sampling intensity. Fink uses median for all effort features except CCI
    (observer skill), which uses the 75th percentile to represent a skilled
    observer. BGBW has no CCI analogue, so all features fall back to median.

    Fink's separate ``covariates_to_maximize`` mechanism (weather, time-of-day
    set to optimal-detection values via ``multivariate_maximize``) is not
    implemented here — BGBW is a single weekend event with no intra-annual
    detection-condition covariates to optimise over.
    """
    out = {}
    for col in training_effort.columns:
        if hasattr(reference, col):
            out[col] = np.full(n, float(getattr(reference, col)), dtype=np.float64)
        else:
            out[col] = np.full(
                n, float(training_effort[col].median()), dtype=np.float64
            )
    return pd.DataFrame(out)


# ── Result container ───────────────────────────────────────────────────────


@dataclass
class TrainedSpeciesModels:
    """Fitted forests and training matrices for post-hoc interpretation.

    Returned from ``train_predict_species(..., return_models=True)`` alongside
    a :class:`SpeciesPrediction`. Used by :mod:`bgbw_trends.interpret` to
    compute SHAP values for the marginal forests (SDM drivers) and for
    τ(W) via a surrogate model. Also the right object to pass into
    :func:`predict_grid_variance` if you need sigmas on a subset of cells.
    """

    y_bin_forest: object
    y_count_forest: object
    bin_cf: object  # econml.grf.CausalForest (occurrence)
    count_cf: object  # econml.grf.CausalForest (count)
    X_marg_occ: pd.DataFrame
    X_marg_count: pd.DataFrame
    W_occ: np.ndarray
    W_count: np.ndarray
    W_feature_names: list[str]
    # Precomputed training-time CATE predictions — reused by the SHAP
    # surrogate so we avoid a second cf.predict call and get full-N coverage.
    tau_occ: np.ndarray
    tau_count: np.ndarray


@dataclass
class SpeciesPrediction:
    """Combined SDM + trend predictions for one species.

    Percent-per-year values follow Fink's convention (``train-predict.R`` L203):
    occurrence and abundance ``ppy`` are relative rates (× 100 = percent),
    ``count_ppy`` is the log-count slope (≈ relative rate).
    """

    species: str
    bng_ref: np.ndarray

    # SDM (predicted at grid with reference effort)
    p_occurrence: np.ndarray
    e_count_given_occ: np.ndarray
    e_abundance: np.ndarray

    # Spatial trend τ(W) at grid (ratios; multiply by 100 for percent-per-year)
    occ_ppy: np.ndarray
    count_ppy: np.ndarray
    abd_ppy: np.ndarray

    # Variance estimates from the causal forests (for CI construction)
    occ_sigma: np.ndarray
    count_sigma: np.ndarray

    # Predictive diagnostics for the nuisance models used by this pipeline
    diagnostics: SpeciesModelDiagnostics

    # Bookkeeping
    n_train_occ: int
    n_train_count: int

    # Per-observation PLR components, for yearly-trend and calibration plots.
    # Columns: year, raw, m_hat, s_hat, tau. With these any plot-side
    # consumer can reconstruct fitted = m_hat + tau*(year - s_hat) or
    # the deconfounded line raw.mean() + ate*(year - year.mean()).
    # obs_occ  — occurrence stage (probability scale, [0, 1])
    # obs_count — count stage (log1p scale), detected rows only
    obs_occ: pd.DataFrame = field(default_factory=pd.DataFrame)
    obs_count: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ATE = mean of τ̂(W) over training rows. Probability-per-year for
    # occurrence, log-count-per-year for count. These are the slopes of
    # the deconfounded trend lines (Fink §2.1; vignette fig1a analogue).
    ate_occ: float = float("nan")
    ate_count: float = float("nan")


# ── Feature assembly helpers ───────────────────────────────────────────────


def _assemble_X(
    df: pd.DataFrame,
    columns: Sequence[str] | None,
) -> pd.DataFrame:
    """Build the feature matrix used by a given forest.

    ``columns=None`` means "use X_hab + X_effort" (the pipeline default,
    matching Fink's shared propensity/marginal feature set). Passing an
    explicit list mirrors Fink's ``propensity_covariates`` /
    ``marginal_covariates`` arguments.
    """
    if columns is None:
        X_hab = make_habitat_features(df)
        X_eff = make_effort_features(df)
        return pd.concat([X_hab, X_eff], axis=1).reset_index(drop=True)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns missing from frame: {missing}")
    return df[list(columns)].astype(np.float64).reset_index(drop=True)


def _assemble_W(
    df: pd.DataFrame,
    columns: Sequence[str] | None,
) -> np.ndarray:
    """Build the heterogeneity matrix W. ``None`` means embeddings + lat/lon."""
    if columns is None:
        return make_spatial_features_W(df).to_numpy(dtype=np.float64)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Heterogeneity columns missing from frame: {missing}")
    return df[list(columns)].astype(np.float64).to_numpy()


def _assemble_W_names(
    df: pd.DataFrame,
    columns: Sequence[str] | None,
) -> list[str]:
    """Names of the columns that ``_assemble_W`` produces, in order."""
    if columns is None:
        return list(make_spatial_features_W(df).columns)
    return list(columns)


def predict_grid_variance(
    models: "TrainedSpeciesModels",
    grid: pd.DataFrame,
    *,
    treatment_covariates: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute τ(W) standard errors on a grid using the trained causal forests.

    Split out from ``train_predict_species`` because variance estimation
    via ``predict_and_var`` is expensive at 1 km BGBW scale — often 5-10×
    the cost of the point prediction. Call this only on the cells you
    actually plan to map with significance shading, e.g. a filtered
    subset of the full grid.

    Returns ``(sigma_occ, sigma_count)`` aligned with ``grid`` rows.
    """
    W_grid = _assemble_W(grid, treatment_covariates)
    _, var_occ = models.bin_cf.predict_and_var(W_grid)
    _, var_count = models.count_cf.predict_and_var(W_grid)
    sigma_occ = np.sqrt(np.asarray(var_occ).reshape(len(W_grid), -1)[:, 0])
    sigma_count = np.sqrt(np.asarray(var_count).reshape(len(W_grid), -1)[:, 0])
    return sigma_occ, sigma_count


# ── The pipeline ──────────────────────────────────────────────────────────


def train_predict_species(
    df_occ: pd.DataFrame,
    df_count: pd.DataFrame,
    *,
    species: str,
    cfg: RunConfig,
    grid: pd.DataFrame,
    reference_effort: ReferenceEffort,
    propensity_covariates: Sequence[str] | None = None,
    marginal_covariates: Sequence[str] | None = None,
    treatment_covariates: Sequence[str] | None = None,
    seed: int = 0,
    compute_variance: bool = False,
    return_models: bool = False,
) -> "SpeciesPrediction | tuple[SpeciesPrediction, TrainedSpeciesModels]":
    """Fit the two-part DML trend + SDM for one species.

    Parameters
    ----------
    df_occ, df_count
        Aggregated ``(bng_ref, year)`` rows for the occurrence / count
        stages. Must contain ``year``, ``bng_ref``, the species count
        column, and the feature columns. Pass the *same* frame twice if
        you only have one subsample; for unbiased hurdle trends pass two
        *independently* drawn subsamples (Johnston §3.4).
    species
        Species count column.
    cfg
        ``RunConfig`` controlling RF hyperparameters.
    grid
        One row per prediction cell (``bng_ref`` + features). Effort
        columns may be absent — reference values are broadcast.
    reference_effort
        Fixed effort values for SDM prediction (Fink §3.3 / Johnston §3.1).
    propensity_covariates, marginal_covariates
        Feature sets for the propensity and marginal forests. ``None``
        → X_hab + X_effort (the default "all confounders" setup).
        Separate sets mirror Fink's signature where
        e.g. propensity can include short-term temporal features that
        marginal excludes.
    treatment_covariates
        Feature set over which τ(W) varies. ``None`` →
        embeddings + lat/lon (Fink §3.3).
    seed
        Random seed for the causal forest.
    compute_variance
        If True, call ``CausalForest.predict_and_var`` on the grid to
        populate ``occ_sigma`` / ``count_sigma``. This is expensive at
        grid scale; default False (sigma arrays are returned empty).
        Use :func:`predict_grid_variance` to compute sigmas separately
        on a subset of cells when needed for significance maps.

    Returns
    -------
    SpeciesPrediction
        SDM and trend fields aligned on ``bng_ref``.
    """
    from econml.grf import CausalForest

    logger.info(
        "train_predict_species[%s] n_occ=%d n_count=%d",
        species,
        len(df_occ),
        len(df_count),
    )

    # ── Occurrence stage ────────────────────────────────────────────────
    # Fink train-predict.R L88-146: marginal RF on (X, Y_bin), propensity
    # RF on (X, year), then causal forest on residuals with W as the
    # heterogeneity space.
    Y_bin = (df_occ[species] > 0).to_numpy(dtype=np.float64)
    T_occ = df_occ["year"].to_numpy(dtype=np.float64)

    X_marg_occ = _assemble_X(df_occ, marginal_covariates)
    # If the propensity and marginal covariate sets are identical
    # (the default), share the assembled frame instead of rebuilding it.
    if propensity_covariates is marginal_covariates or (
        propensity_covariates is None and marginal_covariates is None
    ):
        X_prop_occ = X_marg_occ
    else:
        X_prop_occ = _assemble_X(df_occ, propensity_covariates)
    W_occ = _assemble_W(df_occ, treatment_covariates)

    y_bin_forest, Y_bin_hat = _fit_nuisance(X_marg_occ, Y_bin, cfg, seed=seed)
    Y_bin_hat = np.clip(Y_bin_hat, 0.0, 1.0)
    _, W_bin_hat = _fit_nuisance(X_prop_occ, T_occ, cfg, seed=seed + 1)

    # Causal forest on residuals → τ_occ(W). econml's CausalForest does
    # not accept pre-computed Y.hat / W.hat (unlike grf), so we pass
    # residualised inputs — by Frisch-Waugh-Lovell this identifies the
    # same τ as grf's residual-on-residual regression (Robinson 1988).
    bin_cf = CausalForest(
        n_estimators=cfg.cf_n_estimators,
        min_samples_leaf=cfg.cf_min_samples_leaf,
        honest=True,
        inference=True,
        random_state=seed,
        n_jobs=-1,
    )
    bin_cf.fit(X=W_occ, T=(T_occ - W_bin_hat), y=(Y_bin - Y_bin_hat))

    # ── Count stage — detected rows only, log1p outcome ────────────────
    detected = (df_count[species] > 0).to_numpy(dtype=bool)
    dc = df_count.loc[detected].reset_index(drop=True)
    if len(dc) < 50:
        raise ValueError(
            f"train_predict_species[{species}]: only {len(dc)} detections — "
            "too few to fit the count stage"
        )

    # Fink uses log(obs + 1) (alpha = 1) exactly — aggregated per-cell
    # counts may be <1 so log1p matches the reference (L149-151).
    Y_log = np.log1p(dc[species].to_numpy(dtype=np.float64))
    T_count = dc["year"].to_numpy(dtype=np.float64)

    X_marg_count = _assemble_X(dc, marginal_covariates)
    if propensity_covariates is marginal_covariates or (
        propensity_covariates is None and marginal_covariates is None
    ):
        X_prop_count = X_marg_count
    else:
        X_prop_count = _assemble_X(dc, propensity_covariates)
    W_count = _assemble_W(dc, treatment_covariates)

    y_count_forest, Y_count_hat = _fit_nuisance(X_marg_count, Y_log, cfg, seed=seed + 2)
    _, W_count_hat = _fit_nuisance(X_prop_count, T_count, cfg, seed=seed + 3)

    diagnostics = SpeciesModelDiagnostics(
        occurrence_brier=float(brier_score_loss(Y_bin, Y_bin_hat)),
        occurrence_auc=_safe_roc_auc(Y_bin, Y_bin_hat),
        occurrence_pr_auc=_safe_pr_auc(Y_bin, Y_bin_hat),
        count_r2=_r2_rmse(Y_log, Y_count_hat)[0],
        count_rmse=_r2_rmse(Y_log, Y_count_hat)[1],
        count_mae=float(mean_absolute_error(Y_log, Y_count_hat)),
        propensity_occ_r2=_r2_rmse(T_occ, W_bin_hat)[0],
        propensity_occ_rmse=_r2_rmse(T_occ, W_bin_hat)[1],
        propensity_count_r2=_r2_rmse(T_count, W_count_hat)[0],
        propensity_count_rmse=_r2_rmse(T_count, W_count_hat)[1],
    )

    count_cf = CausalForest(
        n_estimators=cfg.cf_n_estimators,
        min_samples_leaf=cfg.cf_min_samples_leaf,
        honest=True,
        inference=True,
        random_state=seed,
        n_jobs=-1,
    )
    count_cf.fit(X=W_count, T=(T_count - W_count_hat), y=(Y_log - Y_count_hat))

    # ── Per-observation PLR components ────────────────────────────────
    # Store raw inputs and nuisance/causal predictions per training row,
    # on the stage's native scale. Plots derive yearly summaries from
    # these; we avoid pre-aggregating here because the right summary
    # (raw mean vs conditional-mean projection vs deconfounded line)
    # depends on what the consumer wants to show.
    tau_occ_train = np.asarray(bin_cf.predict(W_occ)).ravel()
    obs_occ = pd.DataFrame(
        {
            "year": T_occ.astype(int),
            "raw": Y_bin,
            "m_hat": Y_bin_hat,
            "s_hat": W_bin_hat,
            "tau": tau_occ_train,
        }
    )
    ate_occ = float(np.mean(tau_occ_train))

    tau_count_train = np.asarray(count_cf.predict(W_count)).ravel()
    obs_count = pd.DataFrame(
        {
            "year": T_count.astype(int),
            "raw": Y_log,
            "m_hat": Y_count_hat,
            "s_hat": W_count_hat,
            "tau": tau_count_train,
        }
    )
    ate_count = float(np.mean(tau_count_train))

    # ── SDM prediction at the grid, effort fixed to reference ─────────
    grid = grid.reset_index(drop=True)
    X_hab_grid = make_habitat_features(grid)
    n_grid = len(X_hab_grid)

    # Build effort frames aligned with whatever columns each forest saw.
    # When the caller supplies explicit covariate lists these may or may
    # not include effort columns; we detect that per forest.
    def _build_grid_frame(
        train_X: pd.DataFrame,
        train_eff: pd.DataFrame,
    ) -> pd.DataFrame:
        eff_cols = [c for c in train_X.columns if c in train_eff.columns]
        hab_cols = [c for c in train_X.columns if c not in train_eff.columns]
        parts: list[pd.DataFrame] = []
        if hab_cols:
            parts.append(X_hab_grid.reindex(columns=hab_cols))
        if eff_cols:
            eff_grid = _build_prediction_effort(
                train_eff[eff_cols], reference_effort, n_grid
            )
            parts.append(eff_grid)
        out = pd.concat(parts, axis=1)
        # enforce training-time column order (critical for sklearn RF)
        return out[list(train_X.columns)]

    # Effort frames for the training data (source of medians for non-reference cols)
    X_eff_train_occ = make_effort_features(df_occ)
    X_eff_train_count = make_effort_features(dc)

    X_grid_occ = _build_grid_frame(X_marg_occ, X_eff_train_occ)
    p_occ = np.clip(_predict(y_bin_forest, X_grid_occ), 0.0, 1.0)

    X_grid_count = _build_grid_frame(X_marg_count, X_eff_train_count)
    log_count_hat = _predict(y_count_forest, X_grid_count)
    e_count = np.maximum(np.expm1(log_count_hat), 0.0)

    e_abundance = p_occ * e_count

    # ── Trend prediction at the grid ──────────────────────────────────
    # Variance estimation (predict_and_var) is expensive at grid scale —
    # it computes per-tree leaf statistics for every cell. Default is
    # point prediction only; call predict_grid_variance separately when
    # significance maps are needed.
    W_grid = _assemble_W(grid, treatment_covariates)
    if compute_variance:
        tau_occ, var_occ = bin_cf.predict_and_var(W_grid)
        tau_count, var_count = count_cf.predict_and_var(W_grid)
        tau_occ = np.asarray(tau_occ).ravel()
        tau_count = np.asarray(tau_count).ravel()
        sigma_occ = np.sqrt(np.asarray(var_occ).reshape(len(tau_occ), -1)[:, 0])
        sigma_count = np.sqrt(np.asarray(var_count).reshape(len(tau_count), -1)[:, 0])
    else:
        tau_occ = np.asarray(bin_cf.predict(W_grid)).ravel()
        tau_count = np.asarray(count_cf.predict(W_grid)).ravel()
        sigma_occ = np.array([], dtype=np.float64)
        sigma_count = np.array([], dtype=np.float64)

    # Chain rule (Fink train-predict.R L197-199): occurrence CATE is an
    # absolute-probability rate; dividing by P(occ) converts it to a
    # relative rate, matching the log-count slope's units. We set
    # infinite ratios (tiny P(occ)) to NaN exactly as Fink does.
    with np.errstate(divide="ignore", invalid="ignore"):
        occ_ppy = np.where(p_occ > 0.0, tau_occ / p_occ, np.nan)
    occ_ppy = np.where(np.isfinite(occ_ppy), occ_ppy, np.nan)
    count_ppy = tau_count
    abd_ppy = occ_ppy + count_ppy
    abd_ppy = np.where(np.isfinite(abd_ppy), abd_ppy, np.nan)

    logger.info(
        "train_predict_species[%s]: median P(occ)=%.3f E[count|occ]=%.2f "
        "occ_ppy=%+.3f%% count_ppy=%+.3f%% abd_ppy=%+.3f%%",
        species,
        float(np.nanmedian(p_occ)),
        float(np.nanmedian(e_count)),
        float(np.nanmedian(occ_ppy)) * 100,
        float(np.nanmedian(count_ppy)) * 100,
        float(np.nanmedian(abd_ppy)) * 100,
    )

    prediction = SpeciesPrediction(
        species=species,
        bng_ref=grid["bng_ref"].to_numpy(),
        p_occurrence=p_occ,
        e_count_given_occ=e_count,
        e_abundance=e_abundance,
        occ_ppy=occ_ppy,
        count_ppy=count_ppy,
        abd_ppy=abd_ppy,
        occ_sigma=sigma_occ,
        count_sigma=sigma_count,
        diagnostics=diagnostics,
        n_train_occ=len(df_occ),
        n_train_count=len(dc),
        obs_occ=obs_occ,
        obs_count=obs_count,
        ate_occ=ate_occ,
        ate_count=ate_count,
    )
    if return_models:
        W_names = _assemble_W_names(df_occ, treatment_covariates)
        models = TrainedSpeciesModels(
            y_bin_forest=y_bin_forest,
            y_count_forest=y_count_forest,
            bin_cf=bin_cf,
            count_cf=count_cf,
            X_marg_occ=X_marg_occ,
            X_marg_count=X_marg_count,
            W_occ=W_occ,
            W_count=W_count,
            W_feature_names=W_names,
            tau_occ=tau_occ_train,
            tau_count=tau_count_train,
        )
        return prediction, models
    return prediction
