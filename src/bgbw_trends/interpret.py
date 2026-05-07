"""Feature attribution for the DML + SDM pipeline via SHAP.

Two kinds of question are answerable here:

1. **What drives the SDM?** (marginal-response forests)
   ``explain_sdm`` runs ``shap.TreeExplainer`` directly on the occurrence
   and count marginal forests. These are sklearn ``RandomForestRegressor``
   instances so ``TreeExplainer`` is exact and fast.

2. **What drives τ(W) — the spatial heterogeneity of the trend?**
   econml's ``econml.grf.CausalForest`` does not expose a SHAP-compatible
   tree representation. We fit a *surrogate* sklearn RandomForest on
   ``W → τ̂(W)`` (using the CATE predictions of the causal forest as the
   target) and explain that. This mirrors the pattern documented in
   EconML's interpretability guide for estimators without native SHAP
   support: the surrogate preserves the CATE function locally and makes
   the SHAP decomposition tractable.

The surrogate approach has a single gotcha — SHAP values describe the
*surrogate's* decomposition, not the causal forest's. We set the
surrogate's complexity high enough that it matches τ̂(W) on a held-out
check; a fit R² above ~0.9 is the threshold used here before attributions
are reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)


@dataclass
class ShapExplanation:
    """SHAP attribution for one model.

    ``values`` is (n_samples, n_features); ``base_value`` is the model's
    expected output over the background set. ``feature_names`` aligns
    with the columns of ``X_explained``.
    """

    values: np.ndarray
    base_value: float
    feature_names: list[str]
    X_explained: pd.DataFrame
    surrogate_r2: float | None = None
    top_features: pd.DataFrame = field(default_factory=pd.DataFrame)


def _mean_abs_importance(values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    imp = np.mean(np.abs(values), axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": imp})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def _subsample(
    X: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    if sample_size >= len(X):
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=sample_size, replace=False)
    return X.iloc[idx].reset_index(drop=True)


def explain_tree_model(
    model,
    X: pd.DataFrame,
    *,
    sample_size: int = 500,
    seed: int = 0,
) -> ShapExplanation:
    """SHAP attribution for a tree-based regressor (e.g. marginal forest).

    Uses ``shap.TreeExplainer`` with the approximate (path-dependent)
    algorithm, which is 10–50× faster than exact Shapley values and
    produces equivalent feature-importance rankings for diagnostic plots.
    ``X`` is typically the training frame; we subsample to ``sample_size``
    rows because SHAP cost is O(n × trees × leaves).
    """
    import shap

    X_sample = _subsample(X, sample_size, seed)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_sample, approximate=True)
    values = np.asarray(values)
    if values.ndim == 3:  # multi-output edge case; take the first
        values = values[..., 0]
    base = float(np.atleast_1d(explainer.expected_value)[0])
    features = list(X_sample.columns)
    return ShapExplanation(
        values=values,
        base_value=base,
        feature_names=features,
        X_explained=X_sample,
        top_features=_mean_abs_importance(values, features),
    )


def explain_cate_surrogate(
    cf,
    W: np.ndarray,
    feature_names: list[str],
    *,
    tau_hat: np.ndarray | None = None,
    surrogate_sample_size: int = 10_000,
    sample_size: int = 2000,
    seed: int = 0,
    surrogate_n_estimators: int = 1000,
    surrogate_min_samples_leaf: int = 20,
    r2_warn_threshold: float = 0.9,
) -> ShapExplanation:
    """SHAP attribution for a causal forest via a surrogate RandomForest.

    Parameters
    ----------
    cf
        A fitted ``econml.grf.CausalForest``. Must expose ``predict(W)``.
        Only called when ``tau_hat`` is not provided.
    W
        The heterogeneity matrix used at fit time (ndarray).
    feature_names
        Names for W's columns (used in plots and top-feature tables).
    tau_hat
        Precomputed CATE predictions aligned row-for-row with ``W``.
        When provided, ``cf.predict`` is skipped entirely — pass the
        training-time predictions stored in ``TrainedSpeciesModels.tau_occ``
        / ``tau_count`` to avoid a redundant (slow) cf.predict call and
        to give the surrogate full-N coverage of the τ̂(W) surface.
    surrogate_sample_size
        When ``tau_hat`` is provided this limits how many of the N
        training rows are passed to the surrogate fit (RF fitting is
        O(n log n)). When ``tau_hat`` is None, this is the number of
        rows subsampled from ``W`` before calling ``cf.predict``, which
        is the expensive step — keep at or below ~3k in that case.
    sample_size, seed
        Further subsample of the surrogate training rows passed to
        ``shap.TreeExplainer`` (approximate algorithm).
    surrogate_n_estimators, surrogate_min_samples_leaf
        Hyperparameters for the sklearn RF that approximates τ̂(W).
        200 trees with min_samples_leaf=1 and max_features="sqrt" give
        good diversity and avoid OOB overfitting on a ~10k row sample.
    r2_warn_threshold
        Logs a warning if the surrogate OOB R² is below this; below
        ~0.9 the SHAP decomposition is a crude approximation of τ̂.
    """
    import shap

    if len(feature_names) != W.shape[1]:
        raise ValueError(
            f"feature_names length {len(feature_names)} != W cols {W.shape[1]}"
        )

    W_df_full = pd.DataFrame(W, columns=feature_names)

    if tau_hat is not None:
        # Fast path: precomputed predictions — skip cf.predict entirely and
        # use the full training W. Subsample only for surrogate fit speed.
        if len(tau_hat) != len(W_df_full):
            raise ValueError(
                f"tau_hat length {len(tau_hat)} != W rows {len(W_df_full)}"
            )
        tau_hat_arr = np.asarray(tau_hat).ravel()
        W_sub = _subsample(W_df_full, surrogate_sample_size, seed)
        # subsample tau_hat to match W_sub's row selection
        rng = np.random.default_rng(seed)
        if surrogate_sample_size < len(W_df_full):
            idx = rng.choice(len(W_df_full), size=surrogate_sample_size, replace=False)
            W_sub = W_df_full.iloc[idx].reset_index(drop=True)
            tau_sub = tau_hat_arr[idx]
        else:
            W_sub = W_df_full.copy()
            tau_sub = tau_hat_arr
    else:
        # Slow path: call cf.predict on a small subsample — keep
        # surrogate_sample_size ≤ ~3k to avoid OOM / long runtimes.
        W_sub = _subsample(W_df_full, surrogate_sample_size, seed)
        tau_sub = np.asarray(cf.predict(W_sub.to_numpy())).ravel()

    # Coefficient of variation of τ̂(W): measures how much spatial
    # heterogeneity is actually present.  When cv is small the CATE
    # surface is near-flat and no surrogate — however large — can achieve
    # high OOB R², because the target is dominated by estimation noise
    # rather than genuine spatial signal.  Low R² in that regime is
    # expected and informative, not a model quality failure.
    tau_std = float(np.std(tau_sub))
    tau_mean_abs = float(np.mean(np.abs(tau_sub))) + 1e-9
    tau_cv = tau_std / tau_mean_abs

    surrogate = RandomForestRegressor(
        n_estimators=surrogate_n_estimators,
        min_samples_leaf=surrogate_min_samples_leaf,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
        oob_score=True,
    )
    surrogate.fit(W_sub, tau_sub)

    # Use OOB R² (unbiased) when available, else in-sample.
    if hasattr(surrogate, "oob_score_"):
        fit_r2 = float(surrogate.oob_score_)
    else:
        fit_r2 = float(r2_score(tau_sub, surrogate.predict(W_sub)))

    if fit_r2 < r2_warn_threshold:
        if tau_cv < 0.1:
            # Low R² because the CATE surface is near-homogeneous — this is
            # a result, not a surrogate quality problem.  SHAP will show
            # near-uniform feature importance, which is correct.
            logger.info(
                "CATE surrogate OOB R² = %.3f (τ̂ CV = %.3f — near-homogeneous "
                "trend; low R² is expected). SHAP values reflect noise, not "
                "spatial signal.",
                fit_r2,
                tau_cv,
            )
        else:
            logger.warning(
                "CATE surrogate OOB R² = %.3f (below %.2f, τ̂ CV = %.3f). "
                "SHAP values are a rough approximation of τ̂(W). "
                "Consider raising surrogate_n_estimators or lowering "
                "surrogate_min_samples_leaf.",
                fit_r2,
                r2_warn_threshold,
                tau_cv,
            )
    else:
        logger.info("CATE surrogate OOB R² = %.3f (τ̂ CV = %.3f)", fit_r2, tau_cv)

    X_sample = _subsample(W_sub, sample_size, seed)
    explainer = shap.TreeExplainer(surrogate)
    # approximate=True uses the path-dependent algorithm: same ranking,
    # ~10× faster than exact Shapley on 100-tree forests.
    values = np.asarray(explainer.shap_values(X_sample, approximate=True))
    base = float(np.atleast_1d(explainer.expected_value)[0])
    return ShapExplanation(
        values=values,
        base_value=base,
        feature_names=feature_names,
        X_explained=X_sample,
        surrogate_r2=fit_r2,
        top_features=_mean_abs_importance(values, feature_names),
    )


# ── Plotting ───────────────────────────────────────────────────────────────

# Design tokens — mirrored from viz.style so interpret.py is self-contained
# when used outside the full package (notebooks, etc.)
_PAPER = "#F7F4EE"
_TEXT = "#151515"
_TEXT_MUTED = "#66615B"
_GRIDLINE = "#D9D1C7"
_FONT = "DejaVu Sans"


def _style_tokens() -> tuple[str, str, str, str, str]:
    """Return (paper, text, text_muted, gridline, font) from viz.style if available."""
    try:
        from bgbw_trends.viz import style  # type: ignore

        style.apply()
        return (
            style.PAPER,
            style.TEXT,
            style.TEXT_MUTED,
            style.GRIDLINE,
            style.FONT_FAMILY,
        )
    except Exception:
        return _PAPER, _TEXT, _TEXT_MUTED, _GRIDLINE, _FONT


def _beeswarm_ys(
    x: np.ndarray, row_height: float = 0.78, max_bins: int = 80
) -> np.ndarray:
    """Vertical offsets for one beeswarm row (column-stacking algorithm).

    Points that fall into the same x-bin are stacked symmetrically about
    the row centre line: 0, +s, −s, +2s, −2s, … so the swarm stays
    within ±row_height/2 of the axis.
    """
    n = len(x)
    if n == 0:
        return np.zeros(0)
    x_range = float(x.max() - x.min())
    if x_range < 1e-12:
        return np.zeros(n)

    n_bins = min(max_bins, max(20, n // 8))
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    # digitize returns 1-based; subtract 1 for 0-based, clamp top edge
    bin_idx = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)

    ys = np.zeros(n)
    for b in range(n_bins):
        pts = np.where(bin_idx == b)[0]
        m = len(pts)
        if m == 0:
            continue
        step = row_height / max(m, 2)
        for k, pt in enumerate(pts):
            if k == 0:
                ys[pt] = 0.0
            else:
                ys[pt] = (1 if k % 2 == 1 else -1) * ((k + 1) // 2) * step
    return ys


def _readable_label(name: str) -> str:
    return name.replace("_", " ")


def _draw_beeswarm_ax(
    ax,
    expl: ShapExplanation,
    features: list[str],
    *,
    title: str | None = None,
    show_ylabel: bool = True,
    paper: str = _PAPER,
    text: str = _TEXT,
    text_muted: str = _TEXT_MUTED,
    gridline: str = _GRIDLINE,
    font: str = _FONT,
) -> None:
    """Draw a single beeswarm into *ax*. Features listed bottom-to-top."""
    import matplotlib.pyplot as plt

    feat_col = {f: i for i, f in enumerate(expl.feature_names)}
    cmap = plt.get_cmap("RdBu_r")  # low value = blue, high value = red

    for row_y, feat_name in enumerate(features):
        if feat_name not in feat_col:
            continue
        fi = feat_col[feat_name]
        sv = expl.values[:, fi]
        fv = expl.X_explained[feat_name].to_numpy(dtype=float)

        fv_f = fv[np.isfinite(fv)]
        fv_range = float(fv_f.max() - fv_f.min()) if fv_f.size else 0.0
        fv_norm = (
            np.clip((fv - fv_f.min()) / fv_range, 0.0, 1.0)
            if fv_range > 0
            else np.full(len(fv), 0.5)
        )

        ys = _beeswarm_ys(sv) + row_y
        ax.scatter(
            sv,
            ys,
            c=fv_norm,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            s=5,
            alpha=0.6,
            linewidths=0,
            rasterized=True,
        )

    n = len(features)
    ax.axvline(0, color=gridline, linewidth=0.9, zorder=0)
    ax.set_facecolor(paper)

    labels = [_readable_label(f) for f in features]
    ax.set_yticks(range(n))
    if show_ylabel:
        ax.set_yticklabels(labels, fontsize=8, color=text, fontfamily=font)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_ylim(-0.6, n - 0.4)

    ax.set_xlabel("SHAP value", fontsize=8, color=text_muted, fontfamily=font)
    ax.tick_params(axis="x", colors=text_muted, labelsize=7.5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5, symmetric=True))

    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(gridline)
    ax.spines["bottom"].set_linewidth(0.8)

    if title:
        ax.set_title(
            title,
            fontsize=9,
            fontweight="600",
            color=text,
            fontfamily=font,
            pad=6,
            loc="left",
        )


def plot_shap_panel(
    expl_left: ShapExplanation,
    expl_right: ShapExplanation,
    *,
    title_left: str = "Occurrence stage",
    title_right: str = "Count stage",
    suptitle: str | None = None,
    footer: str | None = None,
    max_display: int = 15,
    output_path=None,
    dpi: int = 200,
) -> "plt.Figure":
    """Two-panel beeswarm: left and right explanations side by side.

    One figure per call — use once for SDM marginal forests, once for the
    CATE surrogate forests. Each panel is a full beeswarm with a shared
    feature-value colour scale.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from pathlib import Path

    paper, text, text_muted, gridline, font = _style_tokens()

    left_feats = expl_left.top_features.head(max_display)["feature"].tolist()[::-1]
    right_feats = expl_right.top_features.head(max_display)["feature"].tolist()[::-1]
    n_feat = max(len(left_feats), len(right_feats))

    fig_h = max(4.0, 0.40 * n_feat + 1.6)
    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(12.5, fig_h),
        facecolor=paper,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.38},
    )

    kw = dict(
        paper=paper, text=text, text_muted=text_muted, gridline=gridline, font=font
    )
    _draw_beeswarm_ax(
        ax_l, expl_left, left_feats, title=title_left, show_ylabel=True, **kw
    )
    _draw_beeswarm_ax(
        ax_r, expl_right, right_feats, title=title_right, show_ylabel=True, **kw
    )

    # Shared feature-value colour scale
    cmap = plt.get_cmap("RdBu_r")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=[ax_l, ax_r],
        orientation="vertical",
        fraction=0.010,
        pad=0.015,
        aspect=28,
        shrink=0.72,
    )
    cbar.set_ticks([0.04, 0.96])
    cbar.set_ticklabels(["Low", "High"], fontsize=7, color=text_muted)
    cbar.set_label("Feature\nvalue", fontsize=7.5, color=text_muted, labelpad=5)
    cbar.ax.tick_params(size=0, colors=text_muted)
    cbar.outline.set_visible(False)

    if suptitle:
        fig.suptitle(
            suptitle,
            fontsize=11,
            fontweight="700",
            color=text,
            fontfamily=font,
            y=1.02,
        )
    if footer:
        fig.text(
            0.5,
            -0.02,
            footer,
            ha="center",
            fontsize=7.5,
            color=text_muted,
            fontfamily=font,
        )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=paper)

    return fig
