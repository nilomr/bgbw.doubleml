"""Plotting utilities for DML trend results."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bgbw_trends.pipeline import SpeciesPrediction

logger = logging.getLogger(__name__)

# ── BNG raster helper ────────────────────────────────────────────────────────
_BNG_CELL_M = 1_000  # 1 km BNG grid


def _bng_to_scalar_raster(
    eastings: np.ndarray,
    northings: np.ndarray,
    values: np.ndarray,
    cell_m: int = _BNG_CELL_M,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Rasterise scalar values (keyed by SW-corner easting/northing) to a 2-D array.

    Parameters
    ----------
    eastings, northings : np.ndarray
        SW-corner BNG coordinates in metres (integer multiples of cell_m).
    values : np.ndarray
        Scalar value per cell.
    cell_m : int
        Cell side length in metres (1000 for 1 km BNG).

    Returns
    -------
    raster : (n_rows, n_cols) float32, NaN where no data
    extent : (e_min, e_max, n_min, n_max) — imshow extent in metres
    """
    sw_e = eastings.astype(np.int64)
    sw_n = northings.astype(np.int64)

    e_min, e_max = int(sw_e.min()), int(sw_e.max()) + cell_m
    n_min, n_max = int(sw_n.min()), int(sw_n.max()) + cell_m

    n_cols = (e_max - e_min) // cell_m
    n_rows = (n_max - n_min) // cell_m

    raster = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    col_idx = (sw_e - e_min) // cell_m
    row_idx = (sw_n - n_min) // cell_m
    raster[row_idx, col_idx] = values.astype(np.float32)

    return raster, (float(e_min), float(e_max), float(n_min), float(n_max))


_Z_80 = 1.2816  # one-sided 90th percentile of N(0,1) ⇒ 80% CI half-width


def plot_trend_map(
    prediction: "SpeciesPrediction",
    metric: str = "abd_ppy",
    output_path: Path | None = None,
) -> plt.Figure:
    """Map of spatially varying trend from a ``SpeciesPrediction`` grid.

    Plots the chosen per-year rate (``occ_ppy``, ``count_ppy`` or
    ``abd_ppy``) and flags 80% CI significance. The abundance rate has no
    directly attached variance (it combines two causal forests), so
    significance there is reported only for metrics with ``*_sigma``
    available on the prediction object.
    """
    # Stored trend values are ratios per year; convert to percent-per-year
    # for visualization to match terminal summaries and viz/maps.py.
    tau = np.asarray(getattr(prediction, metric), dtype=np.float64).ravel() * 100.0
    sigma_attr = {"occ_ppy": "occ_sigma", "count_ppy": "count_sigma"}.get(metric)
    sigma = (
        np.asarray(getattr(prediction, sigma_attr), dtype=np.float64).ravel() * 100.0
        if sigma_attr is not None
        else None
    )

    df = pd.DataFrame({"bng_ref": prediction.bng_ref, "tau": tau})
    coords = df["bng_ref"].apply(_parse_bng_approx)
    df["easting"] = coords.apply(lambda c: c[0])
    df["northing"] = coords.apply(lambda c: c[1])
    df = df.dropna(subset=["easting", "northing"])
    east = df["easting"].to_numpy()
    north = df["northing"].to_numpy()
    tau = df["tau"].to_numpy()

    if sigma is not None:
        sigma = sigma[df.index.to_numpy()]
        significant = (np.abs(tau) - _Z_80 * sigma) > 0
    else:
        significant = np.ones_like(tau, dtype=bool)

    vmax = max(abs(np.nanquantile(tau, 0.05)), abs(np.nanquantile(tau, 0.95)))
    vmax = max(vmax, 1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu")

    raster_all, extent = _bng_to_scalar_raster(east, north, tau)
    tau_sig = tau.copy()
    tau_sig[~significant] = np.nan
    raster_sig, _ = _bng_to_scalar_raster(east, north, tau_sig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    e_min, e_max, n_min, n_max = extent

    panels = (
        (axes[0], raster_all, "all trends"),
        (axes[1], raster_sig, "significant (80% CI)"),
    )
    for ax, raster, title_suffix in panels:
        ax.set_facecolor("#cccccc")
        rgba = cmap(norm(np.nan_to_num(raster, nan=0.0)))
        rgba[np.isnan(raster), 3] = 0.0
        ax.imshow(
            rgba,
            origin="lower",
            extent=(e_min, e_max, n_min, n_max),
            interpolation="nearest",
            aspect="equal",
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="annual change (%/year)", shrink=0.7)
        ax.set_title(f"{prediction.species} — {title_suffix}")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")

    fig.suptitle(f"Spatially varying {metric}: {prediction.species}", fontsize=13)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved trend map to %s", output_path)
    return fig


def plot_sdm_prediction_maps(
    prediction: "SpeciesPrediction",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot effort-adjusted SDM maps for occurrence and abundance as 1 km BNG rasters."""
    df = pd.DataFrame(
        {
            "bng_ref": prediction.bng_ref,
            "p_occurrence": prediction.p_occurrence,
            "e_abundance": prediction.e_abundance,
        }
    )

    coords = df["bng_ref"].apply(_parse_bng_approx)
    df["easting"] = coords.apply(lambda c: c[0])
    df["northing"] = coords.apply(lambda c: c[1])
    df = df.dropna(subset=["easting", "northing"])

    east = df["easting"].to_numpy()
    north = df["northing"].to_numpy()

    raster_occ, extent = _bng_to_scalar_raster(
        east, north, df["p_occurrence"].to_numpy()
    )
    e_abund = df["e_abundance"].to_numpy()
    emax = float(np.nanquantile(e_abund, 0.99)) if len(e_abund) else 1.0
    raster_abund, _ = _bng_to_scalar_raster(east, north, e_abund)
    e_min, e_max, n_min, n_max = extent

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Occurrence
    norm_occ = mcolors.Normalize(vmin=0, vmax=1)
    cmap_occ = plt.get_cmap("viridis")
    rgba_occ = cmap_occ(norm_occ(np.nan_to_num(raster_occ, nan=0.0)))
    rgba_occ[np.isnan(raster_occ), 3] = 0.0
    axes[0].imshow(
        rgba_occ,
        origin="lower",
        extent=(e_min, e_max, n_min, n_max),
        interpolation="nearest",
        aspect="equal",
    )
    sm0 = plt.cm.ScalarMappable(cmap=cmap_occ, norm=norm_occ)
    sm0.set_array([])
    plt.colorbar(sm0, ax=axes[0], label="P(occurrence)", shrink=0.7)
    axes[0].set_title(f"{prediction.species} — distribution P(occurrence)")
    axes[0].set_xlabel("Easting (m)")
    axes[0].set_ylabel("Northing (m)")

    # Abundance
    norm_abund = mcolors.Normalize(vmin=0, vmax=max(emax, 1e-6))
    cmap_abund = plt.get_cmap("magma")
    rgba_abund = cmap_abund(norm_abund(np.nan_to_num(raster_abund, nan=0.0)))
    rgba_abund[np.isnan(raster_abund), 3] = 0.0
    axes[1].imshow(
        rgba_abund,
        origin="lower",
        extent=(e_min, e_max, n_min, n_max),
        interpolation="nearest",
        aspect="equal",
    )
    sm1 = plt.cm.ScalarMappable(cmap=cmap_abund, norm=norm_abund)
    sm1.set_array([])
    plt.colorbar(sm1, ax=axes[1], label="E[count]", shrink=0.7)
    axes[1].set_title(f"{prediction.species} — relative abundance")
    axes[1].set_xlabel("Easting (m)")

    fig.suptitle(f"Effort-adjusted SDM predictions: {prediction.species}", fontsize=13)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved SDM prediction maps to %s", output_path)
    return fig


# ── BNG grid reference parsing ──────────────────────────────────────────────

_BNG_LETTERS = {
    "S": (0, 0),
    "T": (5, 0),
    "N": (0, 5),
    "O": (5, 5),
    "H": (0, 10),
}

_BNG_SECOND = {
    "A": (0, 4),
    "B": (1, 4),
    "C": (2, 4),
    "D": (3, 4),
    "E": (4, 4),
    "F": (0, 3),
    "G": (1, 3),
    "H": (2, 3),
    "J": (3, 3),
    "K": (4, 3),
    "L": (0, 2),
    "M": (1, 2),
    "N": (2, 2),
    "O": (3, 2),
    "P": (4, 2),
    "Q": (0, 1),
    "R": (1, 1),
    "S": (2, 1),
    "T": (3, 1),
    "U": (4, 1),
    "V": (0, 0),
    "W": (1, 0),
    "X": (2, 0),
    "Y": (3, 0),
    "Z": (4, 0),
}


def _yearly_trend_summary(obs: pd.DataFrame, ate: float) -> pd.DataFrame:
    """Per-year summary for the deconfounded-trend plot.

    Columns:
    - ``raw_mean``, ``raw_se``: observed yearly mean of Y and its SE
      (binomial for binary ``raw``, sample SEM otherwise).
    - ``cond_mean``: yearly mean of ``m̂(X_i)``, the conditional mean
      model evaluated at each year's actual X rows. This series tracks
      the component of the raw yearly change that is attributable to
      X-drift across years (i.e. confounding as captured by X).
    - ``deconf``: ``Ȳ + ATE·(year − T̄)``. The pure-trend line the
      DML model claims after partialling out X — the direct analogue
      of the "true trend" orange line in Fink's motivating-example.Rmd
      fig1a, but with ``ATE`` substituted for the unknown true slope.
    """
    g = obs.groupby("year", sort=True)
    years = np.asarray(sorted(obs["year"].unique()), dtype=int)
    raw_mean = g["raw"].mean().reindex(years).to_numpy()
    n = g["raw"].count().reindex(years).to_numpy()

    is_binary = set(np.unique(obs["raw"].to_numpy())).issubset({0.0, 1.0})
    if is_binary:
        raw_se = np.sqrt(np.maximum(raw_mean * (1 - raw_mean) / np.maximum(n, 1), 0))
    else:
        raw_se = g["raw"].sem().reindex(years).to_numpy()

    cond_mean = g["m_hat"].mean().reindex(years).to_numpy()

    y_bar = float(obs["raw"].mean())
    t_bar = float(obs["year"].mean())
    deconf = y_bar + ate * (years.astype(float) - t_bar)

    return pd.DataFrame(
        {
            "year": years,
            "n": n,
            "raw_mean": raw_mean,
            "raw_se": raw_se,
            "cond_mean": cond_mean,
            "deconf": deconf,
        }
    )


def plot_yearly_trend(
    prediction: "SpeciesPrediction",
    output_path: Path | None = None,
) -> plt.Figure:
    """Raw vs deconfounded model trend, one panel per stage.

    For each stage (occurrence, count) we plot:

    - **Raw** (blue): observed yearly mean of Y with SE bars. Confounded —
      this is what the data shows before any adjustment.
    - **Conditional mean** (grey, dashed): yearly mean of ``m̂(X_i)`` for
      observations in year y. Captures how the *confounders* drift year to
      year; the gap between raw and this line is the year effect *not*
      explained by X (approximately).
    - **Deconfounded trend** (orange): ``Ȳ + ATE·(y − T̄)`` where
      ``ATE = mean τ̂(W)``. The model's answer to "what is the trend after
      partialling out X?" — direct analogue of the true-trend line in
      motivating-example.Rmd fig1a.

    Occurrence is on the probability scale; count is on log1p.
    """
    if prediction.obs_occ.empty or prediction.obs_count.empty:
        raise ValueError(
            "SpeciesPrediction has no per-observation data. "
            "Run train_predict_species to populate obs_occ / obs_count."
        )

    summary_occ = _yearly_trend_summary(prediction.obs_occ, prediction.ate_occ)
    summary_count = _yearly_trend_summary(prediction.obs_count, prediction.ate_count)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panels = [
        (axes[0], summary_occ, prediction.ate_occ, "Occurrence", "P(occurrence)", True),
        (
            axes[1],
            summary_count,
            prediction.ate_count,
            "Count (log1p)",
            "log(1 + count)",
            False,
        ),
    ]
    for ax, s, ate, title, ylabel, is_binary in panels:
        years = s["year"].to_numpy()
        ax.errorbar(
            years,
            s["raw_mean"],
            yerr=s["raw_se"],
            fmt="o",
            color="steelblue",
            capsize=4,
            label="raw",
            zorder=3,
        )
        ax.plot(
            years,
            s["cond_mean"],
            "--o",
            color="grey",
            markersize=4,
            linewidth=1.0,
            alpha=0.8,
            label="conditional mean  m̂(X)",
            zorder=2,
        )
        ax.plot(
            years,
            s["deconf"],
            "-",
            color="darkorange",
            linewidth=2.2,
            label=f"deconfounded trend (ATE={ate:+.4f}/yr)",
            zorder=4,
        )
        if is_binary:
            ax.set_ylim(0, 1)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(years)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"Yearly trend — raw vs deconfounded: {prediction.species}", fontsize=13
    )
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved yearly trend plot to %s", output_path)
    return fig


def _reliability(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reliability curve via equal-mass (quantile) bins.

    Equal-mass bins are more honest than equal-width: each bucket
    contains a similar number of observations, so the observed
    frequency per bucket has comparable sampling error.
    """
    order = np.argsort(y_prob)
    p_sorted = y_prob[order]
    y_sorted = y_true[order]
    splits = np.array_split(np.arange(len(p_sorted)), n_bins)
    pmean = np.array([p_sorted[s].mean() for s in splits if len(s)])
    ymean = np.array([y_sorted[s].mean() for s in splits if len(s)])
    n = np.array([len(s) for s in splits if len(s)])
    return pmean, ymean, n


def plot_raw_vs_fitted(
    prediction: "SpeciesPrediction",
    output_path: Path | None = None,
    n_occ_bins: int = 12,
) -> plt.Figure:
    """Model-fit diagnostics: calibration for occurrence, scatter for count.

    The "fitted" value used here is the PLR reconstruction at each
    training row: ``Ŷ_i = m̂(X_i) + τ̂(W_i)·(T_i − ŝ(X_i))``. Binary
    stage fitted probabilities are clipped to ``[0, 1]`` only for
    display of the reliability diagram.

    - **Occurrence (left)**: equal-mass reliability diagram. Points on
      the diagonal mean the fitted probabilities match observed
      frequencies. Deviations indicate miscalibration of the combined
      nuisance + CATE fit.
    - **Count (right)**: raw vs fitted log1p, one dot per detected
      observation, coloured by year to reveal systematic year drift in
      the fit. The diagonal is a perfect fit.
    """
    if prediction.obs_occ.empty or prediction.obs_count.empty:
        raise ValueError(
            "SpeciesPrediction has no per-observation data. "
            "Run train_predict_species to populate obs_occ / obs_count."
        )

    obs_occ = prediction.obs_occ
    obs_count = prediction.obs_count

    fitted_occ = np.clip(
        obs_occ["m_hat"].to_numpy()
        + obs_occ["tau"].to_numpy()
        * (obs_occ["year"].to_numpy() - obs_occ["s_hat"].to_numpy()),
        0.0,
        1.0,
    )
    fitted_count = obs_count["m_hat"].to_numpy() + obs_count["tau"].to_numpy() * (
        obs_count["year"].to_numpy() - obs_count["s_hat"].to_numpy()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Occurrence reliability ─────────────────────────────────────────
    ax = axes[0]
    pmean, ymean, n_per = _reliability(
        obs_occ["raw"].to_numpy(), fitted_occ, n_occ_bins
    )
    sizes = 20 + 180 * (n_per / n_per.max())
    ax.scatter(
        pmean,
        ymean,
        s=sizes,
        color="steelblue",
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Fitted P(occurrence)")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title(
        f"Occurrence — reliability (Brier={prediction.diagnostics.occurrence_brier:.3f})"
    )

    # ── Count scatter coloured by year ─────────────────────────────────
    ax = axes[1]
    years = np.asarray(sorted(obs_count["year"].unique()))
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=years.min(), vmax=years.max())
    ax.scatter(
        fitted_count,
        obs_count["raw"].to_numpy(),
        c=obs_count["year"].to_numpy(),
        cmap=cmap,
        norm=norm,
        alpha=0.35,
        s=8,
        linewidth=0,
    )
    lo = float(min(obs_count["raw"].min(), fitted_count.min()))
    hi = float(max(obs_count["raw"].max(), fitted_count.max()))
    ax.plot([lo, hi], [lo, hi], "--", color="grey", linewidth=0.8)
    ax.set_xlabel("Fitted log(1 + count)")
    ax.set_ylabel("Raw log(1 + count)")
    ax.set_title(f"Count — raw vs fitted (R²={prediction.diagnostics.count_r2:.3f})")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Year", shrink=0.8)

    fig.suptitle(f"Model-fit diagnostics: {prediction.species}", fontsize=13)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved raw-vs-fitted plot to %s", output_path)
    return fig


# ── BNG grid reference parsing ──────────────────────────────────────────────


def _parse_bng_approx(ref: str) -> tuple[float, float]:
    """Parse a BNG grid reference to approximate easting/northing (metres)."""
    try:
        first, second = ref[0], ref[1]
        digits = ref[2:]
        half = len(digits) // 2
        e_digits, n_digits = digits[:half], digits[half:]

        base_e, base_n = _BNG_LETTERS.get(first, (0, 0))
        sq_e, sq_n = _BNG_SECOND.get(second, (0, 0))

        easting = (base_e + sq_e) * 100_000 + int(e_digits) * 10 ** (5 - half)
        northing = (base_n + sq_n) * 100_000 + int(n_digits) * 10 ** (5 - half)
        return (easting, northing)
    except Exception:
        return (np.nan, np.nan)
