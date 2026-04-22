"""End-to-end recovery test: known-truth BGBW-like simulation.

Mirrors the DML trend paradigm of Fink et al. (2023) (SI §3,
``motivating-example.Rmd``) which itself is applied to *aggregated grid
cells* (27 km × 1 week) with a hurdle response — i.e. the same paradigm
we use for BGBW, but at a different resolution.

BGBW-specific choices in this simulation:

* 1 km × year aggregation (vs Fink's 27 km × 1 week),
* BGBW effort covariates (``n_surveys``, ``proportion_online``,
  ``mean_observers``, ``proportion_adult``) in place of Fink's CCI /
  duration / distance / time-of-day,
* year-drifting effort covariates so year is causally entangled with
  effort the way it is in the real data — the confounding the DML step
  has to remove.

The simulated generating process:

    emb ~ N(0, 1)                                      (cell-stable habitat)
    proportion_online_t = σ(-1 + 0.25·(t - T̄)) + noise  (increases over years)
    n_surveys_t ~ 1 + Poisson(max(0.5 + 0.1·(t - T̄), 0.1))
    η_occ   = a0 + a1·emb_0 + g·prop_online + h·(n_surveys - 2) + τ_occ·(t - T̄)
    P(occ)  = σ(η_occ);   occ ~ Bern(P(occ))
    η_count = b0 + b1·emb_0 + k·prop_online + τ_count·(t - T̄)
    count   = occ · expm1(η_count + N(0, σ²))

So the *true* per-year rates are ``τ_occ`` (absolute probability change)
and ``τ_count`` (log-count slope). We assert the pipeline recovers them
when effort is included in X, and that dropping effort from X leaves
residual confounding — the point of the DML step.

Kept small enough (≈3200 rows, trimmed forests) to run in CI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from bgbw_trends.config import RunConfig
from bgbw_trends.features.effort import compute_reference_effort
from bgbw_trends.pipeline import train_predict_species

# Known-truth parameters used by _simulate and asserted against.
_TAU_OCC_TRUE = -0.01
_TAU_COUNT_TRUE = 0.02

_N_EMB = 8
_YEARS = tuple(range(2016, 2024))  # 8 years, centred at 2019.5
_N_CELLS = 400


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _simulate_bgbw_cell_year(
    n_cells: int = _N_CELLS,
    years: tuple[int, ...] = _YEARS,
    n_emb: int = _N_EMB,
    tau_occ: float = _TAU_OCC_TRUE,
    tau_count: float = _TAU_COUNT_TRUE,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    year_arr = np.asarray(years, dtype=float)
    tbar = year_arr.mean()

    # Stable per-cell habitat (embeddings + coords)
    emb = rng.normal(0.0, 1.0, size=(n_cells, n_emb))
    lat = 51.0 + rng.uniform(0.0, 5.0, size=n_cells)
    lon = -4.0 + rng.uniform(0.0, 5.0, size=n_cells)

    frames: list[pd.DataFrame] = []
    for t in year_arr:
        t_c = float(t - tbar)

        # Effort drift: proportion_online grows; n_surveys intensifies.
        # Both are genuine confounders — they push year and response
        # simultaneously, so omitting them leaks into τ̂.
        p_online_mean = _sigmoid(-1.0 + 0.25 * t_c)
        p_online = np.clip(
            p_online_mean + rng.normal(0.0, 0.05, size=n_cells), 0.0, 1.0
        )
        nsurv_rate = max(0.5 + 0.1 * t_c, 0.1)
        n_surveys = 1 + rng.poisson(nsurv_rate, size=n_cells)
        mean_observers = rng.uniform(1.0, 2.0, size=n_cells)
        proportion_adult = rng.uniform(0.7, 1.0, size=n_cells)

        eta_occ = (
            -0.8
            + 1.2 * emb[:, 0]
            + 1.5 * p_online
            + 0.10 * (n_surveys - 2.0)
            + tau_occ * t_c
        )
        p_occ = _sigmoid(eta_occ)
        occ = rng.binomial(1, p_occ)

        eta_count = 0.8 + 0.6 * emb[:, 0] + 0.5 * p_online + tau_count * t_c
        log_count = eta_count + rng.normal(0.0, 0.5, size=n_cells)
        count = np.where(occ > 0, np.maximum(np.expm1(log_count), 0.0), 0.0)

        frames.append(
            pd.DataFrame(
                {
                    "bng_ref": [f"c{c:04d}" for c in range(n_cells)],
                    "year": int(t),
                    "blue_tit": count,
                    "n_surveys": n_surveys.astype(float),
                    "mean_observers": mean_observers,
                    "proportion_adult": proportion_adult,
                    "proportion_online": p_online,
                    "latitude": lat,
                    "longitude": lon,
                    **{f"emb_{i}": emb[:, i] for i in range(n_emb)},
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _grid_from_df(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values("year")
        .drop_duplicates("bng_ref", keep="last")
        .reset_index(drop=True)
    )


def _run_pipeline(
    df: pd.DataFrame,
    *,
    propensity_covariates=None,
    marginal_covariates=None,
    seed: int = 0,
):
    grid = _grid_from_df(df)
    cfg = RunConfig(
        species=["blue_tit"],
        years=[int(df["year"].min()), int(df["year"].max())],
        n_estimators=100,
        cf_n_estimators=400,
        cf_min_samples_leaf=10,
        n_folds=3,
        n_rep=1,
        n_pca_components=0,
    )
    return train_predict_species(
        df,
        df,
        species="blue_tit",
        cfg=cfg,
        grid=grid,
        reference_effort=compute_reference_effort(df),
        propensity_covariates=propensity_covariates,
        marginal_covariates=marginal_covariates,
        seed=seed,
    )


# ── Fixtures (module-scoped — pipeline is the slow part) ─────────────────


@pytest.fixture(scope="module")
def sim_df() -> pd.DataFrame:
    return _simulate_bgbw_cell_year(seed=0)


@pytest.fixture(scope="module")
def pred_full(sim_df):
    """Pipeline with all X (habitat + effort) — the correctly specified case."""
    return _run_pipeline(sim_df, seed=0)


@pytest.fixture(scope="module")
def pred_hab_only(sim_df):
    """Pipeline with habitat-only X — effort confounders omitted on purpose."""
    hab_only = [f"emb_{i}" for i in range(_N_EMB)] + ["latitude", "longitude"]
    return _run_pipeline(
        sim_df,
        propensity_covariates=hab_only,
        marginal_covariates=hab_only,
        seed=0,
    )


# ── Tests ────────────────────────────────────────────────────────────────


class TestTrendRecovery:
    """Under correct specification the ATE should approximate the simulated truth."""

    def test_ate_occ_close_to_truth(self, pred_full):
        err = pred_full.ate_occ - _TAU_OCC_TRUE
        assert abs(err) < 0.02, (
            f"ate_occ={pred_full.ate_occ:+.4f} true={_TAU_OCC_TRUE:+.4f} err={err:+.4f}"
        )

    def test_ate_count_close_to_truth(self, pred_full):
        err = pred_full.ate_count - _TAU_COUNT_TRUE
        assert abs(err) < 0.02, (
            f"ate_count={pred_full.ate_count:+.4f} true={_TAU_COUNT_TRUE:+.4f} "
            f"err={err:+.4f}"
        )

    def test_chain_rule_mean_consistent(self, pred_full):
        # abd_ppy = τ_occ/P(occ) + τ_count at each grid cell (Fink L197).
        # Averaged it should match the same combination at the mean P(occ).
        mean_abd = float(np.nanmean(pred_full.abd_ppy))
        mean_p = float(np.nanmean(pred_full.p_occurrence))
        expected = _TAU_OCC_TRUE / mean_p + _TAU_COUNT_TRUE
        assert abs(mean_abd - expected) < 0.05, (
            f"mean abd_ppy={mean_abd:+.4f} expected≈{expected:+.4f}"
        )

    def test_sdm_tracks_driver_embedding(self, sim_df, pred_full):
        # emb_0 drives both stages positively in the DGP; the SDM's
        # P(occurrence) at the grid should correlate strongly with it.
        grid = _grid_from_df(sim_df)
        corr = float(
            np.corrcoef(pred_full.p_occurrence, grid["emb_0"].to_numpy())[0, 1]
        )
        assert corr > 0.5, f"corr(p_occ, emb_0)={corr:.2f} (expected > 0.5)"


class TestEffortIsLoadBearing:
    """Direct BGBW analogue of vignette §2: omitting a real confounder
    should bias τ̂. This is the 'without_ps' arm in motivating-example.Rmd —
    if our pipeline is responsive to the X spec, dropping effort from X must
    leak year-correlated effort drift into τ_occ, inflating it above the
    correctly-specified estimate.
    """

    def test_occ_bias_larger_without_effort(self, pred_full, pred_hab_only):
        # True τ_occ<0; rising online participation masks the decline, so
        # habitat-only τ̂_occ should be larger (less negative, possibly positive)
        # than the fully-adjusted one.
        gap = pred_hab_only.ate_occ - pred_full.ate_occ
        assert gap > 0.005, (
            f"ate_occ hab_only={pred_hab_only.ate_occ:+.4f} "
            f"full={pred_full.ate_occ:+.4f} gap={gap:+.4f} "
            "(expected positive — effort is a real confounder here)"
        )

    def test_full_spec_closer_to_truth_than_hab_only_occ(
        self, pred_full, pred_hab_only
    ):
        err_full = abs(pred_full.ate_occ - _TAU_OCC_TRUE)
        err_hab = abs(pred_hab_only.ate_occ - _TAU_OCC_TRUE)
        assert err_full < err_hab, (
            f"|err_full|={err_full:.4f} not smaller than |err_hab_only|={err_hab:.4f}"
        )


# ── Sanity-check plot ─────────────────────────────────────────────────────


def test_simulation_sanity_plot(pred_full, pred_hab_only, tmp_path):
    """Save a vignette-style fig1a analogue for visual inspection.

    Two panels:
    - Left:  per-year mean occurrence (raw) + deconfounded trend lines
             for full and hab-only specs, plus the true τ_occ.
    - Right: per-year mean log1p-count (detected rows) + trend lines
             for the full spec, plus the true τ_count.

    Written to ``output/prototype/simulation_sanity_check.png``.
    """
    out_dir = Path(__file__).parents[1] / "output" / "prototype"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "simulation_sanity_check.png"

    year_center = np.mean(_YEARS)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Simulation sanity check — vignette fig1a analogue", fontsize=12)

    # ── left: occurrence ──────────────────────────────────────────────────
    ax = axes[0]
    occ = pred_full.obs_occ.copy()
    occ_by_year = occ.groupby("year")["raw"].agg(["mean", "sem"]).reset_index()

    ax.errorbar(
        occ_by_year["year"],
        occ_by_year["mean"],
        yerr=occ_by_year["sem"],
        fmt="o",
        color="#555555",
        ecolor="#aaaaaa",
        capsize=3,
        label="per-year mean (raw)",
        zorder=3,
    )

    grand_mean_occ = float(occ["raw"].mean())
    years_plot = np.array(sorted(occ["year"].unique()), dtype=float)

    def _trend_line(ate: float) -> np.ndarray:
        return grand_mean_occ + ate * (years_plot - year_center)

    ax.plot(
        years_plot,
        grand_mean_occ + _TAU_OCC_TRUE * (years_plot - year_center),
        color="#ff8c00",
        linewidth=2.5,
        label=f"true τ_occ = {_TAU_OCC_TRUE:+.3f}",
    )
    ax.plot(
        years_plot,
        _trend_line(pred_full.ate_occ),
        color="#00008b",
        linewidth=2,
        linestyle="--",
        label=f"full: τ̂ = {pred_full.ate_occ:+.4f}",
    )
    ax.plot(
        years_plot,
        _trend_line(pred_hab_only.ate_occ),
        color="#7f00ff",
        linewidth=2,
        linestyle=":",
        label=f"hab-only: τ̂ = {pred_hab_only.ate_occ:+.4f}",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean occurrence (raw)")
    ax.set_title("Occurrence stage")
    ax.legend(fontsize=8)

    # ── right: log-count (detected rows only) ─────────────────────────────
    ax = axes[1]
    cnt = pred_full.obs_count.copy()
    cnt_by_year = cnt.groupby("year")["raw"].agg(["mean", "sem"]).reset_index()

    ax.errorbar(
        cnt_by_year["year"],
        cnt_by_year["mean"],
        yerr=cnt_by_year["sem"],
        fmt="o",
        color="#555555",
        ecolor="#aaaaaa",
        capsize=3,
        label="per-year mean (raw, detected)",
        zorder=3,
    )

    grand_mean_cnt = float(cnt["raw"].mean())
    years_cnt = np.array(sorted(cnt["year"].unique()), dtype=float)

    ax.plot(
        years_cnt,
        grand_mean_cnt + _TAU_COUNT_TRUE * (years_cnt - year_center),
        color="#ff8c00",
        linewidth=2.5,
        label=f"true τ_count = {_TAU_COUNT_TRUE:+.3f}",
    )
    ax.plot(
        years_cnt,
        grand_mean_cnt + pred_full.ate_count * (years_cnt - year_center),
        color="#00008b",
        linewidth=2,
        linestyle="--",
        label=f"full: τ̂ = {pred_full.ate_count:+.4f}",
    )
    ax.plot(
        years_cnt,
        grand_mean_cnt + pred_hab_only.ate_count * (years_cnt - year_center),
        color="#7f00ff",
        linewidth=2,
        linestyle=":",
        label=f"hab-only: τ̂ = {pred_hab_only.ate_count:+.4f}",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean log1p-count (detected rows)")
    ax.set_title("Count stage (detection rows only)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    assert out_path.exists(), f"Plot was not written to {out_path}"
