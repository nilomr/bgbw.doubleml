"""Smoke tests for the two-part DML + SDM pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bgbw_trends.config import ReferenceEffort, RunConfig
from bgbw_trends.features.effort import compute_reference_effort
from bgbw_trends.pipeline import (
    SpeciesModelDiagnostics,
    SpeciesPrediction,
    _build_prediction_effort,
    _fit_nuisance,
    train_predict_species,
)


def _make_cell_year_frame(
    n_cells: int = 40, years: tuple[int, ...] = (2018, 2019, 2020), n_emb: int = 4
) -> pd.DataFrame:
    """Aggregated-cell frame with a trend-generating response."""
    rng = np.random.default_rng(0)
    rows = []
    for yr in years:
        for c in range(n_cells):
            # cell-level habitat characteristics (stable over years)
            emb = rng.normal(0, 1, size=n_emb)
            lat = 51.0 + 0.1 * (c // 10)
            lon = -1.0 + 0.1 * (c % 10)
            # abundance varies over years (small positive trend)
            mean = np.exp(0.5 + 0.02 * (yr - 2018) + 0.2 * emb[0])
            count = float(rng.poisson(mean))
            rows.append(
                {
                    "bng_ref": f"c{c:03d}",
                    "year": yr,
                    "blue_tit": count,
                    "n_surveys": int(rng.integers(1, 6)),
                    "mean_observers": float(rng.uniform(1.0, 2.0)),
                    "proportion_adult": float(rng.uniform(0.7, 1.0)),
                    "proportion_online": float(rng.uniform(0.5, 0.9)),
                    "latitude": lat,
                    "longitude": lon,
                    **{f"emb_{i}": float(emb[i]) for i in range(n_emb)},
                }
            )
    return pd.DataFrame(rows)


def _make_grid(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values("year")
        .drop_duplicates("bng_ref", keep="last")
        .reset_index(drop=True)
    )


@pytest.fixture(scope="module")
def small_run():
    df = _make_cell_year_frame()
    grid = _make_grid(df)
    cfg = RunConfig(
        species=["blue_tit"],
        years=[2018, 2019, 2020],
        n_estimators=40,
        cf_n_estimators=40,
        cf_min_samples_leaf=5,
        n_pca_components=0,
    )
    pred = train_predict_species(
        df,
        df,
        species="blue_tit",
        cfg=cfg,
        grid=grid,
        reference_effort=compute_reference_effort(df),
        seed=0,
    )
    return df, grid, pred


class TestPipelineOutput:
    def test_returns_species_prediction(self, small_run):
        _, _, pred = small_run
        assert isinstance(pred, SpeciesPrediction)
        assert isinstance(pred.diagnostics, SpeciesModelDiagnostics)

    def test_alignment(self, small_run):
        _, grid, pred = small_run
        np.testing.assert_array_equal(pred.bng_ref, grid["bng_ref"].to_numpy())

    def test_sdm_bounded(self, small_run):
        _, _, pred = small_run
        assert np.all(pred.p_occurrence >= 0)
        assert np.all(pred.p_occurrence <= 1)
        assert np.all(pred.e_count_given_occ >= 0)
        assert np.all(pred.e_abundance >= 0)
        np.testing.assert_allclose(
            pred.e_abundance, pred.p_occurrence * pred.e_count_given_occ
        )

    def test_chain_rule(self, small_run):
        _, _, pred = small_run
        # abd_ppy = occ_ppy + count_ppy wherever both are finite
        mask = np.isfinite(pred.occ_ppy) & np.isfinite(pred.count_ppy)
        np.testing.assert_allclose(
            pred.abd_ppy[mask], pred.occ_ppy[mask] + pred.count_ppy[mask]
        )

    def test_trend_sigmas_positive(self, small_run):
        _, _, pred = small_run
        assert np.all(pred.occ_sigma >= 0)
        assert np.all(pred.count_sigma >= 0)


class TestOobShortcut:
    """On sklearn we get OOB from the single full-data fit."""

    def test_oob_predictions_match_length(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(80, 5)))
        y = rng.normal(size=80)
        cfg = RunConfig(n_estimators=30)
        model, oof = _fit_nuisance(X, y, cfg, seed=0)
        assert oof.shape == (80,)
        assert np.all(np.isfinite(oof))
        # Same model predicts the same X without needing to refit
        assert model.predict(X).shape == (80,)


class TestPredictionEffort:
    def test_reference_values_used(self):
        training = pd.DataFrame(
            {"n_surveys": [1, 2, 3, 4], "proportion_online": [0.2, 0.3, 0.4, 0.5]}
        )
        ref = ReferenceEffort(n_surveys=5.0, proportion_online=0.8)
        out = _build_prediction_effort(training, ref, n=3)
        np.testing.assert_array_equal(out["n_surveys"].to_numpy(), [5.0, 5.0, 5.0])
        np.testing.assert_array_equal(
            out["proportion_online"].to_numpy(), [0.8, 0.8, 0.8]
        )

    def test_unknown_column_uses_median(self):
        training = pd.DataFrame({"foo": [1.0, 2.0, 3.0, 4.0, 5.0]})
        out = _build_prediction_effort(training, ReferenceEffort(), n=2)
        np.testing.assert_array_equal(out["foo"].to_numpy(), [3.0, 3.0])
