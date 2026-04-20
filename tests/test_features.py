"""Tests for feature engineering."""

import numpy as np
import pandas as pd

from bgbw_trends.features import (
    make_log_count,
    make_occurrence,
    make_treatment,
)


def _make_dummy_df(n: int = 100, n_emb: int = 8) -> pd.DataFrame:
    """Create a minimal DataFrame mimicking assembled analysis data."""
    rng = np.random.default_rng(42)
    data = {
        "year": rng.choice([2018, 2019], size=n),
        "bng_ref": [f"TQ{i % 50:04d}" for i in range(n)],
        "n_surveys": rng.integers(1, 10, size=n),
        "mean_observers": rng.uniform(1.0, 2.0, size=n),
        "proportion_adult": rng.uniform(0.7, 1.0, size=n),
        "proportion_online": rng.uniform(0.5, 0.9, size=n),
        "blue_tit": rng.poisson(3, size=n).astype(float),
        "house_sparrow": rng.poisson(4, size=n).astype(float),
    }
    for i in range(n_emb):
        data[f"emb_{i}"] = rng.normal(0, 1, size=n)
    return pd.DataFrame(data)


class TestOutcomes:
    def test_occurrence_binary(self):
        df = _make_dummy_df()
        occ = make_occurrence(df, "blue_tit")
        assert set(occ.unique()).issubset({0.0, 1.0})
        assert occ.dtype == np.float64

    def test_log_count_nan_for_zeros(self):
        df = _make_dummy_df()
        lc = make_log_count(df, "blue_tit")
        zero_mask = df["blue_tit"] == 0
        assert lc[zero_mask].isna().all()
        assert lc[~zero_mask].notna().all()

    def test_log_count_uses_log1p(self):
        """Fink train-predict.R L149-151: log(obs + alpha) with alpha=1."""
        df = _make_dummy_df()
        lc = make_log_count(df, "blue_tit")
        detected = df["blue_tit"] > 0
        expected = np.log1p(df.loc[detected, "blue_tit"])
        np.testing.assert_array_almost_equal(
            lc[detected].values, expected.values
        )


class TestTreatment:
    def test_centred(self):
        df = _make_dummy_df()
        T = make_treatment(df, [2018, 2019])
        assert abs(T.mean()) < 0.2
        assert T.min() == -0.5
        assert T.max() == 0.5
