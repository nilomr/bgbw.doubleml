"""Tests for survey aggregation to BNG grid."""

import numpy as np
import pandas as pd
import pytest

from bgbw_trends.aggregate import aggregate_surveys


def _make_harmonized(n: int = 200) -> pd.DataFrame:
    """Create minimal harmonized survey data with known properties."""
    rng = np.random.default_rng(42)
    # Coordinates near central England (roughly BNG extent)
    lat = rng.uniform(51.0, 53.0, size=n)
    lon = rng.uniform(-2.0, 0.5, size=n)
    return pd.DataFrame(
        {
            "year": rng.choice([2018, 2019], size=n),
            "latitude": lat,
            "longitude": lon,
            "postcode_valid": True,
            "source": rng.choice(["online", "paper"], size=n, p=[0.7, 0.3]),
            "adults": rng.choice([1, 2, 3], size=n).astype(float),
            "children": rng.choice([0, 0, 1, 2], size=n).astype(float),
            "blue_tit": rng.poisson(3, size=n).astype(float),
            "house_sparrow": rng.poisson(4, size=n).astype(float),
        }
    )


class TestAggregateSurveys:
    def test_output_columns(self):
        df = _make_harmonized()
        result = aggregate_surveys(df)
        assert "bng_ref" in result.columns
        assert "year" in result.columns
        assert "n_surveys" in result.columns
        assert "mean_observers" in result.columns
        assert "proportion_adult" in result.columns
        assert "proportion_online" in result.columns
        assert "blue_tit" in result.columns
        assert "house_sparrow" in result.columns

    def test_rows_are_cell_year(self):
        df = _make_harmonized()
        result = aggregate_surveys(df)
        # No duplicate (bng_ref, year) pairs
        assert not result.duplicated(subset=["bng_ref", "year"]).any()

    def test_n_surveys_positive(self):
        df = _make_harmonized()
        result = aggregate_surveys(df)
        assert (result["n_surveys"] > 0).all()

    def test_proportions_bounded(self):
        df = _make_harmonized()
        result = aggregate_surveys(df)
        assert result["proportion_adult"].dropna().between(0, 1).all()
        assert result["proportion_online"].between(0, 1).all()

    def test_mean_observers_positive(self):
        df = _make_harmonized()
        result = aggregate_surveys(df)
        assert (result["mean_observers"] >= 0).all()

    def test_fewer_rows_than_input(self):
        df = _make_harmonized(500)
        result = aggregate_surveys(df)
        assert len(result) < len(df)

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"latitude": [51.5], "longitude": [-0.1]})
        with pytest.raises(ValueError, match="Missing required columns"):
            aggregate_surveys(df)
