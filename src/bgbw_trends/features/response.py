"""Response and treatment variable construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_occurrence(df: pd.DataFrame, species: str) -> pd.Series:
    """Binary indicator: was the species detected?"""
    return (df[species] > 0).astype(np.float64)


def make_log_count(df: pd.DataFrame, species: str) -> pd.Series:
    """log1p-transformed count for the count-given-occurrence model.

    Matches Fink et al. (2023) ``train-predict.R`` L149-151 which uses
    ``log(obs + alpha)`` with ``alpha = 1``. BNG-aggregated counts are
    cell-year means and can fall between 0 and 1 for detections with
    fractional mean counts; ``log1p`` is defined on [0, ∞) and is the
    robust choice. Returns NaN for non-detections.
    """
    counts = df[species].copy()
    result = np.full(len(counts), np.nan)
    detected = counts > 0
    result[detected] = np.log1p(counts[detected])
    return pd.Series(result, index=df.index, name=f"log_{species}")


