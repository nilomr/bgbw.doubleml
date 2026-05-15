"""Effort and observation-process feature construction.

These are X_effort features — cell-level summaries of the observation
process (observer counts, survey mode) that:
- Are included in both DML X and SDM training features
- Are held at REFERENCE values for SDM prediction
- Are left at actual values for DML nuisance models

The features come from richer aggregation of individual surveys to the
1 km BNG grid (see aggregate.py), following Johnston 2025 / Fink 2023.
"""

from __future__ import annotations

import logging
from dataclasses import fields

import numpy as np
import pandas as pd

from bgbw_trends.config import EFFORT_FEATURES, ReferenceEffort

logger = logging.getLogger(__name__)


def make_effort_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the effort/observation-process feature matrix (X_effort).

    Expects the aggregated effort columns produced by ``aggregate.aggregate_surveys``.
    """
    present = [f for f in EFFORT_FEATURES if f in df.columns]
    missing = [f for f in EFFORT_FEATURES if f not in df.columns]
    if missing:
        logger.warning("Missing effort features (will be excluded): %s", missing)
    if not present:
        raise ValueError(
            "No effort features found. Expected columns: %s" % EFFORT_FEATURES
        )

    result = df[present].astype(np.float64).copy()

    # Fill any remaining NaN with column medians (e.g. proportion_adult
    # is NaN when total observers = 0 for a cell).
    n_nan = result.isna().sum().sum()
    if n_nan > 0:
        result = result.fillna(result.median())
        logger.debug(
            "Filled %d NaN values in effort features with column medians", n_nan
        )

    logger.debug("Effort features (X_effort): %s", list(result.columns))
    return result


def compute_reference_effort(df: pd.DataFrame) -> ReferenceEffort:
    """Compute reference-effort values from training data.

    Uses median for all numeric effort features.
    """
    ref = ReferenceEffort()
    for f in fields(ref):
        if f.name not in df.columns:
            continue
        col = df[f.name].dropna()
        if len(col) == 0:
            continue
        ref_val = float(col.median())
        setattr(ref, f.name, ref_val)

    logger.info("Computed reference effort from %d rows", len(df))
    return ref
