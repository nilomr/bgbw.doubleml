"""Shared feature engineering for trends and SDM modules."""

from bgbw_trends.features.habitat import (
    make_habitat_features,
    make_spatial_features_W,
    reduce_embeddings_pca,
    apply_embedding_pca,
)
from bgbw_trends.features.effort import make_effort_features, compute_reference_effort
from bgbw_trends.features.response import (
    make_occurrence,
    make_log_count,
)

__all__ = [
    "make_habitat_features",
    "make_spatial_features_W",
    "reduce_embeddings_pca",
    "apply_embedding_pca",
    "make_effort_features",
    "compute_reference_effort",
    "make_occurrence",
    "make_log_count",
]
