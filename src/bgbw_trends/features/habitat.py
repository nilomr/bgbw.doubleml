"""Habitat and spatial feature construction.

These are X_hab features — stable properties of the location that are
varied at SDM predict time and included in DML's X.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def _group_embedding_cols(cols: list[str]) -> dict[str, list[str]]:
    """Group emb_* columns by their stat prefix.

    e.g. ``emb_0`` → ``"emb"``, ``emb_std_0`` → ``"emb_std"``.
    """
    groups: dict[str, list[str]] = {}
    for col in sorted(cols):
        prefix = re.sub(r"_\d+$", "", col)
        groups.setdefault(prefix, []).append(col)
    return groups


def get_embedding_columns(df: pd.DataFrame) -> list[str]:
    """Return all embedding columns present in df.

    Matches any column starting with ``emb_``, covering mean (``emb_0..d-1``)
    and distributional summaries (``emb_std_*``, ``emb_p10_*``, ``emb_p90_*``).
    Sorted for deterministic ordering.
    """
    return sorted(c for c in df.columns if c.startswith("emb_"))


def reduce_embeddings_pca(
    df: pd.DataFrame, n_components: int
) -> tuple[pd.DataFrame, dict[str, PCA]]:
    """Fit per-stat PCA on embedding columns and return reduced DataFrame.

    Applies a separate PCA to each distributional summary group (mean, std,
    p10, p90 etc.), preserving the column-naming convention so all downstream
    code works transparently. Each group contributes ``n_components`` PCs,
    so the total embedding dimensionality is ``n_groups × n_components``.

    Keeping stats separate is intentional: mean embeddings capture landscape
    composition; std captures within-cell heterogeneity. Their variance
    structures are unrelated, so a joint PCA would let mean dimensions
    dominate and compress the std signal.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least one set of emb_* columns.
    n_components : int
        Number of principal components to retain *per stat group*.

    Returns
    -------
    df_reduced : pd.DataFrame
        df with all emb_* groups replaced by their respective PCA columns.
    pca_models : dict[str, PCA]
        Fitted PCA per stat prefix (pass to apply_embedding_pca for new data).
    """
    emb_cols = get_embedding_columns(df)
    if not emb_cols:
        raise ValueError("No embedding columns found in df")

    groups = _group_embedding_cols(emb_cols)
    non_emb = [c for c in df.columns if c not in set(emb_cols)]
    parts: list[pd.DataFrame] = [df[non_emb]]
    pca_models: dict[str, PCA] = {}

    for prefix, cols in groups.items():
        values = df[cols].to_numpy(dtype=np.float64)
        k = min(n_components, len(cols))
        if k >= len(cols):
            logger.info(
                "Embedding PCA [%s]: n_components=%d >= n_dims=%d; skipping",
                prefix,
                n_components,
                len(cols),
            )
            pca = PCA(n_components=len(cols)).fit(values)
            pca_models[prefix] = pca
            parts.append(df[cols])
            continue

        pca = PCA(n_components=k, random_state=42)
        pc_values = pca.fit_transform(values)
        var_explained = pca.explained_variance_ratio_.sum()
        logger.info(
            "Embedding PCA [%s]: %d → %d components, %.1f%% variance explained",
            prefix,
            len(cols),
            k,
            var_explained * 100,
        )
        pc_cols = [f"{prefix}_{i}" for i in range(k)]
        pca_models[prefix] = pca
        parts.append(pd.DataFrame(pc_values, columns=pc_cols, index=df.index))

    return pd.concat(parts, axis=1), pca_models


def apply_embedding_pca(df: pd.DataFrame, pca_models: dict[str, PCA]) -> pd.DataFrame:
    """Apply per-stat fitted PCAs to embedding columns (e.g. for the SDM grid).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the same stat groups as were used to fit pca_models.
    pca_models : dict[str, PCA]
        Mapping ``{prefix: PCA}`` returned by reduce_embeddings_pca.

    Returns
    -------
    pd.DataFrame
        df with all emb_* groups replaced by their respective PCA components.
    """
    emb_cols = get_embedding_columns(df)
    if not emb_cols:
        raise ValueError("No embedding columns found in df")

    groups = _group_embedding_cols(emb_cols)
    non_emb = [c for c in df.columns if c not in set(emb_cols)]
    parts: list[pd.DataFrame] = [df[non_emb]]

    for prefix, cols in groups.items():
        pca = pca_models[prefix]
        values = df[cols].to_numpy(dtype=np.float64)
        pc_values = pca.transform(values)
        pc_cols = [f"{prefix}_{i}" for i in range(pca.n_components_)]
        parts.append(pd.DataFrame(pc_values, columns=pc_cols, index=df.index))

    return pd.concat(parts, axis=1)


def make_habitat_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the habitat/spatial feature matrix (X_hab).

    For grid-aggregated data, this is embeddings + coordinates.
    For individual-level data, also includes garden structural features.
    """
    parts = []

    # Geospatial embeddings (landscape composition at 1 km)
    emb_cols = get_embedding_columns(df)
    if emb_cols:
        parts.append(df[emb_cols].astype(np.float64))

    # Coordinates (always include)
    for coord in ["latitude", "longitude"]:
        if coord in df.columns:
            parts.append(df[[coord]].astype(np.float64))

    # Garden structural features (individual-level data only)
    garden_structural = [
        "garden_size",
        "nearest_farmland",
        "urban_rural",
        "lawn",
        "pond",
        "wildlife_area",
        "compost_heap",
        "flower_beds",
        "large_trees",
        "small_trees",
        "potted_plants",
        "deciduous_shrubs",
        "coniferous_shrubs",
        "climber",
    ]
    for feat in garden_structural:
        if feat in df.columns:
            col = df[feat]
            if col.dtype == object or hasattr(col, "str"):
                # Encode categoricals as integer codes
                parts.append(
                    pd.DataFrame(
                        {feat: col.astype("category").cat.codes.astype(np.float64)},
                        index=df.index,
                    )
                )
            else:
                parts.append(df[[feat]].astype(np.float64))

    if not parts:
        raise ValueError("No habitat features could be constructed")

    result = pd.concat(parts, axis=1)
    logger.debug("Habitat features (X_hab): %d features", result.shape[1])
    return result


def make_spatial_features_W(df: pd.DataFrame) -> pd.DataFrame:
    """Build effect-modifier features W for heterogeneous trends.

    W = embeddings + lat/lon. The trend τ(W) varies across these.
    Do NOT put effort/observation-process variables in W.
    """
    parts = []

    emb_cols = get_embedding_columns(df)
    if emb_cols:
        parts.append(df[emb_cols].astype(np.float64))

    for coord in ["latitude", "longitude"]:
        if coord in df.columns:
            parts.append(df[[coord]].astype(np.float64))

    if not parts:
        raise ValueError("No spatial features found for W")

    result = pd.concat(parts, axis=1)
    logger.debug("Spatial features (W): %d features", result.shape[1])
    return result
