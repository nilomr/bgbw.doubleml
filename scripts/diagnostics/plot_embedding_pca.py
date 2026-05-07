"""Visualise the 2019 tessera embeddings via PCA-RGB.

Loads mean and std embedding vectors for each 1 km BNG cell, fits PCA
on each separately, and maps PC1→R, PC2→G, PC3→B after per-channel
percentile-stretch.  Saves two figures: one for the mean, one for the std.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from bgbw_trends.config import GEOTESSERA_DIR
from bgbw_trends.data import load_embeddings
from bgbw_trends.viz._bng import refs_to_en
from bgbw_trends.viz import style

OUT_DIR = Path("output/prototype/figures/diagnostics")
UK_SHP = Path(
    "/home/nilomr/projects/bgbw.metadata.pipeline"
    "/data/metadata/uk-shapefile/UK-shapefile.shp"
)

# ── Shared helpers ───────────────────────────────────────────────────────────


def pct_stretch(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    vmin, vmax = np.percentile(arr, lo), np.percentile(arr, hi)
    return np.clip((arr - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0)


def rasterise_rgb(
    refs: np.ndarray,
    rgb_vals: np.ndarray,  # (n, 3) floats in [0, 1]
    cell_m: int = 1_000,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Return (rgba_image, mask, extent) on a BNG raster grid.

    origin='lower' convention: row 0 = south.  No manual flip needed.
    """
    easting, northing = refs_to_en(refs)
    valid = np.isfinite(easting) & np.isfinite(northing)

    sw_e = np.round(easting[valid] / cell_m).astype(np.int64) * cell_m
    sw_n = np.round(northing[valid] / cell_m).astype(np.int64) * cell_m

    e_min, e_max = int(sw_e.min()), int(sw_e.max()) + cell_m
    n_min, n_max = int(sw_n.min()), int(sw_n.max()) + cell_m

    ncols = (e_max - e_min) // cell_m
    nrows = (n_max - n_min) // cell_m

    rgb_img = np.zeros((nrows, ncols, 3), dtype=np.float32)
    mask = np.zeros((nrows, ncols), dtype=bool)

    col_idx = ((sw_e - e_min) // cell_m).astype(int)
    row_idx = ((sw_n - n_min) // cell_m).astype(int)

    rgb_img[row_idx, col_idx, :] = rgb_vals[valid]
    mask[row_idx, col_idx] = True

    # Build RGBA: fully transparent where no cell exists
    rgba = np.zeros((nrows, ncols, 4), dtype=np.float32)
    rgba[:, :, :3] = rgb_img
    rgba[:, :, 3] = mask.astype(np.float32)

    return rgba, mask, (float(e_min), float(e_max), float(n_min), float(n_max))


_uk: gpd.GeoDataFrame | None = None


def _get_uk() -> gpd.GeoDataFrame | None:
    global _uk
    if _uk is not None:
        return _uk
    if not UK_SHP.exists():
        return None
    gdf = gpd.read_file(UK_SHP)
    if gdf.crs.to_epsg() != 27700:
        gdf = gdf.to_crs(epsg=27700)
    _uk = gdf
    return _uk


def make_pca_rgb_figure(
    X: np.ndarray,
    refs: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Fit PCA(3), rasterise as RGB, save figure."""
    print(f"  Fitting PCA for '{title}' …")
    pca = PCA(n_components=3, random_state=0)
    scores = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_ * 100
    print(f"    Explained variance: {ev}")

    rgb_vals = np.stack([pct_stretch(scores[:, i]) for i in range(3)], axis=1)
    rgba, _mask, extent = rasterise_rgb(refs, rgb_vals)
    e_min, e_max, n_min, n_max = extent

    fig, ax = plt.subplots(figsize=(8, 11), facecolor=style.PAPER)
    ax.set_facecolor(style.PAPER)

    uk = _get_uk()
    if uk is not None:
        uk.plot(ax=ax, color=style.LAND, linewidth=0, zorder=1)

    # origin='lower' → row 0 is south; no manual flip needed
    ax.imshow(
        rgba,
        extent=extent,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        zorder=2,
    )

    if uk is not None:
        uk.boundary.plot(ax=ax, color=style.COAST, linewidth=0.4, zorder=5)

    ax.set_xlim(e_min, e_max)
    ax.set_ylim(n_min, n_max)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        f"{title}\nPC1={ev[0]:.1f}%  PC2={ev[1]:.1f}%  PC3={ev[2]:.1f}%",
        fontsize=11,
        color=style.TEXT,
        pad=10,
    )
    for i, (ch, label) in enumerate(zip(["R", "G", "B"], ["PC1", "PC2", "PC3"])):
        ax.annotate(
            f"{ch} = {label}",
            xy=(0.02, 0.04 - i * 0.025),
            xycoords="axes fraction",
            fontsize=8,
            color=style.TEXT_MUTED,
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Mean embeddings ──────────────────────────────────────────────────────────
print("Loading 2019 embeddings (mean + std) …")
emb = load_embeddings(year=2019, directory=GEOTESSERA_DIR, summaries=["mean", "std"])
refs = emb.index.values

mean_cols = sorted(
    c for c in emb.columns if c.startswith("emb_") and not c.startswith("emb_std")
)
std_cols = sorted(c for c in emb.columns if c.startswith("emb_std_"))

print(f"  {len(refs):,} cells  |  mean dims={len(mean_cols)}  std dims={len(std_cols)}")

make_pca_rgb_figure(
    emb[mean_cols].values.astype(np.float32),
    refs,
    title="Tessera 2019 — mean embeddings PCA RGB",
    out_path=OUT_DIR / "embedding_mean_pca_rgb_2019.png",
)

make_pca_rgb_figure(
    emb[std_cols].values.astype(np.float32),
    refs,
    title="Tessera 2019 — std embeddings PCA RGB (within-cell heterogeneity)",
    out_path=OUT_DIR / "embedding_std_pca_rgb_2019.png",
)
