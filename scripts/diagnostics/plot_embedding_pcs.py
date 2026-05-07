"""Grid of individual PCA components from the 2019 tessera mean embeddings.

Plots PC1–PC12 as individual grayscale maps to reveal tile/grid artifacts
that are invisible when only the first 3 PCs are shown as RGB.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import numpy as np
from sklearn.decomposition import PCA

from bgbw_trends.config import GEOTESSERA_DIR
from bgbw_trends.data import load_embeddings
from bgbw_trends.viz._bng import refs_to_en, to_raster
from bgbw_trends.viz import style

N_COMPONENTS = 12
OUT = Path("output/prototype/figures/diagnostics/embedding_pca_components_2019.png")
UK_SHP = Path(
    "/home/nilomr/projects/bgbw.metadata.pipeline"
    "/data/metadata/uk-shapefile/UK-shapefile.shp"
)

# ── Load & fit ───────────────────────────────────────────────────────────────
print("Loading 2019 mean embeddings …")
emb = load_embeddings(year=2019, directory=GEOTESSERA_DIR, summaries=["mean"])
mean_cols = sorted(
    c for c in emb.columns if c.startswith("emb_") and not c.startswith("emb_std")
)
X = emb[mean_cols].values.astype(np.float32)
refs = emb.index.values
print(f"  {X.shape[0]:,} cells × {X.shape[1]} dims")

pca = PCA(n_components=N_COMPONENTS, random_state=0)
scores = pca.fit_transform(X)
ev = pca.explained_variance_ratio_ * 100
print(f"  Explained variance per PC: {ev.round(1)}")

easting, northing = refs_to_en(refs)

# ── UK shapefile ─────────────────────────────────────────────────────────────
uk = None
if UK_SHP.exists():
    uk = gpd.read_file(UK_SHP)
    if uk.crs.to_epsg() != 27700:
        uk = uk.to_crs(epsg=27700)

# ── Plot ─────────────────────────────────────────────────────────────────────
ncols = 4
nrows = int(np.ceil(N_COMPONENTS / ncols))
fig, axes = plt.subplots(
    nrows, ncols, figsize=(ncols * 4, nrows * 5.5), facecolor=style.PAPER
)
fig.subplots_adjust(hspace=0.05, wspace=0.04)

for i, ax in enumerate(axes.flat):
    if i >= N_COMPONENTS:
        ax.set_visible(False)
        continue

    raster, extent = to_raster(easting, northing, scores[:, i])
    e_min, e_max, n_min, n_max = extent

    # Symmetric colour scale centred on 0
    vabs = np.nanpercentile(np.abs(raster[np.isfinite(raster)]), 98)
    norm = mc.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

    ax.set_facecolor(style.PAPER)
    if uk is not None:
        uk.plot(ax=ax, color=style.LAND, linewidth=0, zorder=1)

    ax.imshow(
        raster,
        extent=extent,
        origin="lower",
        aspect="equal",
        cmap="RdBu_r",
        norm=norm,
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
    ax.set_title(f"PC{i + 1}  ({ev[i]:.1f}%)", fontsize=10, color=style.TEXT, pad=5)

fig.suptitle(
    "Tessera 2019 mean embeddings — PCs 1–12 (RdBu: −/+)",
    fontsize=13,
    color=style.TEXT,
    y=1.001,
)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")
