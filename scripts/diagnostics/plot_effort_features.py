"""Spatial maps of all effort (observation-process) features for 2019.

Loads harmonised survey data for 2019, aggregates to 1 km BNG cells,
and plots n_surveys, mean_observers, proportion_adult, proportion_online
as a 2×2 map panel.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import numpy as np
import pandas as pd

from bgbw_trends.aggregate import aggregate_surveys
from bgbw_trends.config import EFFORT_FEATURES, HARMONIZED_DIR
from bgbw_trends.data import load_harmonized
from bgbw_trends.viz._bng import refs_to_en, to_raster
from bgbw_trends.viz import style

OUT = Path("output/prototype/figures/diagnostics/effort_features_2019.png")
UK_SHP = Path(
    "/home/nilomr/projects/bgbw.metadata.pipeline"
    "/data/metadata/uk-shapefile/UK-shapefile.shp"
)

# ── Load and aggregate ───────────────────────────────────────────────────────
print("Loading harmonized 2019 …")
raw = load_harmonized(2019, HARMONIZED_DIR)
print(f"  {len(raw):,} individual surveys")
surveys = aggregate_surveys(raw)
print(f"  Aggregated to {len(surveys):,} BNG cells")

refs = surveys["bng_ref"].values
easting, northing = refs_to_en(refs)

# ── UK shapefile ─────────────────────────────────────────────────────────────
uk = None
if UK_SHP.exists():
    uk = gpd.read_file(UK_SHP)
    if uk.crs.to_epsg() != 27700:
        uk = uk.to_crs(epsg=27700)

# ── Plot helpers ─────────────────────────────────────────────────────────────
LABELS = {
    "n_surveys": "N surveys",
    "mean_observers": "Mean observers",
    "proportion_adult": "Proportion adult",
    "proportion_online": "Proportion online",
}
CMAPS = {
    "n_surveys": "magma",
    "mean_observers": "magma",
    "proportion_adult": "viridis",
    "proportion_online": "viridis",
}

present = [f for f in EFFORT_FEATURES if f in surveys.columns]
print(f"  Effort features available: {present}")

fig, axes = plt.subplots(2, 2, figsize=(14, 18), facecolor=style.PAPER)
fig.subplots_adjust(hspace=0.05, wspace=0.04)

for ax, feat in zip(axes.flat, present):
    vals = surveys[feat].values.astype(float)
    raster, extent = to_raster(easting, northing, vals)
    e_min, e_max, n_min, n_max = extent

    vmax = np.nanpercentile(raster, 98)
    norm = mc.Normalize(vmin=float(np.nanmin(raster)), vmax=vmax)

    ax.set_facecolor(style.PAPER)
    if uk is not None:
        uk.plot(ax=ax, color=style.LAND, linewidth=0, zorder=1)

    im = ax.imshow(
        raster,
        extent=extent,
        origin="lower",
        aspect="equal",
        cmap=CMAPS[feat],
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
    ax.set_title(LABELS[feat], fontsize=11, color=style.TEXT, pad=6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, aspect=30)
    cbar.ax.tick_params(labelsize=7, colors=style.TEXT_MUTED)
    cbar.outline.set_visible(False)

# hide any unused axes
for ax in axes.flat[len(present) :]:
    ax.set_visible(False)

fig.suptitle(
    "Effort features — 2019 BNG aggregation", fontsize=13, color=style.TEXT, y=0.995
)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Saved → {OUT}")
