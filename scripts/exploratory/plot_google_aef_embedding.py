"""Download and visualise Google AlphaEarth Foundations (AEF) embeddings.

Downloads 2019 tiles covering the bottom half of the UK from Source
Cooperative via Cloud-Optimized GeoTIFF VRTs (required – raw TIFFs are
bottom-up), reads at a coarse overview level (~640 m effective resolution),
mosaics, runs PCA and plots PC1-3 as an RGB image.

Data: https://source.coop/tge-labs/aef  (CC-BY 4.0, Google/DeepMind)
Attribution: "The AlphaEarth Foundations Satellite Embedding dataset is
produced by Google and Google DeepMind."
"""

from __future__ import annotations

import io
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio
import requests
from rasterio.enums import Resampling
from sklearn.decomposition import PCA

from bgbw_trends.viz import style

# ── Config ────────────────────────────────────────────────────────────────────
YEAR = 2019
# Bottom half of UK (south of ~54.5 N)
BBOX_WGS84 = (-6.5, 49.8, 2.5, 54.5)   # west, south, east, north
INDEX_URL  = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.csv"
# S3 bucket base for path → HTTPS conversion
S3_PREFIX  = "s3://us-west-2.opendata.source.coop/"
HTTPS_BASE = "https://data.source.coop/"
OUT_DIR    = Path("output/prototype/figures/diagnostics")
OUT_PATH   = OUT_DIR / "google_aef_pca_rgb_2019_southern_uk.png"
UK_SHP     = Path(
    "/home/nilomr/projects/bgbw.metadata.pipeline"
    "/data/metadata/uk-shapefile/UK-shapefile.shp"
)

# Each tile is 8192×8192 px at 10 m → 81.92 km.
# Read at 128×128 → ~640 m/px.  Adjust up/down for speed vs. resolution.
TILE_PX = 128


# ── Helpers ───────────────────────────────────────────────────────────────────
def s3_to_vrt_url(s3_path: str) -> str:
    """Convert S3 URI to VSICURL VRT URL."""
    https = s3_path.replace(S3_PREFIX, HTTPS_BASE, 1).replace(".tiff", ".vrt")
    return f"/vsicurl/{https}"


def dequantize(arr: np.ndarray) -> np.ndarray:
    """Signed int8 [-127, 127] → float32 [-1, 1].  -128 = nodata → NaN."""
    raw = arr.astype(np.float32)
    nodata = arr == -128
    out = (raw / 127.5) ** 2 * np.sign(raw)
    out[nodata] = np.nan
    return out


def pct_stretch(arr: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return np.zeros_like(arr)
    vmin, vmax = np.percentile(finite, lo), np.percentile(finite, hi)
    return np.clip((arr - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0)


# ── 1. Fetch index and filter ────────────────────────────────────────────────
print("Fetching AEF index …")
resp = requests.get(INDEX_URL, timeout=60)
resp.raise_for_status()
idx = pd.read_csv(io.StringIO(resp.text))
print(f"  {len(idx):,} rows  columns: {idx.columns.tolist()}")

west, south, east, north = BBOX_WGS84
tiles = idx[
    (idx["year"]        == YEAR)
    & (idx["wgs84_east"]  > west)
    & (idx["wgs84_west"]  < east)
    & (idx["wgs84_north"] > south)
    & (idx["wgs84_south"] < north)
].copy()
print(f"  {len(tiles)} tiles for year={YEAR} within bbox")

# ── 2. Read each tile at coarse resolution via VRT ────────────────────────────
_tf_cache: dict[str, pyproj.Transformer] = {}

def get_tf(crs_str: str) -> pyproj.Transformer:
    if crs_str not in _tf_cache:
        _tf_cache[crs_str] = pyproj.Transformer.from_crs(
            crs_str, "EPSG:4326", always_xy=True
        )
    return _tf_cache[crs_str]


all_emb: list[np.ndarray] = []
all_lon: list[np.ndarray] = []
all_lat: list[np.ndarray] = []
n_err = 0

for i, (_, row) in enumerate(tiles.iterrows()):
    vsicurl = s3_to_vrt_url(row["path"])
    label = row["path"].split("/")[-1][:44]
    print(f"  [{i+1:02d}/{len(tiles)}] {label} …", end=" ", flush=True)

    try:
        with rasterio.open(vsicurl) as src:
            n_bands  = src.count          # 64
            h_full   = src.height         # 8192
            w_full   = src.width          # 8192

            data = src.read(
                out_shape=(n_bands, TILE_PX, TILE_PX),
                resampling=Resampling.average,
            )  # int8 (64, TILE_PX, TILE_PX)

            emb = dequantize(data)        # float32 (64, H, W)

            # Pixel-centre coordinates in the tile's native UTM CRS
            scale_x = w_full / TILE_PX
            scale_y = h_full / TILE_PX
            col_c = (np.arange(TILE_PX) + 0.5) * scale_x
            row_c = (np.arange(TILE_PX) + 0.5) * scale_y
            xs_grid, ys_grid = np.meshgrid(col_c, row_c)

            nat_x, nat_y = rasterio.transform.xy(
                src.transform,
                ys_grid.ravel().tolist(),
                xs_grid.ravel().tolist(),
            )
            nat_x = np.asarray(nat_x, dtype=np.float64)
            nat_y = np.asarray(nat_y, dtype=np.float64)

            lons, lats = get_tf(src.crs.to_string()).transform(nat_x, nat_y)

            flat = emb.reshape(n_bands, -1).T          # (N, 64)
            valid = ~np.isnan(flat).any(axis=1)

            if valid.sum() == 0:
                print("no valid pixels, skip")
                continue

            all_emb.append(flat[valid])
            all_lon.append(np.asarray(lons)[valid])
            all_lat.append(np.asarray(lats)[valid])
            print(f"{valid.sum():,} px")

    except Exception as exc:
        print(f"ERROR: {exc}")
        n_err += 1

print(f"\nRead {len(all_emb)} tiles OK, {n_err} errors")
if not all_emb:
    raise RuntimeError("No tiles read successfully.")

embeddings = np.concatenate(all_emb, axis=0)   # (N_total, 64)
lons_all   = np.concatenate(all_lon)
lats_all   = np.concatenate(all_lat)
print(f"Total pixels: {len(embeddings):,}")

# ── 3. PCA → 3 components ────────────────────────────────────────────────────
print("Fitting PCA …")
pca = PCA(n_components=3, random_state=0)
scores = pca.fit_transform(embeddings)
ev = pca.explained_variance_ratio_ * 100
print(f"  EV: PC1={ev[0]:.1f}%  PC2={ev[1]:.1f}%  PC3={ev[2]:.1f}%")

rgb_vals = np.stack([pct_stretch(scores[:, k]) for k in range(3)], axis=1)

# ── 4. Rasterise onto a regular WGS84 grid ───────────────────────────────────
RES_DEG = 0.008  # ~640 m at mid-lat
ncols = int(np.ceil((east  - west)  / RES_DEG))
nrows = int(np.ceil((north - south) / RES_DEG))

col_idx = np.floor((lons_all - west)  / RES_DEG).astype(int)
row_idx = np.floor((lats_all - south) / RES_DEG).astype(int)

in_grid = (col_idx >= 0) & (col_idx < ncols) & (row_idx >= 0) & (row_idx < nrows)
rgb_img   = np.zeros((nrows, ncols, 3), dtype=np.float32)
count_img = np.zeros((nrows, ncols),    dtype=np.int32)
np.add.at(rgb_img,   (row_idx[in_grid], col_idx[in_grid]), rgb_vals[in_grid])
np.add.at(count_img, (row_idx[in_grid], col_idx[in_grid]), 1)

has = count_img > 0
rgb_img[has] /= count_img[has, np.newaxis]

rgba = np.zeros((nrows, ncols, 4), dtype=np.float32)
rgba[:, :, :3] = rgb_img
rgba[:, :, 3]  = has.astype(np.float32)

# ── 5. Plot ───────────────────────────────────────────────────────────────────
uk = None
if UK_SHP.exists():
    uk = gpd.read_file(UK_SHP).to_crs(epsg=4326)

fig, ax = plt.subplots(figsize=(8, 9), facecolor=style.PAPER)
ax.set_facecolor(style.PAPER)

if uk is not None:
    uk.plot(ax=ax, color=style.LAND, linewidth=0, zorder=1)

ax.imshow(rgba, extent=(west, east, south, north),
          origin="lower", aspect="auto",
          interpolation="nearest", zorder=2)

if uk is not None:
    uk.boundary.plot(ax=ax, color=style.COAST, linewidth=0.4, zorder=5)

ax.set_xlim(west, east)
ax.set_ylim(south, north)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_title(
    f"Google AEF embeddings {YEAR} — PCA RGB (southern UK)\n"
    f"PC1={ev[0]:.1f}%  PC2={ev[1]:.1f}%  PC3={ev[2]:.1f}%"
    f"  |  {len(tiles)} tiles  ·  ~640 m/px",
    fontsize=10, color=style.TEXT, pad=8,
)
for k, (ch, label) in enumerate(zip(["R", "G", "B"], ["PC1", "PC2", "PC3"])):
    ax.annotate(f"{ch} = {label}", xy=(0.02, 0.05 - k * 0.026),
                xycoords="axes fraction", fontsize=8, color=style.TEXT_MUTED)

plt.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
