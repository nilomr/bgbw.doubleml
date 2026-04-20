"""BNG coordinate utilities for rasterisation."""

from __future__ import annotations

import numpy as np

# ── Grid-reference → metre conversion ─────────────────────────────────────

_FIRST = {"S": (0, 0), "T": (5, 0), "N": (0, 5), "O": (5, 5), "H": (0, 10)}
_SECOND = {
    "A": (0, 4), "B": (1, 4), "C": (2, 4), "D": (3, 4), "E": (4, 4),
    "F": (0, 3), "G": (1, 3), "H": (2, 3), "J": (3, 3), "K": (4, 3),
    "L": (0, 2), "M": (1, 2), "N": (2, 2), "O": (3, 2), "P": (4, 2),
    "Q": (0, 1), "R": (1, 1), "S": (2, 1), "T": (3, 1), "U": (4, 1),
    "V": (0, 0), "W": (1, 0), "X": (2, 0), "Y": (3, 0), "Z": (4, 0),
}


def refs_to_en(refs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert an array of BNG grid references to SW-corner easting/northing.

    Returns two float64 arrays (easting, northing), NaN for unparseable refs.
    """
    n = len(refs)
    east = np.full(n, np.nan, dtype=np.float64)
    north = np.full(n, np.nan, dtype=np.float64)

    for i, ref in enumerate(refs):
        try:
            f, s = ref[0], ref[1]
            digits = ref[2:]
            half = len(digits) // 2
            be, bn = _FIRST[f]
            se, sn = _SECOND[s]
            scale = 10 ** (5 - half)
            east[i] = (be + se) * 100_000 + int(digits[:half]) * scale
            north[i] = (bn + sn) * 100_000 + int(digits[half:]) * scale
        except Exception:
            pass

    return east, north


def to_raster(
    easting: np.ndarray,
    northing: np.ndarray,
    values: np.ndarray,
    cell_m: int = 1_000,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Rasterise scalar values to a 2-D float32 array (NaN = no data).

    Returns ``(raster, (e_min, e_max, n_min, n_max))``.  The ``extent``
    tuple is in metres and can be passed directly to ``imshow``.
    """
    sw_e = np.round(easting / cell_m).astype(np.int64) * cell_m
    sw_n = np.round(northing / cell_m).astype(np.int64) * cell_m

    valid = np.isfinite(sw_e) & np.isfinite(sw_n)
    e_min = int(sw_e[valid].min())
    e_max = int(sw_e[valid].max()) + cell_m
    n_min = int(sw_n[valid].min())
    n_max = int(sw_n[valid].max()) + cell_m

    ncols = (e_max - e_min) // cell_m
    nrows = (n_max - n_min) // cell_m

    raster = np.full((nrows, ncols), np.nan, dtype=np.float32)
    col = (sw_e[valid] - e_min) // cell_m
    row = (sw_n[valid] - n_min) // cell_m
    raster[row.astype(int), col.astype(int)] = values[valid].astype(np.float32)

    return raster, (float(e_min), float(e_max), float(n_min), float(n_max))
