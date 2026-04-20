"""Design tokens and matplotlib configuration for BGBW maps."""

from __future__ import annotations

import matplotlib.pyplot as plt

# ── Palette ────────────────────────────────────────────────────────────────

PAPER = "#F7F4EE"  # warm paper — figure surround
WATER = "#F7F4EE"  # sea / negative space — same as PAPER for unified background
LAND = "#242424"  # rich charcoal — land with no model coverage
COAST = "#5C5C5C"  # coastline stroke
TEXT = "#151515"  # near-black — titles, labels
TEXT_MUTED = "#66615B"  # secondary labels, captions
GRIDLINE = "#D9D1C7"  # restrained dividers under legends

# Use standard Matplotlib colormaps (no custom palettes).
TREND_CMAP = plt.get_cmap("RdBu")
OCC_CMAP = plt.get_cmap("viridis")
ABD_CMAP = plt.get_cmap("magma")


# ── Typography ─────────────────────────────────────────────────────────────

FONT_FAMILY = "DejaVu Sans"
FONT_TITLE = dict(fontsize=22, fontweight="700", color=TEXT, fontfamily=FONT_FAMILY)
FONT_SUBTITLE = dict(fontsize=11.5, color=TEXT_MUTED, fontfamily=FONT_FAMILY)
FONT_SECTION = dict(fontsize=9.5, fontweight="600", color=TEXT, fontfamily=FONT_FAMILY)
FONT_LABEL = dict(fontsize=8.5, color=TEXT_MUTED, fontfamily=FONT_FAMILY)
FONT_ANNOTATION = dict(fontsize=8, color=TEXT_MUTED, fontfamily=FONT_FAMILY)


# ── rcParams ───────────────────────────────────────────────────────────────

RC = {
    "figure.facecolor": PAPER,
    "axes.facecolor": WATER,
    "savefig.facecolor": PAPER,
    "font.family": FONT_FAMILY,
    "text.color": TEXT,
    "axes.edgecolor": COAST,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "axes.titlelocation": "left",
}


def apply() -> None:
    """Apply BGBW rc params globally."""
    plt.rcParams.update(RC)
