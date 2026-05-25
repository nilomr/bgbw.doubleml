"""Shared helpers for product plotting scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml


# ── Species common names ──────────────────────────────────────────────────────

COMMON_NAMES: dict[str, str] = {
    "barn_owl": "Barn Owl",
    "black_headed_gull": "Black-headed Gull",
    "black_redstart": "Black Redstart",
    "blackbird": "Blackbird",
    "blackcap": "Blackcap",
    "blue_tit": "Blue Tit",
    "brambling": "Brambling",
    "bullfinch": "Bullfinch",
    "buzzard": "Buzzard",
    "carrion_crow": "Carrion Crow",
    "chaffinch": "Chaffinch",
    "chiffchaff": "Chiffchaff",
    "coal_tit": "Coal Tit",
    "collared_dove": "Collared Dove",
    "common_gull": "Common Gull",
    "corn_bunting": "Corn Bunting",
    "dunnock": "Dunnock",
    "feral_pigeon": "Feral Pigeon",
    "fieldfare": "Fieldfare",
    "goldcrest": "Goldcrest",
    "goldfinch": "Goldfinch",
    "great_black_backed_gull": "Great Black-backed Gull",
    "great_spotted_woodpecker": "Gt. Spotted Woodpecker",
    "great_tit": "Great Tit",
    "green_woodpecker": "Green Woodpecker",
    "greenfinch": "Greenfinch",
    "grey_heron": "Grey Heron",
    "grey_partridge": "Grey Partridge",
    "grey_wagtail": "Grey Wagtail",
    "hawfinch": "Hawfinch",
    "herring_gull": "Herring Gull",
    "hooded_crow": "Hooded Crow",
    "house_sparrow": "House Sparrow",
    "jackdaw": "Jackdaw",
    "jay": "Jay",
    "kestrel": "Kestrel",
    "lapwing": "Lapwing",
    "lesser_black_backed_gull": "Lesser Black-backed Gull",
    "lesser_spotted_woodpecker": "Lt. Spotted Woodpecker",
    "linnet": "Linnet",
    "little_owl": "Little Owl",
    "long_tailed_tit": "Long-tailed Tit",
    "magpie": "Magpie",
    "mallard": "Mallard",
    "marsh_tit": "Marsh Tit",
    "meadow_pipit": "Meadow Pipit",
    "mistle_thrush": "Mistle Thrush",
    "moorhen": "Moorhen",
    "mute_swan": "Mute Swan",
    "nuthatch": "Nuthatch",
    "pheasant": "Pheasant",
    "pied_wagtail": "Pied Wagtail",
    "raven": "Raven",
    "red_kite": "Red Kite",
    "red_legged_partridge": "Red-legged Partridge",
    "redpoll_lesser": "Lesser Redpoll",
    "redwing": "Redwing",
    "reed_bunting": "Reed Bunting",
    "ring_necked_parakeet": "Ring-necked Parakeet",
    "robin": "Robin",
    "rook": "Rook",
    "siskin": "Siskin",
    "skylark": "Skylark",
    "song_thrush": "Song Thrush",
    "sparrowhawk": "Sparrowhawk",
    "starling": "Starling",
    "stock_dove": "Stock Dove",
    "tawny_owl": "Tawny Owl",
    "tree_sparrow": "Tree Sparrow",
    "treecreeper": "Treecreeper",
    "waxwing": "Waxwing",
    "willow_tit": "Willow Tit",
    "wood_pigeon": "Wood Pigeon",
    "wren": "Wren",
    "yellowhammer": "Yellowhammer",
}


def species_label(sp: str) -> str:
    return COMMON_NAMES.get(sp, sp.replace("_", " ").title())


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config_defaults(config_path: str | None) -> dict[str, Any]:
    """Read paths.output_dir (with run.output_dir fallback) from a YAML config."""
    if config_path is None:
        return {}
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    paths = raw.get("paths", {})
    run = raw.get("run", {})
    out: dict[str, Any] = {}
    output_dir = paths.get("output_dir") or run.get("output_dir")
    if output_dir:
        out["results_dir"] = output_dir
    if "uk_shp" in paths:
        out["uk_shp"] = paths["uk_shp"]
    return out


# ── Figure layout helpers (mirrors plotting.py private functions) ─────────────

def panel_chip(ax: plt.Axes, text: str, style: Any) -> None:
    ax.text(
        0.02, 1.01, text,
        transform=ax.transAxes, va="bottom", ha="left", clip_on=False,
        bbox={
            "boxstyle": "round,pad=0.22,rounding_size=0.14",
            "facecolor": style.PAPER,
            "edgecolor": "none",
        },
        **style.FONT_SECTION,
    )


def header_block(fig: plt.Figure, title: str, subtitle: str, style: Any) -> None:
    fig.text(0.06, 0.975, title, ha="left", va="top", **style.FONT_TITLE)
    fig.text(0.06, 0.928, subtitle, ha="left", va="top", **style.FONT_SUBTITLE)
    fig.add_artist(
        mpl.lines.Line2D(
            [0.06, 0.94], [0.905, 0.905],
            transform=fig.transFigure, color=style.GRIDLINE, linewidth=0.8,
        )
    )


def footer_note(fig: plt.Figure, text: str, style: Any) -> None:
    fig.text(0.94, 0.03, text, ha="right", va="center", **style.FONT_ANNOTATION)


def style_panel_axis(ax: plt.Axes, style: Any) -> None:
    """Apply the standard non-map panel style (no top/right spine, gridlines)."""
    ax.set_facecolor(style.PAPER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(style.GRIDLINE)
    ax.spines["bottom"].set_color(style.GRIDLINE)
    ax.tick_params(colors=style.TEXT_MUTED, labelsize=8.5)
    ax.grid(axis="y", color=style.GRIDLINE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
