#!/usr/bin/env python3
"""Generate poster-quality maps from model output parquet files.

Reads any ``species_*.parquet`` files in a results directory and produces
trend and SDM map figures, without needing to re-run the model.

Usage
-----
    uv run python scripts/pipeline/make_maps.py                        # default output/prototype/
    uv run python scripts/pipeline/make_maps.py --results output/run1  # custom path
    uv run python scripts/pipeline/make_maps.py --species blackbird starling
    uv run python scripts/pipeline/make_maps.py --metric occ_ppy --dpi 150

Output
------
For each species found:
    <results_dir>/figures/maps/<species>/trend_<metric>.png
    <results_dir>/figures/maps/<species>/sdm.png

Each figure is ~3500 × 2600 px at 300 dpi — ready for print or web.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

# Ensure the package is importable when running as a script.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from bgbw_trends.viz.maps import plot_bivariate, plot_sdm, plot_trend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Sigma column corresponding to each metric
_SIGMA = {
    "occ_ppy": "occ_sigma",
    "count_ppy": "count_sigma",
    "abd_ppy": None,  # chain-rule sum has no direct sigma column
}

# Human-readable common names (extend as needed)
_COMMON_NAMES: dict[str, str] = {
    "blackbird": "Common Blackbird  —  Turdus merula",
    "blue_tit": "Blue Tit  —  Cyanistes caeruleus",
    "house_sparrow": "House Sparrow  —  Passer domesticus",
    "starling": "Common Starling  —  Sturnus vulgaris",
    "robin": "European Robin  —  Erithacus rubecula",
    "wren": "Eurasian Wren  —  Troglodytes troglodytes",
    "great_tit": "Great Tit  —  Parus major",
    "chaffinch": "Common Chaffinch  —  Fringilla coelebs",
    "wood_pigeon": "Common Wood Pigeon  —  Columba palumbus",
    "goldfinch": "European Goldfinch  —  Carduelis carduelis",
    "greenfinch": "European Greenfinch  —  Chloris chloris",
    "song_thrush": "Song Thrush  —  Turdus philomelos",
}


def _load_config_defaults(config_path: str | None) -> dict:
    if config_path is None:
        return {}
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    out: dict = {}
    if "run" in raw and "output_dir" in raw["run"]:
        out["results"] = raw["run"]["output_dir"]
    if "paths" in raw and "uk_shp" in raw["paths"]:
        out["uk_shp"] = raw["paths"]["uk_shp"]
    return out


def _parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    cfg = _load_config_defaults(pre_args.config)

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        default=None,
        help="pipeline YAML config file; sets --results and --uk-shp defaults",
    )
    p.add_argument(
        "--results",
        default=cfg.get("results", "output/prototype"),
        help="directory containing species prediction parquets (default: output/prototype)",
    )
    p.add_argument(
        "--species",
        nargs="+",
        default=None,
        help="species to map; defaults to all found in --results",
    )
    p.add_argument(
        "--metric",
        choices=["abd_ppy", "occ_ppy", "count_ppy"],
        default="abd_ppy",
        help="trend metric to map (default: abd_ppy)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="output resolution in dots per inch (default: 300)",
    )
    p.add_argument(
        "--pct-vmax",
        type=float,
        default=97.5,
        dest="pct_vmax",
        help="percentile of |trend| used to cap the colour scale (default: 97.5)",
    )
    p.add_argument(
        "--uk-shp",
        default=cfg.get("uk_shp"),
        help="path to UK boundary shapefile; read from --config if omitted",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    results_dir = Path(args.results)
    maps_dir = results_dir / "figures" / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    # Discover species from parquet files. Prefer the new hierarchy, but
    # retain compatibility with the original flat prototype outputs.
    hierarchical = sorted(results_dir.glob("predictions/species/*/prediction.parquet"))
    if hierarchical:
        found = {p.parent.name: p for p in hierarchical}
    else:
        flat = sorted(results_dir.glob("species_*.parquet"))
        found = {p.stem.removeprefix("species_"): p for p in flat}

    all_parquets = list(found.values())
    if not all_parquets:
        logger.error(
            "No species parquet files found in %s (looked in predictions/species/*/prediction.parquet and species_*.parquet)",
            results_dir,
        )
        sys.exit(1)

    species_list = args.species or list(found.keys())

    missing = [s for s in species_list if s not in found]
    if missing:
        logger.warning("Parquet files not found for: %s", ", ".join(missing))
    species_list = [s for s in species_list if s in found]
    if not species_list:
        logger.error("No valid species to map.")
        sys.exit(1)

    shp_kwargs = {"uk_shp": args.uk_shp} if args.uk_shp else {}
    sigma_col = _SIGMA.get(args.metric)

    logger.info("Mapping %d species → %s", len(species_list), maps_dir)

    for sp in species_list:
        df = pd.read_parquet(found[sp])
        label = _COMMON_NAMES.get(sp, sp.replace("_", " ").title())
        species_maps_dir = maps_dir / sp
        species_maps_dir.mkdir(parents=True, exist_ok=True)

        # ── Trend map ──────────────────────────────────────────────────
        out_trend = species_maps_dir / f"trend_{args.metric}.png"
        logger.info("[%s] trend map → %s", sp, out_trend)
        plot_trend(
            df,
            title=sp,
            species_label=label,
            metric=args.metric,
            sigma_col=sigma_col,
            pct_vmax=args.pct_vmax,
            output_path=out_trend,
            dpi=args.dpi,
            **shp_kwargs,
        )

        # ── SDM map ────────────────────────────────────────────────────
        if "p_occurrence" in df.columns and "e_abundance" in df.columns:
            out_sdm = species_maps_dir / "sdm.png"
            logger.info("[%s] SDM map → %s", sp, out_sdm)
            plot_sdm(
                df,
                title=sp,
                species_label=label,
                output_path=out_sdm,
                dpi=args.dpi,
                **shp_kwargs,
            )

        # ── Bivariate map ───────────────────────────────────────────────
        if args.metric in df.columns and "e_abundance" in df.columns:
            out_biv = species_maps_dir / f"bivariate_{args.metric}.png"
            logger.info("[%s] bivariate map → %s", sp, out_biv)
            plot_bivariate(
                df,
                title=sp,
                species_label=label,
                metric=args.metric,
                output_path=out_biv,
                dpi=args.dpi,
                **shp_kwargs,
            )

    logger.info("Done. Maps written to %s", maps_dir)


if __name__ == "__main__":
    main()
