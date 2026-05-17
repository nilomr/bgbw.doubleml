"""Configuration and paths for the BGBW trends pipeline."""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

# ── Data paths ──────────────────────────────────────────────────────────────
# All five paths can be overridden via environment variables.  The hard-coded
# values below are the originating-machine defaults; they will be wrong on any
# other host.  Set the corresponding env vars (e.g. in a shell rc file or a
# wrapper script) before running the pipeline.
#
#   BGBW_HARMONIZED_DIR      — individual-level BGBW submissions (harmonized/)
#   BGBW_AEF_DIR             — geospatial embedding parquets
#   BGBW_WEATHER_PATH        — survey-weekend HadUK-Grid parquet
#   BGBW_JANUARY_WEATHER_PATH — January HadUK-Grid parquet
HARMONIZED_DIR = Path(
    os.environ.get(
        "BGBW_HARMONIZED_DIR",
        "/home/nilomr/projects/bgbw.data.pipeline-r/output/data/harmonized",
    )
)
AEF_DIR = Path(
    os.environ.get("BGBW_AEF_DIR", "/mnt/w/projects/BGBW/data/metadata/spatial/aef")
)
WEATHER_PATH = Path(
    os.environ.get(
        "BGBW_WEATHER_PATH",
        "/mnt/w/projects/BGBW/data/metadata/weather/bgbw_weather.parquet",
    )
)
JANUARY_WEATHER_PATH = Path(
    os.environ.get(
        "BGBW_JANUARY_WEATHER_PATH",
        "/mnt/w/projects/BGBW/data/metadata/weather/bgbw_january_weather.parquet",
    )
)

# ── Species ─────────────────────────────────────────────────────────────────
SPECIES_COLUMNS = [
    "barn_owl",
    "black_headed_gull",
    "black_redstart",
    "blackbird",
    "blackcap",
    "blue_tit",
    "brambling",
    "bullfinch",
    "buzzard",
    "carrion_crow",
    "chaffinch",
    "chiffchaff",
    "coal_tit",
    "collared_dove",
    "common_gull",
    "corn_bunting",
    "dunnock",
    "feral_pigeon",
    "fieldfare",
    "goldcrest",
    "goldfinch",
    "great_black_backed_gull",
    "great_spotted_woodpecker",
    "great_tit",
    "green_woodpecker",
    "greenfinch",
    "grey_heron",
    "grey_partridge",
    "grey_wagtail",
    "hawfinch",
    "herring_gull",
    "hooded_crow",
    "house_sparrow",
    "jackdaw",
    "jay",
    "kestrel",
    "lapwing",
    "lesser_black_backed_gull",
    "lesser_spotted_woodpecker",
    "linnet",
    "little_owl",
    "long_tailed_tit",
    "magpie",
    "mallard",
    "marsh_tit",
    "meadow_pipit",
    "mistle_thrush",
    "moorhen",
    "mute_swan",
    "nuthatch",
    "pheasant",
    "pied_wagtail",
    "raven",
    "red_kite",
    "red_legged_partridge",
    "redpoll_lesser",
    "redwing",
    "reed_bunting",
    "ring_necked_parakeet",
    "robin",
    "rook",
    "siskin",
    "skylark",
    "song_thrush",
    "sparrowhawk",
    "starling",
    "stock_dove",
    "tawny_owl",
    "tree_sparrow",
    "treecreeper",
    "waxwing",
    "willow_tit",
    "wood_pigeon",
    "wren",
    "yellowhammer",
]

# ── Feature definitions ─────────────────────────────────────────────────────
# Feature split used in this project: features are partitioned into
# X_hab (habitat/spatial — varied at SDM predict time) and X_effort
# (observation process — held fixed at reference effort for SDM, but at
# actual values for DML nuisance). Both go into X for DML.

# Habitat / spatial features (X_hab) — stable properties of the location
HABITAT_FEATURES = [
    "garden_size",  # Small / Medium / Large — stable garden property
    "nearest_farmland",  # distance category — stable
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
    "urban_rural",
    "latitude",
    "longitude",
]

# Effort / observation-process features (X_effort) — vary with the count event
# and are held at reference values for SDM prediction.
#
# These are cell-level summaries computed during aggregation (see aggregate.py).
# Following Johnston 2025 / Fink 2023: retain rich effort summaries at the
# grid-cell level so the propensity model can capture interannual confounding
# in observer demographics and survey mode.
EFFORT_FEATURES = [
    "n_surveys",  # sampling intensity per cell-year
    "mean_observers",  # mean total observers (adults + children) per survey
    "proportion_adult",  # proportion of total observers who are adults
    "proportion_online",  # proportion of surveys submitted online (vs paper)
    # Survey-weekend weather — ambient detection conditions on the day.
    # X_effort only; frozen at median for SDM; NOT in W (see DEVELOPER_GUIDE §4.3).
    "mean_temp_c",  # mean daily temperature across the survey weekend (°C)
    "total_rain_mm",  # total rainfall across the survey weekend (mm)
    # Pre-survey January weather — accumulated winter conditions before the weekend.
    # Captures year-level confounding: cold/wet Januaries depress bird activity
    # and observer motivation independently of conditions on the survey weekend.
    # X_effort only; frozen at median for SDM; NOT in W (see DEVELOPER_GUIDE §4.3).
    "jan_mean_temp_c",  # mean daily temperature over non-survey January days (°C)
    "jan_total_rain_mm",  # cumulative rainfall over non-survey January days (mm)
]

# All confounders for DML (X = X_hab + X_effort)
ALL_CONFOUNDER_COLUMNS = HABITAT_FEATURES + EFFORT_FEATURES

# ── Spatial effect-modifier features (W) ────────────────────────────────────
# The actual W column list is built at predict time by
# ``features.habitat.make_spatial_features_W``, which globs ``emb_*`` so it
# stays in sync with whatever PCA dimension the run is using.


# ── Reference effort configuration (§4.4) ──────────────────────────────────
# Used by SDM at predict time: freeze effort to these values so the map
# reflects habitat, not sampling intensity. There are no defaults: every
# field must be set explicitly, normally via
# ``features.effort.compute_reference_effort(df)``. NaN defaults make any
# accidental use surface immediately in ``_build_prediction_effort`` rather
# than silently producing maps from arbitrary fallback values.
@dataclass
class ReferenceEffort:
    """Reference-effort values for SDM prediction.

    Values correspond to the aggregated effort features (see EFFORT_FEATURES).
    All fields default to NaN — populate via ``compute_reference_effort`` or
    construct with explicit values.
    """

    n_surveys: float = math.nan
    mean_observers: float = math.nan
    proportion_adult: float = math.nan
    proportion_online: float = math.nan
    mean_temp_c: float = math.nan  # survey-weekend mean temperature (°C)
    total_rain_mm: float = math.nan  # 3-day rainfall total (mm)
    jan_mean_temp_c: float = math.nan  # January mean temperature (°C)
    jan_total_rain_mm: float = math.nan  # cumulative January rainfall (mm)


@dataclass(frozen=True)
class OutputPaths:
    """Canonical directory layout for analysis artifacts."""

    root: Path
    predictions_dir: Path
    figures_dir: Path
    diagnostics_dir: Path
    ensemble_dir: Path
    maps_dir: Path
    shap_dir: Path
    rc_dir: Path

    def species_prediction_file(self, species: str) -> Path:
        return self.predictions_dir / species / "prediction.parquet"

    def species_figures_dir(self, species: str) -> Path:
        return self.figures_dir / "species" / species

    def species_sdm_figure(self, species: str) -> Path:
        return self.species_figures_dir(species) / "sdm.png"

    def species_trend_figure(self, species: str, metric: str) -> Path:
        return self.species_figures_dir(species) / f"trend_{metric}.png"

    def species_trend_ppy_panels_figure(self, species: str) -> Path:
        return self.species_figures_dir(species) / "trend_ppy_panels.png"

    def species_distribution_summary_figure(self, species: str) -> Path:
        return self.species_figures_dir(species) / "distribution_summary.png"

    def species_yearly_trend_figure(self, species: str) -> Path:
        return self.species_figures_dir(species) / "yearly_trend.png"

    def species_raw_vs_fitted_figure(self, species: str) -> Path:
        return self.species_figures_dir(species) / "raw_vs_fitted.png"

    def species_ensemble_file(self, species: str) -> Path:
        return self.ensemble_dir / species / "ensemble.parquet"

    def species_shap_figure(self, species: str, which: str) -> Path:
        """SHAP figure path. ``which`` is e.g. ``sdm_occ``, ``sdm_count``,
        ``cate_occ``, ``cate_count``; also ``_bar`` / ``_beeswarm`` suffixes."""
        return self.species_figures_dir(species) / f"shap_{which}.png"

    def species_shap_data(self, species: str, which: str) -> Path:
        """Long-format SHAP parquet. ``which`` is ``sdm`` or ``cate``."""
        return self.shap_dir / species / f"{which}.parquet"

    def species_rc_dir(self, species: str) -> Path:
        return self.rc_dir / species

    def species_rc_adjustment_file(self, species: str) -> Path:
        return self.species_rc_dir(species) / "adjustment.json"

    def species_rc_simulations_file(self, species: str) -> Path:
        return self.species_rc_dir(species) / "simulations.parquet"


# ── Run configuration ───────────────────────────────────────────────────────


@dataclass
class RunConfig:
    """Configuration for a single analysis run."""

    species: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=lambda: list(range(2015, 2020)))
    embedding_years: list[int] | None = None  # None → all available years in AEF_DIR
    output_dir: Path = Path("output")

    # Embedding PCA: reduce 128-dim embeddings to n_pca_components before
    # fitting any model (Fink et al. style). Set to 0 to disable.
    n_pca_components: int = 15

    n_estimators: int = 500
    # grf's default for regression forests is 5. With BGBW's ~1-2 surveys
    # per cell and rare detections, 10-20 gives smoother nuisance fits.
    rf_min_samples_leaf: int = 10

    # Causal Forest settings (econml)
    cf_n_estimators: int = 500
    cf_min_samples_leaf: int = 20

    # SDM settings
    sdm_n_estimators: int = 500
    # Reference effort for SDM prediction. Must be populated before any
    # call into ``train_predict_species`` (typically via
    # ``compute_reference_effort(df)``). Carrier slot only — pipeline
    # functions take ``reference_effort`` as a positional argument.
    reference_effort: ReferenceEffort = field(default_factory=ReferenceEffort)

    # Parallelization
    # ``n_workers``  — outer (process-level) parallelism for the ensemble and
    # residual-confounding loops. Each worker fits the full two-part pipeline.
    # 1 = sequential. -1 = all cores. On a workstation, prefer 1 (the inner
    # forests already use all threads). On a cluster node, set ``n_workers``
    # to the number of pipeline replicas you can run concurrently and
    # ``inner_n_jobs`` to ``cores // n_workers`` to avoid CPU thrashing.
    n_workers: int = 1
    # ``inner_n_jobs`` — threads per individual RF / CausalForest fit.
    # -1 = all cores (sklearn default). When ``n_workers > 1``, set to
    # ``max(1, cores // n_workers)`` to give each replica its own slice
    # of the machine and avoid oversubscription.
    inner_n_jobs: int = -1

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def output_paths(self) -> OutputPaths:
        """Create and return canonical output directories for this run."""
        paths = OutputPaths(
            root=self.output_dir,
            predictions_dir=self.output_dir / "predictions" / "species",
            figures_dir=self.output_dir / "figures",
            diagnostics_dir=self.output_dir / "diagnostics",
            ensemble_dir=self.output_dir / "ensemble" / "species",
            maps_dir=self.output_dir / "figures" / "maps",
            shap_dir=self.output_dir / "shap" / "species",
            rc_dir=self.output_dir / "rc" / "species",
        )
        dirs = [
            paths.root,
            paths.predictions_dir,
            paths.figures_dir,
            paths.diagnostics_dir,
            paths.ensemble_dir,
            paths.maps_dir,
            paths.figures_dir / "species",
            paths.shap_dir,
            paths.rc_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        return paths
