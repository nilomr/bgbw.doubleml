"""Configuration and paths for the BGBW trends pipeline."""

from dataclasses import dataclass, field
from pathlib import Path

# ── Data paths ──────────────────────────────────────────────────────────────
HARMONIZED_DIR = Path(
    "/home/nilomr/projects/bgbw.data.pipeline-r/output/data/harmonized"
)
BNG_SURVEYS_DIR = Path(
    "/home/nilomr/projects/bgbw.data.pipeline-r/output/data/bng_surveys"
)
GEOTESSERA_DIR = Path("/mnt/w/projects/BGBW/data/metadata/tmp_geotessera")

# ── Species ─────────────────────────────────────────────────────────────────
PROTOTYPE_SPECIES = ["blue_tit", "house_sparrow"]

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
]

# All confounders for DML (X = X_hab + X_effort)
ALL_CONFOUNDER_COLUMNS = HABITAT_FEATURES + EFFORT_FEATURES

# ── Spatial effect-modifier features (W) ────────────────────────────────────
# Features that the trend τ(W) is allowed to vary across.
# Spatial effect modifiers: spatial features + lat/lon, NOT effort variables.
EMBEDDING_DIM = 128
SPATIAL_FEATURES_W = [f"emb_{i}" for i in range(EMBEDDING_DIM)] + [
    "latitude",
    "longitude",
]


# ── Reference effort configuration (§4.4) ──────────────────────────────────
# Used by SDM at predict time: freeze effort to these values so the map
# reflects habitat, not sampling intensity.
# Values should be set to median/modal across the dataset.
# Populated at runtime by features.effort.compute_reference_effort().
@dataclass
class ReferenceEffort:
    """Reference-effort values for SDM prediction.

    Values correspond to the aggregated effort features (see EFFORT_FEATURES).
    Overridden at runtime by compute_reference_effort().
    """

    n_surveys: float = 1.0
    mean_observers: float = 1.3
    proportion_adult: float = 0.9
    proportion_online: float = 0.7


@dataclass(frozen=True)
class OutputPaths:
    """Canonical directory layout for analysis artifacts."""

    root: Path
    predictions_dir: Path
    figures_dir: Path
    diagnostics_dir: Path
    ensemble_dir: Path
    maps_dir: Path

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


# ── Run configuration ───────────────────────────────────────────────────────


@dataclass
class RunConfig:
    """Configuration for a single analysis run."""

    species: list[str] = field(default_factory=lambda: list(PROTOTYPE_SPECIES))
    years: list[int] = field(default_factory=lambda: list(range(2015, 2020)))
    embedding_year: int = 2019
    output_dir: Path = Path("output")

    # Embedding PCA: reduce 128-dim embeddings to n_pca_components before
    # fitting any model (Fink et al. style). Set to 0 to disable.
    n_pca_components: int = 15

    # DML settings
    n_folds: int = 5
    n_rep: int = 5  # repeated cross-fitting; median aggregation (§5 Step 4)
    n_estimators: int = 500
    # grf's default for regression forests is 5. With BGBW's ~1-2 surveys
    # per cell and rare detections, 10-20 gives smoother nuisance fits.
    rf_min_samples_leaf: int = 10

    # Causal Forest settings (econml)
    cf_n_estimators: int = 500
    cf_min_samples_leaf: int = 20

    # SDM settings
    sdm_n_estimators: int = 500
    reference_effort: ReferenceEffort = field(default_factory=ReferenceEffort)

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
        )
        dirs = [
            paths.root,
            paths.predictions_dir,
            paths.figures_dir,
            paths.diagnostics_dir,
            paths.ensemble_dir,
            paths.maps_dir,
            paths.figures_dir / "species",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        return paths
