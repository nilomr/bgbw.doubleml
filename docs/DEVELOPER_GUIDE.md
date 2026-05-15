# BGBW Trends — Developer Guide

Reference guide for the model design and extension points in the pipeline.
Read this before modifying [src/bgbw_trends/](../src/bgbw_trends/). For
setup, configuration, and running the pipeline, see the
[User Guide](USER_GUIDE.md). Citations below refer chiefly to Fink et al.
(2023) and the Johnston et al. (2025) *Science* supplementary materials
(see §8 for full references).

---

## 1. What the pipeline does

For each species, the pipeline fits a **two-part (hurdle) DML model** on
cell-year rows: an **occurrence** stage (was the species detected?) and a
**count-given-occurrence** stage (`log1p(count)`). Each stage fits three
forests — marginal response `m̂(X)`, propensity `ŝ(X)` (year given X), and a
causal forest `τ(W)` — and derives three products in one pass:

1. **Distribution** — `P(occurrence | X, ref effort)` at the grid.
2. **Relative abundance** — `P(occ) · E[count | occ, ref effort]`.
3. **Trend** — per-year rate of change via the chain rule
   `abd_ppy = τ_occ(W)/P(occ) + τ_count(W)` evaluated on the prediction grid
   (Fink et al. 2023, §3.2).

Implementation: [pipeline.py](../src/bgbw_trends/pipeline.py) is a close port
of the `trendswf` R package (`train_predict_trend`), using `econml` in
place of `grf`.

---

## 2. Framework variables

| Symbol | What | BGBW content | Where it lives |
|---|---|---|---|
| **Y** | response | 0/1 occurrence, or `log1p(count)` | `features/response.py` |
| **T** | treatment | calendar year (continuous, not one-hot) | `pipeline.py` (built inline as `df_occ["year"].to_numpy()`) |
| **X** | confounders | habitat + effort — "effects constant across years" (Fink et al. 2023, §2.1) | `features/habitat.py` + `features/effort.py` |
| **W** | effect modifiers | embeddings + lat/lon | `features/habitat.py:make_spatial_features_W` |
| **τ(W)** | CATE | spatially varying annual rate of change | `pipeline.py` (via `econml.grf.CausalForest`) |

"Effects constant across years" means the functional form `μ(X)` is
year-invariant, not that feature values are static. Values in X should vary by
year when the underlying quantity does; yearly land cover is the obvious
example. Static features such as garden size are a special case, not the rule.

---

## 3. Feature state

Current state of each feature class and the intended target.

| Feature class | Current behaviour | Target |
|---|---|---|
| Embeddings | multi-year mean across all available AEF snapshots ([data.py:340](../src/bgbw_trends/data.py#L340) via `load_mean_embeddings`) | one embedding per (`bng_ref`, `year`) |
| lat/lon | constant per cell ✓ | constant per cell ✓ |
| Garden structure (lawn, pond, …) | aggregated away in [aggregate.py](../src/bgbw_trends/aggregate.py) (per-survey, not cell-year) | optional cell-year aggregate; currently omitted |
| Effort (`n_surveys`, `mean_observers`, `proportion_adult`, `proportion_online`, weather) | per (`bng_ref`, `year`) ✓ | per (`bng_ref`, `year`) ✓ |
| Short-term temporal (day-of-year, time-of-day) | absent — BGBW is a single weekend, no intra-year variation to model | n/a for BGBW |

### 3.1 Embeddings: current approach and roadmap

The current code averages AEF embeddings across all available years
(`load_mean_embeddings`, [data.py:127](../src/bgbw_trends/data.py#L127)), then
joins on `bng_ref` alone. This reduces sensitivity to artefacts in any single
snapshot, such as phenology or cloud cover, while preserving the stable spatial
signal of landscape composition. In the current model, each cell therefore has
one fixed embedding vector across all years.

Per-year embeddings would require:

1. **Yearly AEF parquet files** at `AEF_DIR/aef_bng1km_{year}_uk.parquet`.
2. **A stacking loader** — replace `load_mean_embeddings` with a loader that
   returns a frame with a `year` column, one row per `(bng_ref, year)`.
3. **Change the join** in
   [data.py:378 `assemble_analysis_data`](../src/bgbw_trends/data.py#L378)
   from `on="bng_ref"` to `on=["bng_ref", "year"]`.
4. **`embedding_years`** in [`RunConfig`](../src/bgbw_trends/config.py#L234)
   is already `list[int] | None` — repurpose from "years to average over" to
   "years to load".
5. **Grid prediction**: [_build_grid](../scripts/pipeline/run_prototype.py#L81)
   takes the most recent row per `bng_ref`, which is appropriate for a
   current-distribution map. Per-year distribution maps would need the grid
   filtered to the target year.
6. **PCA**: [reduce_embeddings_pca](../src/bgbw_trends/features/habitat.py#L41)
   must be fit on the stacked multi-year frame so the basis captures
   inter-annual variation.

### 3.2 What to do in the meantime

With a span of several years and mean embeddings, inter-annual land-cover
change is small relative to the observation-process confounding the effort
features already absorb. Treat this as an explicit modelling assumption and use
the residual-confounding diagnostic
([trends/residual_confounding.py](../src/bgbw_trends/trends/residual_confounding.py))
to bound the likely bias.

---

## 4. Feature roles

Every new variable needs an explicit functional role. Getting this wrong
silently changes what the SDM and trend maps mean.

### 4.1 The four roles

```
X_hab       X_effort       W         (no role)
────────    ────────────   ───────   ──────────────────────────────────────────
habitat     observation    trend     pure response driver with no
/landscape  process        effect    confounding role — makes the
properties  properties     modifier  model worse if included; leave out
────────    ────────────   ───────   ──────────────────────────────────────────
vary at     FROZEN at      only in   —
SDM time    reference      causal
            values         forest
```

**X_hab** are stable or slow-changing properties of the 1 km cell that describe
where species live. They absorb geographic confounding — if observer
participation is spatially biased across years, habitat differences would leak
into T without them. At SDM prediction time, X_hab varies freely: the SDM asks
"given this landscape, what is P(occurrence)?" Examples include embeddings,
lat/lon, and urban/rural class.

**X_effort** are properties of *how the cell was surveyed that year*. They
absorb temporal confounding — if later years attract systematically different
observers or effort levels, the naive Y∼T slope picks that up as a trend
signal. For the SDM, effort features are frozen to reference (median) values so
the map reflects what a standardised survey would find. Examples: `n_surveys`,
`mean_observers`, `proportion_online`, weather covariates.

**W** is the feature space in which the causal forest estimates τ(W). The model
learns from data how the trend differs between cells with different W values —
it does not impose variation, it discovers it. Including a feature in W gives
the trend surface the capacity to vary along that dimension; whether it actually
does is an empirical outcome. Embeddings + lat/lon give the map its spatial
resolution: the causal forest partitions the landscape into regions with similar
W and fits a local trend slope for each. Keep short-term weather out of W —
year-to-year weather fluctuations are temporal noise, not stable spatial
gradients, and including them would fold weather-correlated variance into the
trend surface.

### 4.2 Classifying a new variable

Work through the questions below in order; a variable can have multiple roles.

**Does it vary year-to-year and correlate with survey participation or
reporting rates?** If yes, include it in X for both the propensity and
marginal models to prevent confounding. If it also tracks the observation
process (intensity, weather) rather than landscape character, it belongs in
X_effort and gets frozen for SDM.

**Does it describe ambient detection conditions at survey time** (time-of-day,
survey-weekend weather) rather than fixed landscape properties? If yes, it is
effort-adjacent and should be frozen for the SDM.

Weather covariates are frozen to the dataset-wide **median** for SDM
prediction. The reference implementation also supports conditioning the SDM on
optimal detection conditions by identifying the covariate values at the 80–90th
percentile of predicted response (`multivariate_maximize` in `trendswf`)
— but this requires multiple independent observations per cell within the year.
BGBW cells have one aggregate per year (all surveys from the survey weekend,
combined), so there is no within-cell distribution of detection conditions to
sample from.

**Is this a stable spatial property along which the trend might genuinely
differ between cells?** If yes, include it in **W**
([make_spatial_features_W](../src/bgbw_trends/features/habitat.py#L201)).
The causal forest will estimate τ as a function of it and learn from the data
whether and how much the trend varies along that dimension. Keep it in X too
if either of the previous questions was yes.

**Is it a fixed geographic property with no link to observation year?** (e.g.
bedrock geology, long-term climate normal) It goes in X_hab if geographic
variation in it would confound the distribution model. It goes in W if trends
plausibly differ along this dimension. Does not need to be frozen for SDM.

### 4.3 Weather as a worked example

Survey-weekend temperature and rainfall, and pre-survey January conditions, are
already implemented. The table is a useful reference for any future
detection-condition covariate.

| Question | Answer | Consequence |
|---|---|---|
| Varies year-to-year and correlated with participation? | Yes — harsh January weather → fewer reports, depressed counts | Goes in **X** (both propensity and marginal) |
| Describes ambient detection conditions? | Yes — cold/wet weather suppresses bird activity and observer effort | X_effort; frozen for SDM |
| Does the trend plausibly vary along weather gradients? | No — year-to-year weather is temporal noise, not a stable spatial gradient | Do not add to W |
| Can we maximise it for SDM? | No — one aggregate per (cell, year), no within-cell replication | Freeze to **median** |

The four weather columns — `mean_temp_c`, `total_rain_mm`, `jan_mean_temp_c`,
`jan_total_rain_mm` — are in `EFFORT_FEATURES`
([config.py:133](../src/bgbw_trends/config.py#L133)) with corresponding fields
in `ReferenceEffort` ([config.py:169](../src/bgbw_trends/config.py#L169)) set
to UK-median defaults. `compute_reference_effort` overrides those defaults from
the training data at runtime.

### 4.4 Long-term climate normals as a worked example

A 30-year mean temperature per cell is fixed in time, so it carries no
confounding role across years. It describes landscape character rather than
detection conditions, so it goes in X_hab without being frozen for SDM. If
trends differ between climate zones — populations in colder regions declining
faster, say — it also goes in W so the trend surface can resolve that gradient.
It does not go in `ReferenceEffort`.

### 4.5 Garden structural features (lawn, pond, garden_size, …)

These are per-survey individual-level variables, not currently aggregated to
the 1 km BNG grid. They live in `_NON_SPECIES_COLS` in
[aggregate.py](../src/bgbw_trends/aggregate.py#L33) and are dropped during
groupby. To use them, compute a cell-level summary per year (proportion of
surveys with a pond, modal garden size code, mean proportion with lawn). These
describe the kind of garden environment represented in the cell, which makes
them closer to X_hab than X_effort. They should not be frozen for SDM, and
since urban gardens
have become more likely to submit online over time, they may also act as
confounders — which they would already handle if included in X_hab.

---

## 5. Known gaps

Ordered roughly by impact.

1. **Embeddings are not per-year** ([data.py:340](../src/bgbw_trends/data.py#L340)).
   See §3.1. The current model uses a multi-year mean rather than year-specific
   embeddings, which reduces sensitivity to any one year's artefacts but keeps
   each cell's representation fixed across the study period.

2. **`config.HABITAT_FEATURES` lists per-survey columns that don't exist on the
   BNG path.** [config.py:107](../src/bgbw_trends/config.py#L107) lists
   `garden_size`, `lawn`, `pond`, `urban_rural`, etc. On the BNG path, X_hab
   is effectively **embeddings + lat/lon only**.
   [make_habitat_features](../src/bgbw_trends/features/habitat.py#L144)
   silently skips missing columns, which means the constant is misleading
   without being harmful. Either remove the per-survey entries from
   `HABITAT_FEATURES` or compute cell-level aggregates.

3. **No observer-skill covariate.** The single most important effort covariate
   in the eBird-based literature is a per-observer skill index (CCI). BGBW has
   no analogue; `proportion_adult` and `proportion_online` are weak proxies. A
   participant-level "repeat surveyor" indicator derived from the harmonized
   data and aggregated to cells would be a meaningful improvement.

4. **`ReferenceEffort` defaults are live until `compute_reference_effort` is
   called.** [config.py:169](../src/bgbw_trends/config.py#L169) sets
   `n_surveys=1.0`, `mean_observers=1.3`, etc. as hard-coded fallbacks.
   The ensemble path ([ensemble.py:170](../src/bgbw_trends/trends/ensemble.py#L170))
   and the prototype script both call `compute_reference_effort`. A new caller
   who skips this step will get SDM maps derived from the wrong effort
   reference without warning.

5. **`config.SPATIAL_FEATURES_W` is dead code.**
   [config.py:157](../src/bgbw_trends/config.py#L157) hard-codes `emb_0..63`,
   but after PCA the columns are `emb_0..{k-1}`.
   [make_spatial_features_W](../src/bgbw_trends/features/habitat.py#L201)
   globs `emb_*` so the constant is never used. Delete it or wire it through
   as the authoritative list.

6. **`PROTOTYPE_SPECIES` is misleading.** `PROTOTYPE_SPECIES = ["blue_tit",
   "house_sparrow"]` ([config.py:20](../src/bgbw_trends/config.py#L20))
   exists alongside 74 species in `SPECIES_COLUMNS`, but the prototype script
   configures its species list inline and doesn't use this constant. Either
   remove it or make it the single source of truth for prototype runs.

7. **`RunConfig.n_folds` and `RunConfig.n_rep` are dead code.**
   [config.py:247](../src/bgbw_trends/config.py#L247) defines these fields
   (defaults 5 and 5) and [console.py:230](../src/bgbw_trends/console.py#L230)
   prints them as "DML folds: 5 × 5 reps", but `train_predict_species` never
   reads them. They were added with the intent of implementing Chernozhukov
   et al.'s repeated K-fold cross-fitting, which was never built. The nuisance
   forests use sklearn OOB predictions instead — the direct Python equivalent
   of what `grf::regression_forest` does internally (a single forest fit whose
   per-sample predictions are debiased via honest/OOB splitting, not explicit
   K-fold). The fields and the console line should be removed to avoid implying
   that cross-fitting is configured and running.

8. **`covariates_to_maximize` is not ported.** The reference implementation
   can condition SDM predictions on optimal detection conditions
   (`multivariate_maximize` in `trendswf`; Fink et al. 2023, §3.3).
   BGBW has no within-cell, within-year detection variation to draw from,
   so this is a correct omission.

---

## 6. How to add a new feature / covariate

Use §4 to decide the role, then:

- **X_hab** → add to `make_habitat_features` in [features/habitat.py](../src/bgbw_trends/features/habitat.py#L144). Must be present in both training frame and prediction grid.
- **X_effort** → add the column name to `EFFORT_FEATURES` in [config.py](../src/bgbw_trends/config.py#L133), add a field to `ReferenceEffort` ([config.py:169](../src/bgbw_trends/config.py#L169)), and `compute_reference_effort` picks it up automatically via `fields()`.
- **W** → add to `make_spatial_features_W` in [features/habitat.py](../src/bgbw_trends/features/habitat.py#L201). Grid-only; not frozen.
- **Observer-skill analogue** → add to X_effort but set its `ReferenceEffort` field **above the median** to represent a skilled-but-typical observer. The exact quantile to use is not pinned in the reference implementation (`prediction_unit` in `trendswf` is passed in externally).

Both the marginal and propensity models receive the same X by default
(X_hab + X_effort). This is the standard DML strategy — Fink et al. (2023,
§4) recommend including all features from the conditional mean model in the
propensity model as well. If you ever need to give the propensity a different
feature set (for instance, a feature that would introduce identifiability
problems in the marginal model), `train_predict_species` accepts separate
`propensity_covariates` and `marginal_covariates` arguments that override
`_assemble_X`.

After adding: run the full script on one species, inspect the diagnostic
summary, and run the residual-confounding diagnostic to confirm the new feature
reduces (or at least does not increase) the null-trend bias.

---

## 7. How to add a new diagnostic or plot

Each [`SpeciesPrediction`](../src/bgbw_trends/pipeline.py#L209) carries the
per-row PLR components (`obs_occ`, `obs_count`). With `return_models=True` you
also get a [`TrainedSpeciesModels`](../src/bgbw_trends/pipeline.py#L182)
with the fitted forests. Together these support:

- Yearly raw vs. fitted trend lines (deconfounded via
  `raw.mean() + ate*(year - year.mean())`).
- SHAP attributions for SDM drivers and for τ(W) via a surrogate forest —
  see [interpret.py](../src/bgbw_trends/interpret.py).
- Grid-cell variance via
  [`predict_grid_variance`](../src/bgbw_trends/pipeline.py#L304), which is
  much cheaper than turning on `compute_variance=True` for every cell.

New plots go in [plotting.py](../src/bgbw_trends/plotting.py); new diagnostics
go in [trends/](../src/bgbw_trends/trends/) next to `residual_confounding.py`.
Pull data off the prediction object rather than recomputing.

The propensity R² ([`SpeciesModelDiagnostics`](../src/bgbw_trends/pipeline.py#L59))
is the most informative single diagnostic from a pipeline run. High R² (> 0.85)
suggests the feature set is absorbing strong temporal confounding; near-zero
R² means either little confounding in the data or a missing confounder. See the
[User Guide §7](USER_GUIDE.md#7-diagnostics) for interpretation guidance.

---

## 8. Reference materials

1. Fink, D., Johnston, A., et al. (2023). *A Double machine learning trend
   model for citizen science data*. Methods in Ecology and Evolution, 14,
   2435–2448. https://doi.org/10.1111/2041-210X.14186 — §2.1 and §3.2–§3.3
   cover DML theory and the chain-rule trend derivation.
2. Johnston, A., et al. (2025). *North American bird declines are greatest
   where species are most abundant*. Science, 388, 532.
   https://doi.org/10.1126/science.adn4381 — supplementary Materials and
   Methods §2.3 and §3.1 describe the production feature set and annual
   land-cover wiring.
3. `trendswf` R package — the reference implementation loosely mirrored by the Python
   pipeline. The core function is `train_predict_trend`; the
   `motivating-example` vignette walks through the residual-confounding
   diagnostic on a toy simulation.
4. [pipeline.py](../src/bgbw_trends/pipeline.py) — docstrings map back to the R implementation.
