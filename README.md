# bgbw.trends

Pipeline for estimating species distribution, relative abundance, and
annual trend from Big Garden Birdwatch (BGBW) citizen-science data.

## What it does

The method is based on a Double Machine Learning (DML) workflow inspired by Fink et al. (2023).
For each species, the pipeline fits a hurdle-style model with two stages:

1. Occurrence stage: whether the species is detected.
2. Count stage: expected count conditional on detection.

Both stages use DML residualization:

- Fit two nuisance models per stage:
	- a marginal response model m(X)
	- a propensity/treatment model e(X) for year
- Residualize outcome and treatment, then fit a Causal Forest on those residuals.
- Estimate spatially varying trend effects tau(W), where W includes embeddings and coordinates.

From one training pass, it produces three products:

- Distribution: `p_occurrence`
- Relative abundance: `e_abundance`
- Trend components: `occ_ppy`, `count_ppy`, and combined `abd_ppy`

The combined abundance trend follows the structure:

- occ_ppy = tau_occ(W) / P(occurrence | W, reference effort, etc.)
- abd_ppy = occ_ppy + count_ppy

This gives a per-year relative abundance change combining occurrence and count components.


## Main entry points

- `scripts/run_prototype.py`: run model training and write predictions/figures.
- `scripts/make_maps.py`: regenerate map figures from saved parquet outputs.

## Setup

```bash
uv sync
```

## Run

```bash
uv run python scripts/run_prototype.py
uv run python scripts/make_maps.py
```

## Outputs

By default, outputs are written under `output/prototype/` in a structured layout:

- `diagnostics/`
- `predictions/species/<species>/prediction.parquet`
- `figures/species/<species>/...`
- `figures/maps/<species>/...`
- `ensemble/species/<species>/ensemble.parquet`

## References

- Fink, D., Johnston, A., Strimas-Mackey, M., Auer, T., Hochachka, W. M., Ligocki, S., Oldham Jaromczyk, L., Robinson, O., Wood, C., Kelling, S., & Rodewald, A. D. (2023). A Double machine learning trend model for citizen science data. Methods in Ecology and Evolution, 14, 2435–2448. https://doi.org/10.1111/2041-210X.14186


## Tests

```bash
uv run pytest
```
