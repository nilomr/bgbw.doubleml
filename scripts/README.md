# Scripts Layout

This directory is organized by purpose.

## Pipeline

Canonical run flow for model fitting and standard maps:

- `pipeline/run_prototype.py`
- `pipeline/make_maps.py`

## Products

Additional map products generated from saved outputs:

- `products/make_hex_map.py`
- `products/make_decoupling_map.py`

## Diagnostics

Quality checks and feature/embedding diagnostics:

- `diagnostics/plot_effort_features.py`
- `diagnostics/plot_embedding_pca.py`
- `diagnostics/plot_embedding_pcs.py`

## Exploratory

Experimental or external-data analysis scripts:

- `exploratory/plot_google_aef_embedding.py`

## Usage

Run scripts from repository root with `uv`:

```bash
uv run python scripts/pipeline/run_prototype.py
uv run python scripts/pipeline/make_maps.py
uv run python scripts/products/make_hex_map.py
uv run python scripts/products/make_decoupling_map.py
```
