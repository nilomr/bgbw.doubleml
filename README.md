# bgbw.doubleml

Estimates species distribution, relative abundance, and annual population
trends from Big Garden Birdwatch (BGBW) survey data. For each species, a
two-part DML model — occurrence and count-given-occurrence — produces all
three outputs in one training pass.

## Quick start

```bash
uv sync
uv run python scripts/pipeline/run_prototype.py configs/local.yaml
```

All run parameters — data paths, species list, model settings, compute budget — live in a
YAML config file. Edit `configs/local.yaml` for local development or copy it to create a
new experiment config. A copy of the config is saved alongside outputs for reproducibility.
See the [User Guide](docs/USER_GUIDE.md) for details.

## Documentation

| | |
|---|---|
| [User Guide](docs/USER_GUIDE.md) | Data requirements, configuration, running scripts, outputs, adding species |
| [Developer Guide](docs/DEVELOPER_GUIDE.md) | Model design, feature roles, known gaps, extending the code |

## Tests

```bash
uv run pytest
```

## References

- Fink, D., Johnston, A., et al. (2023). A Double machine learning trend model for
  citizen science data. *Methods in Ecology and Evolution*, 14, 2435–2448.
  https://doi.org/10.1111/2041-210X.14186

- Johnston, A., et al. (2025). North American bird declines are greatest where
  species are most abundant. *Science*, 388, 532.
  https://doi.org/10.1126/science.adn4381
