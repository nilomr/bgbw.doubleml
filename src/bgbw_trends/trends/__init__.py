"""DML trend estimation module.

Primary entrypoint is the unified two-part pipeline in
``bgbw_trends.pipeline.train_predict_species``, which fits the marginal,
propensity and causal forests exactly once per species and returns both
the SDM and the τ(W) grid (Fink et al. 2023, §3.2).

The modules below are retained for supporting roles:

- ``residual_confounding`` — null-trend simulation diagnostic
  (Fink §2.2, vignette §2.2).
- ``ensemble`` — subsample aggregation for grid-level CATE uncertainty.
"""

from bgbw_trends.trends.residual_confounding import (
    diagnose_residual_confounding,
    ResidualConfoundingResult,
)
from bgbw_trends.trends.ensemble import fit_ensemble_trend, EnsembleTrendResult

__all__ = [
    "diagnose_residual_confounding",
    "ResidualConfoundingResult",
    "fit_ensemble_trend",
    "EnsembleTrendResult",
]
