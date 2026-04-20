"""Double Machine Learning trend models for BGBW citizen science data.

Three products:
- Distribution (where species occur): effort-adjusted SDM → bgbw_trends.sdm
- Relative abundance (expected count): effort-adjusted SDM → bgbw_trends.sdm
- Trends (year-on-year change, adjusted): DML → bgbw_trends.trends

Feature engineering is shared across both via bgbw_trends.features.
"""

__version__ = "0.1.0"

# cuML's Base class defines `_more_tags` without `__sklearn_tags__`, which
# causes sklearn 1.6's MRO walker to fall back to the legacy tag path and warn.
# Patch it once here so all cuML RF subclasses in the package are fixed.
try:
    from cuml.internals.base import Base as _CuMLBase
    from cuml.internals.mixins import TagsMixin as _TagsMixin

    if "__sklearn_tags__" not in vars(_CuMLBase):
        _CuMLBase.__sklearn_tags__ = _TagsMixin.__sklearn_tags__
    del _CuMLBase, _TagsMixin
except ImportError:
    pass
