"""Double Machine Learning trend models for BGBW citizen science data.

Three products:
- Distribution (where species occur): effort-adjusted SDM → bgbw_trends.sdm
- Relative abundance (expected count): effort-adjusted SDM → bgbw_trends.sdm
- Trends (year-on-year change, adjusted): DML → bgbw_trends.trends

Feature engineering is shared across both via bgbw_trends.features.
"""

__version__ = "0.1.0"
