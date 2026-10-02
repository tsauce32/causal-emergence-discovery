"""Tools for discovering macro-causal structure in longitudinal tables."""

from causal_emergence_discovery.discovery import DiscoveryConfig, run_discovery
from causal_emergence_discovery.panel import LaggedDataset, build_lagged_table, validate_panel
from causal_emergence_discovery.scoring import MacroScore, PairedPredictiveScores, SCORE_SCHEMA_VERSION
from causal_emergence_discovery.spec import AdjustmentSpec, ColumnSpec, StudySpec, load_spec
from causal_emergence_discovery.validation import DataSplit, ValidationPlan, build_validation_plan

__version__ = "0.2.0"

__all__ = [
    "ColumnSpec",
    "AdjustmentSpec",
    "DataSplit",
    "DiscoveryConfig",
    "LaggedDataset",
    "MacroScore",
    "PairedPredictiveScores",
    "SCORE_SCHEMA_VERSION",
    "StudySpec",
    "ValidationPlan",
    "build_lagged_table",
    "build_validation_plan",
    "load_spec",
    "run_discovery",
    "validate_panel",
    "__version__",
]
