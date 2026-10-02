"""Public API checks for unsupported outer evaluation and declared types."""

import json

import numpy as np
import pandas as pd
import pytest

from causal_emergence_discovery.adjustment import adjustment_sufficiency_audit
from causal_emergence_discovery.discovery import DiscoveryConfig, run_discovery
from causal_emergence_discovery.panel import build_lagged_table
from causal_emergence_discovery.spec import StudySpec


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(733)
    records = []
    for entity in range(14):
        previous = float(rng.normal())
        for time in range(10):
            x = float(rng.normal() + entity / 8)
            z = float(rng.normal())
            records.append({"entity": entity, "time": time, "x": x, "z": z, "y": previous})
            previous = 1.2 * x - 0.4 * z + float(rng.normal(scale=0.25))
    return pd.DataFrame(records)


def _panel_spec() -> StudySpec:
    return StudySpec.from_dict({
        "dataset": {"id_column": "entity", "time_column": "time"},
        "outcomes": ["y"],
        "columns": {
            "x": {"role": "context", "type": "numeric"},
            "z": {"role": "context", "type": "numeric"},
            "y": {"role": "outcome", "type": "numeric", "allowed_as_cause": False},
        },
    })


def test_unsupported_reserved_outer_report_is_strict_json():
    frame = _panel()
    spec = _panel_spec()
    config = DiscoveryConfig(
        outcome="y", lag=1, max_states=3, paths=1, branching_factor=1,
        folds=2, top_k=2, validation_mode="entity_holdout",
        holdout_fraction=0.2, seed=23,
    )
    baseline = run_discovery(frame, spec, config)
    lagged = build_lagged_table(frame, spec, outcomes=["y"], lag=1).data
    heldout_entities = set(lagged.iloc[baseline["validation"]["outer"]["test_indices"]]["entity"])
    missing_outer_labels = frame.copy()
    missing_outer_labels.loc[missing_outer_labels.entity.isin(heldout_entities), "y"] = np.nan

    report = run_discovery(missing_outer_labels, spec, config)

    assert report["outer_evaluation"]["status"] == "not_evaluable"
    assert report["outer_evaluation"]["macro_r2"] is None
    assert report["outer_evaluation"]["micro_r2"] is None
    assert report["outer_evaluation"]["predictive_r2_difference"] is None
    assert report["searches"]
    assert report["feature_schema"]["x"]["kind"] == "numeric"
    assert {
        "kind", "source", "invalid_outer_values", "unknown_outer_categories",
        "missing_outer_values",
    } <= set(report["feature_schema"]["x"])
    applications = report["outer_evaluation"]["feature_application"]
    assert {"macro_encoder", "macro_design", "micro_design"} <= set(applications)
    assert {"missing", "invalid_numeric", "unknown_category"} <= set(
        applications["macro_encoder"]["x"]
    )
    json.dumps(report, allow_nan=False)


def _adjustment_spec(*, treatment_type: str = "numeric") -> StudySpec:
    return StudySpec.from_dict({
        "dataset": {"id_column": "id", "time_column": "time"},
        "interventions": ["treatment"],
        "outcomes": ["outcome"],
        "columns": {
            "treatment": {"role": "intervention", "type": treatment_type},
            "u": {"role": "context", "type": "numeric"},
            "outcome": {"role": "outcome", "type": "numeric", "allowed_as_cause": False},
        },
        "adjustment": {
            "columns": ["u"],
            "rationale": "u is a measured pre-treatment common cause",
        },
    })


def _audit_coefficients(frame: pd.DataFrame, spec: StudySpec) -> list[dict]:
    return adjustment_sufficiency_audit(
        frame, spec, interventions=["treatment"], target_column="outcome"
    )["pathways"]


def test_declared_numeric_adjustment_handles_object_numeric_treatment_and_covariate():
    rng = np.random.default_rng(81)
    treatment = rng.normal(size=300)
    covariate = rng.normal(size=300)
    outcome = 2.4 * treatment - 1.7 * covariate + rng.normal(scale=0.15, size=300)
    numeric = pd.DataFrame({
        "id": np.arange(len(treatment)), "time": np.arange(len(treatment)),
        "treatment": treatment, "u": covariate, "outcome": outcome,
    })
    numeric_strings = numeric.copy()
    numeric_strings["treatment"] = numeric_strings["treatment"].map(lambda value: f"{value:.17g}")
    numeric_strings["u"] = numeric_strings["u"].map(lambda value: f"{value:.17g}")

    numeric_result = _audit_coefficients(numeric, _adjustment_spec())
    string_result = _audit_coefficients(numeric_strings, _adjustment_spec())

    assert string_result[0]["coefficient"] == pytest.approx(numeric_result[0]["coefficient"], abs=1e-12)
    assert string_result[0]["coefficient"] == pytest.approx(2.4, abs=0.03)


def test_declared_categorical_numeric_treatment_codes_have_no_numeric_slope():
    treatment = np.tile([0, 1], 100)
    covariate = np.linspace(-1.0, 1.0, len(treatment))
    frame = pd.DataFrame({
        "id": np.arange(len(treatment)), "time": np.arange(len(treatment)),
        "treatment": treatment, "u": covariate,
        "outcome": 5.0 * treatment + 1.3 * covariate,
    })

    pathway = _audit_coefficients(frame, _adjustment_spec(treatment_type="categorical"))[0]

    assert pathway["coefficient"] is None
    assert pathway["stderr"] is None
    assert "not represented as a single numeric term" in pathway["note"]
