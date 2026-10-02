import json

import numpy as np
import pandas as pd
import pytest

from causal_emergence_discovery.adjustment import adjustment_sufficiency_audit, resolve_adjustment
from causal_emergence_discovery.macro import (
    apply_macro,
    composite_quantile_macro,
    kmeans_macro,
    merge_macro_states,
    quantile_macro,
    refit_macro,
)
from causal_emergence_discovery.spec import StudySpec


def test_quantile_fit_freezes_cutpoints_for_extremes_and_unseen_rows():
    train = pd.DataFrame({"x": np.arange(12, dtype=float)})
    macro = quantile_macro(train, "x", 3)
    original = macro.labels.copy()
    validation = pd.DataFrame({"x": [1000.0, -1000.0, 5.0]}, index=["hi", "lo", "mid"])

    transformed = apply_macro(macro, validation)
    reversed_transform = apply_macro(macro, validation.iloc[::-1])

    assert transformed.labels.to_dict() == {"hi": 2, "lo": 0, "mid": 1}
    assert reversed_transform.labels.to_dict() == transformed.labels.iloc[::-1].to_dict()
    assert macro.labels.equals(original)
    assert json.loads(json.dumps(macro.metadata["encoder"]))["cutpoints"] == macro.encoder["cutpoints"]


def test_quantile_ties_are_not_arbitrarily_split_and_missing_uses_training_median():
    train = pd.DataFrame({"x": [1.0, 1.0, np.nan, 1.0, 1.0]})
    macro = quantile_macro(train, "x", 3)
    assert macro.state_count == 1
    assert macro.labels.nunique() == 1
    assert apply_macro(macro, pd.DataFrame({"x": [np.nan, 1.0, 8.0]})).labels.tolist() == [0, 0, 1]


def test_composite_and_kmeans_reuse_training_imputation_scaling_and_centers():
    train = pd.DataFrame({"a": [0.0, 1.0, 2.0, 8.0, 9.0, np.nan], "b": [0.0, np.nan, 2.0, 8.0, 9.0, 10.0]})
    validation = pd.DataFrame({"a": [np.nan, 100.0, 0.0], "b": [4.0, -100.0, np.nan]}, index=["m", "extreme", "known"])
    for macro in (composite_quantile_macro(train, ["a", "b"], 2), kmeans_macro(train, ["a", "b"], 2)):
        encoded = apply_macro(macro, validation)
        assert encoded.labels.index.tolist() == validation.index.tolist()
        assert encoded.labels.min() >= 0
        assert encoded.labels.max() < macro.state_count
        assert macro.encoder["preprocess"]["a"]["median"] == 2.0


def test_refit_uses_training_rows_and_preserves_candidate_identity():
    candidate = quantile_macro(pd.DataFrame({"x": range(10)}), "x", 3)
    fitted = refit_macro(candidate, pd.DataFrame({"x": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]}))
    assert fitted.name == candidate.name
    assert fitted.method == candidate.method
    assert fitted.encoder["cutpoints"] != candidate.encoder["cutpoints"]
    assert apply_macro(fitted, pd.DataFrame({"x": [-1000, 1000]})).labels.tolist() == [0, 2]


def test_merged_encoder_applies_and_refits_saved_state_mapping():
    train = pd.DataFrame({"x": np.arange(12, dtype=float)})
    macro = quantile_macro(train, "x", 3)
    merged = merge_macro_states(macro, 0, 2)
    future = pd.DataFrame({"x": [-100.0, 100.0]})
    assert apply_macro(merged, future).labels.tolist() == [0, 0]
    refitted = refit_macro(merged, pd.DataFrame({"x": np.arange(30, 42, dtype=float)}))
    assert refitted.name == merged.name
    assert refitted.labels.nunique() == 2
    assert apply_macro(refitted, future).labels.nunique() == 1


def _confounded_frame(n=4000):
    rng = np.random.default_rng(372)
    u = rng.normal(size=n)
    treatment = u + rng.normal(scale=0.8, size=n)
    outcome = 1.5 * treatment + 4.0 * u + rng.normal(scale=0.7, size=n)
    return pd.DataFrame({"id": np.arange(n), "time": np.arange(n), "u": u, "treatment": treatment, "outcome": outcome, "site": np.where(u > 0, "A", "B")})


def test_adjustment_is_explicit_and_does_not_auto_include_context():
    spec = StudySpec.from_dict({"dataset": {"id_column": "id", "time_column": "time"}, "interventions": ["treatment"], "outcomes": ["outcome"], "columns": {"site": "context"}})
    result = resolve_adjustment(spec, ["id", "time", "site", "treatment", "outcome"], interventions=["treatment"], target_column="outcome")
    assert result["columns"] == []
    assert result["declared"] is False
    assert "inferred" in result["audit"]


def test_spec_rejects_treatment_outcome_time_and_identifier_adjustment():
    for invalid in ("id", "time", "treatment", "outcome"):
        with pytest.raises(ValueError, match="Adjustment columns"):
            StudySpec.from_dict({
                "dataset": {"id_column": "id", "time_column": "time"},
                "interventions": ["treatment"], "outcomes": ["outcome"],
                "adjustment": {"columns": [invalid], "rationale": "pre-treatment cause"},
            })


def test_declared_continuous_confounder_recovers_effect_and_macro_only_is_biased():
    df = _confounded_frame()
    spec = StudySpec.from_dict({
        "dataset": {"id_column": "id", "time_column": "time"},
        "interventions": ["treatment"], "outcomes": ["outcome"],
        "columns": {"u": "context"},
        "adjustment": {"columns": ["u"], "rationale": "pre-treatment common cause", "include_macro": True},
    })
    coarse = quantile_macro(df, "u", 3)
    audit = adjustment_sufficiency_audit(df, spec, coarse, interventions=["treatment"], target_column="outcome")
    macro_coef = audit["comparison"]["macro_only"][0]["coefficient"]
    adjusted_coef = audit["pathways"][0]["coefficient"]

    assert abs(adjusted_coef - 1.5) < 0.08
    assert abs(macro_coef - 1.5) > 0.5
    assert audit["declared_covariate_diagnostics"]["coarsened_by_macro"] == [{"column": "u", "observed_values": len(df), "macro_states": 3}]
    assert audit["evidence_scope"] == "development_only"
    residual_diagnostic = audit["declared_covariate_diagnostics"]["within_macro_residual_information"][0]
    assert residual_diagnostic["within_macro_variance_fraction_remaining"] > 0.2
    assert residual_diagnostic["treatment_covariate_residual_correlations"][0]["residual_correlation"] > 0.5
    assert audit["comparison"]["coefficient_changes"][0]["difference_adjusted_minus_macro_only"] < -1.0
    assert audit["warnings"]
    assert "not causally certified" in audit["interpretation"]


def test_adjustment_spec_serializes_explicit_rationale():
    spec = StudySpec.from_dict({
        "dataset": {"id_column": "id", "time_column": "time"},
        "adjustment": {"columns": ["u"], "rationale": "pre-treatment common cause", "include_macro": False},
    })
    assert StudySpec.from_dict(spec.to_dict()).adjustment.to_dict() == spec.adjustment.to_dict()


@pytest.mark.parametrize("column, column_spec", [
    ("hidden", {"role": "context", "allowed_as_adjustment": False}),
    ("hidden", {"role": "exclude"}),
    ("hidden", {"role": "exposure"}),
    ("__lead_time", {"role": "unknown"}),
    ("__lead1", {"role": "unknown"}),
])
def test_adjustment_rejects_disallowed_aliases_and_reserved_columns(column, column_spec):
    with pytest.raises(ValueError):
        StudySpec.from_dict({
            "dataset": {"id_column": "id", "time_column": "time"},
            "columns": {column: column_spec},
            "adjustment": {"columns": [column], "rationale": "declared for the control"},
        })


def test_resolver_rejects_role_exposure_even_without_interventions_alias():
    with pytest.raises(ValueError, match="Adjustment columns"):
        StudySpec.from_dict({
            "dataset": {"id_column": "id", "time_column": "time"},
            "columns": {"dose": {"role": "exposure"}},
            "adjustment": {"columns": ["dose"], "rationale": "declared for the control"},
        })


def test_refit_rejects_merged_recipe_when_requested_topology_is_absent():
    from causal_emergence_discovery.macro import MacroRefitTopologyError

    original = quantile_macro(pd.DataFrame({"x": np.arange(12, dtype=float)}), "x", 3)
    merged = merge_macro_states(original, 0, 1)
    with pytest.raises(MacroRefitTopologyError):
        refit_macro(merged, pd.DataFrame({"x": [2.0, 2.0, 2.0, 2.0]}))


@pytest.mark.parametrize("adjustment", [
    {"columns": ["u"], "rationale": "x", "include_macro": "false"},
    {"columns": ["u", 3], "rationale": "x"},
    {"columns": ["u", "  "], "rationale": "x"},
])
def test_adjustment_spec_rejects_invalid_types_without_coercion(adjustment):
    with pytest.raises(ValueError):
        StudySpec.from_dict({
            "dataset": {"id_column": "id", "time_column": "time"},
            "adjustment": adjustment,
        })


def test_audit_requires_macro_when_include_macro_is_true():
    df = _confounded_frame(100)
    spec = StudySpec.from_dict({
        "dataset": {"id_column": "id", "time_column": "time"},
        "interventions": ["treatment"], "outcomes": ["outcome"],
        "adjustment": {"columns": ["u"], "rationale": "pre-treatment cause", "include_macro": True},
    })
    with pytest.raises(ValueError, match="requires a fitted MacroAssignment"):
        adjustment_sufficiency_audit(df, spec, None, interventions=["treatment"], target_column="outcome")
