import numpy as np
import pandas as pd
import pytest

from causal_emergence_discovery.adjustment import adjustment_sufficiency_audit, _covariate_residual_diagnostics
from causal_emergence_discovery.spec import StudySpec


def _correlated_treatment_fixture(n=5000):
    rng = np.random.default_rng(8241)
    confounder = rng.normal(size=n)
    treatment_a = confounder + rng.normal(scale=0.8, size=n)
    treatment_b = 0.75 * treatment_a + 0.5 * confounder + rng.normal(scale=0.55, size=n)
    outcome = 1.7 * treatment_a - 0.9 * treatment_b + 1.2 * confounder + rng.normal(scale=0.5, size=n)
    frame = pd.DataFrame({"id": np.arange(n), "time": np.arange(n), "c": confounder,
                          "a": treatment_a, "b": treatment_b, "y": outcome})
    spec = StudySpec.from_dict({
        "dataset": {"id_column": "id", "time_column": "time"},
        "outcomes": ["y"], "interventions": ["a", "b"],
        "columns": {"c": {"role": "context", "type": "continuous"}},
        "adjustment": {"columns": ["c"], "rationale": "generated common cause"},
    })
    return frame, spec


def test_adjustment_defaults_to_joint_conditional_and_marginal_is_explicit():
    frame, spec = _correlated_treatment_fixture()
    default = adjustment_sufficiency_audit(frame, spec, interventions=["a", "b"], target_column="y")
    marginal = adjustment_sufficiency_audit(
        frame, spec, interventions=["a", "b"], target_column="y", estimand="marginal"
    )

    joint_coefficients = {row["intervention"]: row["coefficient"] for row in default["pathways"]}
    marginal_coefficients = {row["intervention"]: row["coefficient"] for row in marginal["pathways"]}
    assert default["estimand"] == "joint_conditional"
    assert joint_coefficients["a"] == pytest.approx(1.7, abs=0.04)
    assert joint_coefficients["b"] == pytest.approx(-0.9, abs=0.04)
    assert abs(marginal_coefficients["a"] - 1.7) > 0.4
    assert abs(marginal_coefficients["b"] + 0.9) > 0.4
    assert default["comparison"]["macro_only"][0]["estimand"] == "joint_conditional"
    assert default["fit_specification"]["joint_conditional_terms"] == ["a", "b", "c"]
    assert default["uncertainty"]["panel_uncertainty_status"] == "unsupported"
    assert "Associational" in default["fit_specification"]["coefficient_interpretation"]


def test_adjustment_declaration_estimand_is_used_when_audit_keyword_is_omitted():
    frame, base = _correlated_treatment_fixture()
    declared = StudySpec.from_dict({
        **base.to_dict(),
        "adjustment": {
            "columns": ["c"], "rationale": "generated common cause", "estimand": "marginal"
        },
    })
    audit = adjustment_sufficiency_audit(frame, declared, interventions=["a", "b"], target_column="y")
    assert audit["estimand"] == "marginal"


def test_categorical_intervention_has_reference_coded_contrast_metadata():
    frame = pd.DataFrame({
        "id": np.arange(90), "time": np.arange(90),
        "arm": ["control", "low", "high"] * 30,
        "y": [0.0, 1.0, 2.0] * 30,
    })
    spec = StudySpec.from_dict({
        "dataset": {"id_column": "id", "time_column": "time"},
        "outcomes": ["y"], "interventions": ["arm"],
    })
    audit = adjustment_sufficiency_audit(frame, spec, interventions=["arm"], target_column="y")
    contrasts = [term for row in audit["pathways"] for term in row.get("terms", []) if term.get("level") is not None]
    assert {row["level"]["value"] for row in contrasts} == {"low", "high"}
    assert {row["reference_level"]["value"] for row in contrasts} == {"control"}
    assert all(row["feature_name"].startswith("arm=") for row in contrasts)
    assert audit["pathways"][0]["estimand"] == "joint_conditional"
    assert {row["term"] for row in audit["comparison"]["coefficient_changes"]} == {
        row["feature_name"] for row in contrasts
    }


def test_estimand_rejects_unknown_values():
    frame, spec = _correlated_treatment_fixture(100)
    try:
        adjustment_sufficiency_audit(frame, spec, interventions=["a", "b"], target_column="y", estimand="total")
    except ValueError as exc:
        assert "estimand" in str(exc)
    else:
        raise AssertionError("invalid estimand was accepted")


def test_residual_diagnostic_keeps_literal_missing_label_distinct_from_null():
    frame = pd.DataFrame({
        "macro": ["__missing__", None, "__missing__", None],
        "u": [0.0, 100.0, 2.0, 102.0],
        "t": [1.0, 1.0, 1.0, 1.0],
    })
    result = _covariate_residual_diagnostics(frame, ["u"], ["t"], "macro")[0]
    assert result["within_macro_variance"] == pytest.approx(1.0)
