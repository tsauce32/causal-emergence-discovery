import pytest
import numpy as np
import pandas as pd

from causal_emergence_discovery.models import effect_estimates
from causal_emergence_discovery.spec import AdjustmentSpec, StudySpec


def test_adjustment_estimand_defaults_to_joint_conditional_and_roundtrips():
    spec = StudySpec.from_dict({
        "dataset": {"id_column": "id", "time_column": "time"},
        "adjustment": {"columns": ["u"], "rationale": "declared pre-treatment cause"},
    })

    assert spec.adjustment.estimand == "joint_conditional"
    encoded = spec.to_dict()["adjustment"]
    assert encoded["estimand"] == "joint_conditional"
    assert StudySpec.from_dict(spec.to_dict()).adjustment == spec.adjustment


def test_adjustment_estimand_accepts_explicit_marginal():
    spec = StudySpec.from_dict({
        "dataset": {"id_column": "id", "time_column": "time"},
        "adjustment": {
            "columns": ["u"],
            "rationale": "declared pre-treatment cause",
            "include_macro": True,
            "estimand": "marginal",
        },
    })

    assert spec.adjustment.to_dict() == {
        "columns": ["u"],
        "rationale": "declared pre-treatment cause",
        "include_macro": True,
        "estimand": "marginal",
    }
    assert StudySpec.from_dict(spec.to_dict()).adjustment.estimand == "marginal"


@pytest.mark.parametrize("value", ["", "joint", "conditional", None, 3, ["marginal"]])
def test_adjustment_estimand_rejects_invalid_mapping_and_direct_construction(value):
    with pytest.raises(ValueError, match="adjustment.estimand"):
        StudySpec.from_dict({
            "dataset": {"id_column": "id", "time_column": "time"},
            "adjustment": {
                "columns": ["u"],
                "rationale": "declared pre-treatment cause",
                "estimand": value,
            },
        })

    with pytest.raises(ValueError, match="adjustment.estimand"):
        AdjustmentSpec(columns=("u",), rationale="declared cause", estimand=value)


def test_legacy_adjustment_mapping_preserves_other_declared_semantics():
    spec = StudySpec.from_dict({
        "dataset": {"id_column": "id", "time_column": "time"},
        "interventions": ["treatment"],
        "outcomes": ["outcome"],
        "adjustment": {"columns": ["u"], "rationale": "pre-treatment cause"},
    })

    assert spec.adjustment.columns == ("u",)
    assert spec.adjustment.rationale == "pre-treatment cause"
    assert spec.adjustment.include_macro is False
    assert spec.adjustment.estimand == "joint_conditional"


def test_effect_estimates_defaults_to_joint_and_can_request_marginal():
    rng = np.random.default_rng(831)
    confounder = rng.normal(size=2400)
    treatment_a = confounder + rng.normal(scale=0.7, size=len(confounder))
    treatment_b = 0.8 * treatment_a + 0.3 * confounder + rng.normal(scale=0.5, size=len(confounder))
    outcome = 1.7 * treatment_a - 0.9 * treatment_b + 1.1 * confounder + rng.normal(scale=0.5, size=len(confounder))
    frame = pd.DataFrame({"a": treatment_a, "b": treatment_b, "u": confounder, "y": outcome})

    joint = effect_estimates(frame, ["a", "b"], "y", ["u"])
    marginal = effect_estimates(frame, ["a", "b"], "y", ["u"], estimand="marginal")
    joint_by_term = {row["intervention"]: row["coefficient"] for row in joint}
    marginal_by_term = {row["intervention"]: row["coefficient"] for row in marginal}

    assert joint_by_term["a"] == pytest.approx(1.7, abs=0.06)
    assert joint_by_term["b"] == pytest.approx(-0.9, abs=0.06)
    assert abs(marginal_by_term["a"] - 1.7) > 0.35
    assert abs(marginal_by_term["b"] + 0.9) > 0.35
