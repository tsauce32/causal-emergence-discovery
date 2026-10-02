"""Exercise the regression contract through discovery and its user-facing CLI."""

import json

import numpy as np
import pandas as pd
import pytest

from causal_emergence_discovery.cli import _print_summary
from causal_emergence_discovery.discovery import DiscoveryConfig, run_discovery
from causal_emergence_discovery.models import fit_linear_model, predict_r2, treatment_effect_records
from causal_emergence_discovery.panel import build_lagged_table
from causal_emergence_discovery.spec import StudySpec


def _panel(*, categorical=False, alias=False):
    rng = np.random.default_rng(988)
    rows = []
    for entity in range(40):
        previous_y = rng.normal()
        for time in range(5):
            u = rng.normal()
            a = rng.normal()
            b = a if alias else 0.85 * a + rng.normal(scale=0.4)
            if categorical:
                a = "active" if a > 0 else "control"
                signal = 1.7 * (a == "active")
            else:
                signal = 1.7 * a
            rows.append(dict(id=entity, time=time, u=u, a=a, b=b, y=previous_y))
            previous_y = signal - 0.9 * b + 1.2 * u + rng.normal(scale=0.2)
    return pd.DataFrame(rows)


def _spec(estimand):
    return StudySpec.from_dict({
        "dataset": {"id_column": "id", "time_column": "time"},
        "interventions": ["a", "b"],
        "outcomes": ["y"],
        "columns": {
            "u": "context",
            "y": {"role": "outcome", "allowed_as_cause": False, "allowed_as_adjustment": False},
        },
        "adjustment": {"columns": ["u"], "rationale": "Generated before treatments for this control.", "estimand": estimand},
    })


def _run(df, study):
    return run_discovery(df, study, DiscoveryConfig(folds=2, max_states=2, paths=1, top_k=1, seed=3))


@pytest.mark.parametrize("estimand", ["joint_conditional", "marginal"])
def test_discovery_uses_declared_regression_estimand_and_only_development_rows(estimand):
    df = _panel()
    study = _spec(estimand)
    result = _run(df, study)
    assert result["adjustment"]["estimand"] == estimand
    assert result["adjustment"]["evidence_scope"] == "development_only"
    lagged = build_lagged_table(df, study).data
    development = lagged.iloc[result["validation"]["outer"]["train_indices"]]
    for pathway in result["candidate_pathways"]:
        treatment = pathway["intervention"]
        columns = ["a", "b", "u"] if estimand == "joint_conditional" else [treatment, "u"]
        design = np.column_stack([np.ones(len(development)), development[columns].to_numpy()])
        expected = np.linalg.lstsq(design, development["y__lead1"].to_numpy(), rcond=None)[0]
        assert pathway["coefficient"] == pytest.approx(expected[1 + columns.index(treatment)], abs=1e-10)
        assert pathway["nobs"] == len(development)
    json.dumps(result, allow_nan=False)


def test_categorical_contrasts_and_aliased_terms_survive_discovery_summary(capsys):
    categorical = _run(_panel(categorical=True), _spec("joint_conditional"))
    pathway = next(p for p in categorical["candidate_pathways"] if p["intervention"] == "a")
    assert pathway["coefficient"] is None
    assert len(pathway["terms"]) == 1
    assert pathway["terms"][0]["coefficient"] is not None
    _print_summary(categorical)
    output = capsys.readouterr().out
    assert "joint_conditional" in output
    assert "active" in output or "control" in output
    assert "associational coef=" in output

    aliased = _run(_panel(alias=True), _spec("joint_conditional"))
    assert all(p["coefficient"] is None for p in aliased["candidate_pathways"])
    assert all(p["stderr"] is None for p in aliased["candidate_pathways"])
    _print_summary(aliased)
    assert "coef=n/a" in capsys.readouterr().out
    json.dumps(aliased, allow_nan=False)


def test_term_extraction_uses_positions_when_display_names_collide():
    rng = np.random.default_rng(462)
    numeric = rng.normal(size=60)
    control = np.tile([False, True], 30)
    frame = pd.DataFrame({
        "arm": np.where(control, "control", "active"),
        "arm='control'": numeric,
        "intercept": rng.normal(size=60),
    })
    frame["y"] = 2 * numeric + 3 * control + 0.5 * frame["intercept"]
    records = treatment_effect_records(frame, ["arm='control'", "arm", "intercept"], "y", [])
    by_intervention = {r["intervention"]: r for r in records}
    assert by_intervention["arm='control'"]["coefficient"] == pytest.approx(2.0)
    assert by_intervention["arm"]["terms"][0]["coefficient"] == pytest.approx(3.0)
    assert by_intervention["intercept"]["coefficient"] == pytest.approx(0.5)


def test_underidentified_wide_design_does_not_report_separate_term_coefficients():
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0], "z": [0.0, 1.0, 0.0], "w": [1.0, 0.0, 0.0], "y": [1.0, 2.0, 3.0]})
    record = treatment_effect_records(frame, ["x"], "y", ["z", "w"])[0]
    assert record["design_columns"] == 4
    assert record["design_rank"] == 3
    assert record["residual_df"] == 0
    assert record["coefficient"] is None
    assert record["stderr"] is None


def test_unrepresentable_validation_r2_fails_instead_of_clipping():
    fit = fit_linear_model(pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]}), ["x"], "y")
    validation = pd.DataFrame({"x": [1e308, -1e308], "y": [0.0, 1.0]})
    with pytest.raises(ValueError, match="R2 is not finite"):
        predict_r2(fit, validation, ["x"], "y")
