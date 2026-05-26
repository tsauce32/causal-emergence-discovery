import pandas as pd

from causal_emergence_discovery.panel import build_lagged_table, validate_panel
from causal_emergence_discovery.spec import StudySpec


def test_study_spec_accepts_minimal_domain_agnostic_roles():
    spec = StudySpec.from_dict(
        {
            "dataset": {"id_column": "person", "time_column": "week"},
            "outcomes": ["score"],
            "interventions": ["program"],
            "environments": ["site"],
        }
    )

    assert spec.id_column == "person"
    assert spec.outcome_columns(["score"]) == ["score"]
    assert spec.intervention_columns(["program"]) == ["program"]
    assert spec.environment_columns(["site"]) == ["site"]


def test_validate_panel_and_build_lagged_table():
    df = pd.DataFrame(
        {
            "person": [1, 1, 1, 2, 2, 2],
            "week": [0, 1, 2, 0, 1, 2],
            "program": [0, 1, 1, 0, 0, 1],
            "score": [10.0, 11.0, 13.0, 8.0, 8.5, 9.0],
        }
    )
    spec = StudySpec.from_dict(
        {
            "dataset": {"id_column": "person", "time_column": "week"},
            "outcomes": ["score"],
            "interventions": ["program"],
        }
    )

    summary = validate_panel(df, spec)
    lagged = build_lagged_table(df, spec, lag=1)

    assert summary.entity_count == 2
    assert len(lagged.data) == 4
    assert lagged.target_columns["score"] == "score__lead1"
    assert lagged.data["score__lead1"].tolist() == [11.0, 13.0, 8.5, 9.0]
