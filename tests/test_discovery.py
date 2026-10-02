import json

import numpy as np
import pandas as pd

from causal_emergence_discovery.cli import main
from causal_emergence_discovery.discovery import DiscoveryConfig, run_discovery
from causal_emergence_discovery.spec import StudySpec


def test_run_discovery_returns_ranked_macro_hypotheses():
    df = _synthetic_panel()
    spec = _spec()

    result = run_discovery(
        df,
        spec,
        DiscoveryConfig(
            outcome="reading_score",
            lag=1,
            max_states=4,
            paths=4,
            branching_factor=2,
            folds=3,
            top_k=3,
        ),
    )

    assert result["status"] == "experimental_hypothesis_generation"
    assert result["top_macros"]
    assert result["top_macros"][0]["ranking_score"] > 0.0
    assert result["top_macros"][0]["state_count"] >= 2
    assert result["candidate_pathways"][0]["intervention"] == "program_hours"


def test_cli_validate_and_discover(tmp_path, capsys):
    csv_path = tmp_path / "panel.csv"
    spec_path = tmp_path / "study.json"
    _synthetic_panel(entity_count=40).to_csv(csv_path, index=False)
    spec_path.write_text(json.dumps(_spec_dict()), encoding="utf-8")

    assert main(["validate", str(csv_path), str(spec_path)]) == 0
    assert main(
        [
            "discover",
            str(csv_path),
            str(spec_path),
            "--outcome",
            "reading_score",
            "--max-states",
            "3",
            "--paths",
            "3",
            "--folds",
            "3",
        ]
    ) == 0

    output = capsys.readouterr().out

    assert "PASS" in output
    assert "causal-emergence discovery" in output
    assert "best macro:" in output


def _synthetic_panel(entity_count: int = 90, periods: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for entity in range(entity_count):
        site = "a" if entity % 2 == 0 else "b"
        latent_support = rng.normal()
        reading = 40 + 3 * latent_support + rng.normal(scale=1.5)
        for time in range(periods):
            stress = rng.normal(loc=-latent_support, scale=0.7)
            attendance = 0.8 + 0.06 * latent_support - 0.03 * stress + rng.normal(scale=0.02)
            program_hours = max(0.0, 2.0 + rng.normal(scale=0.5))
            rows.append(
                {
                    "entity_id": entity,
                    "time": time,
                    "site": site,
                    "stress": stress,
                    "attendance": attendance,
                    "program_hours": program_hours,
                    "reading_score": reading,
                }
            )
            reading = reading + 1.2 * program_hours + 1.1 * attendance - 0.4 * stress + rng.normal(scale=0.8)
    return pd.DataFrame(rows)


def _spec() -> StudySpec:
    return StudySpec.from_dict(_spec_dict())


def _spec_dict() -> dict:
    return {
        "dataset": {"id_column": "entity_id", "time_column": "time"},
        "outcomes": ["reading_score"],
        "interventions": ["program_hours"],
        "environments": ["site"],
        "columns": {
            "site": {"role": "environment", "type": "categorical"},
            "stress": {"role": "context", "type": "numeric"},
            "attendance": {"role": "measurement", "type": "numeric"},
            "program_hours": {"role": "intervention", "type": "numeric"},
            "reading_score": {"role": "outcome", "type": "numeric"},
        },
    }
