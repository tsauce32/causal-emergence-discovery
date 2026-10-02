"""Unsupported reserved evaluation is visible through the CLI, without a crash."""

import json

import numpy as np
import pandas as pd

from causal_emergence_discovery.cli import main
from causal_emergence_discovery.discovery import DiscoveryConfig, run_discovery
from causal_emergence_discovery.panel import build_lagged_table
from causal_emergence_discovery.spec import StudySpec


def test_cli_reports_unevaluable_outer_holdout(tmp_path, capsys):
    rng = np.random.default_rng(391)
    frame = pd.DataFrame([
        {"entity": entity, "time": time, "x": rng.normal(), "y": rng.normal()}
        for entity in range(12) for time in range(6)
    ])
    declaration = {
        "dataset": {"id_column": "entity", "time_column": "time"},
        "outcomes": ["y"],
        "columns": {
            "x": {"role": "context", "type": "numeric"},
            "y": {"role": "outcome", "allowed_as_cause": False, "allowed_as_adjustment": False},
        },
    }
    spec = StudySpec.from_dict(declaration)
    result = run_discovery(frame, spec, DiscoveryConfig(folds=2, max_states=2, paths=1, seed=7))
    lagged = build_lagged_table(frame, spec)
    heldout = lagged.data.iloc[result["validation"]["outer"]["test_indices"]].entity.unique()
    frame.loc[frame.entity.isin(heldout), "y"] = np.nan
    csv_path, spec_path, output_path = tmp_path / "panel.csv", tmp_path / "spec.json", tmp_path / "result.json"
    frame.to_csv(csv_path, index=False)
    spec_path.write_text(json.dumps(declaration), encoding="utf-8")

    exit_code = main([
        "discover", str(csv_path), str(spec_path), "--folds", "2", "--max-states", "2",
        "--paths", "1", "--seed", "7", "--output", str(output_path),
    ])

    assert exit_code == 0
    assert "untouched holdout: not_evaluable" in capsys.readouterr().out
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["validation"]["outer"]["test_indices"] == result["validation"]["outer"]["test_indices"]
    assert saved["outer_evaluation"]["finite_test_targets"] == 0
    assert saved["outer_evaluation"]["macro_r2"] is None
