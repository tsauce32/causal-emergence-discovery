from causal_emergence_discovery.cli import _print_summary


def test_cli_summary_labels_ranking_and_predictive_gap(capsys):
    _print_summary(
        {
            "status": "experimental_hypothesis_generation",
            "config": {"outcome": "y", "lag": 1},
            "panel": {"row_count": 20, "entity_count": 5},
            "columns": {"target": "y__lead1", "macro_features": ["x"]},
            "top_macros": [
                {
                    "macro_name": "q2:x",
                    "ranking_score": 0.18,
                    "macro_r2": 0.08,
                    "micro_r2": -0.01,
                    "predictive_r2_difference": 0.09,
                    "predictive_evaluation_scope": "row_modulo_kfold",
                    "emergence_evidence_status": "not_assessed",
                    "specificity": 0.2,
                    "stability": 0.8,
                    "compression": 0.5,
                    "state_count": 2,
                }
            ],
            "candidate_pathways": [],
            "warnings": [],
        }
    )

    output = capsys.readouterr().out
    assert "ranking_score=0.1800" in output
    assert "predictive R2 difference (macro - micro, mean paired fold gap): 0.0900" in output
    assert "predictive evaluation scope: row_modulo_kfold" in output
    assert "emergence evidence: not_assessed" in output
    assert "delta=" not in output
