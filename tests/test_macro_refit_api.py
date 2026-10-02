import numpy as np
import pandas as pd
import pytest

from causal_emergence_discovery import scoring as scoring_module
from causal_emergence_discovery.macro import MacroRefitTopologyError, quantile_macro
from causal_emergence_discovery.scoring import MacroScoreRefitError, score_macro
from causal_emergence_discovery.search import greedy_macro_search


def _frame():
    return pd.DataFrame({"x": np.arange(12, dtype=float), "y": [0., 1., 2., 3., 5., 7., 8., 9., 10., 11., 12., 13.]})


def test_score_requires_topology_support_on_every_fold(monkeypatch):
    frame = _frame()
    macro = quantile_macro(frame, "x", 2)
    original_refit = scoring_module.refit_macro
    calls = []

    def refit(candidate, train, **kwargs):
        calls.append(len(train))
        if len(calls) == 1:
            raise MacroRefitTopologyError(
                "incomplete topology",
                {"recipe": "quantile", "reason": "incomplete_refit_topology", "phase": "refit"},
            )
        return original_refit(candidate, train, **kwargs)

    monkeypatch.setattr(scoring_module, "refit_macro", refit)
    splits = [(range(6), range(6, 12)), (range(6, 12), range(6))]
    with pytest.raises(MacroScoreRefitError) as raised:
        score_macro(
            frame, macro, outcome="y", target_column="y",
            micro_feature_columns=["x"], intervention_columns=[],
            validation_splits=splits,
        )
    assert len(calls) == 2
    assert raised.value.status == "unsupported_refit_topology"
    assert raised.value.diagnostics == [{
        "fold_index": 0,
        "topology": {"recipe": "quantile", "reason": "incomplete_refit_topology", "phase": "refit"},
    }]


def test_search_excludes_rejected_merge_from_comparable_ranking():
    frame = pd.DataFrame({
        "x": [0.] * 4 + [5.] * 4 + [10.] * 4,
        "y": np.arange(12, dtype=float),
    })
    initial = quantile_macro(frame, "x", 3)
    result = greedy_macro_search(
        frame, initial, outcome="y", target_column="y",
        micro_feature_columns=["x"], intervention_columns=[], folds=2,
        validation_splits=[(range(4), range(4, 12)), (range(4, 8), [0, 1, 2, 3])],
    )
    assert result["best"]["macro_name"] == initial.name
    assert result["best"]["comparable"] is True
    assert result["status"] == "partial_topology_rejections"
    assert len(result["rejected_candidates"]) == 3
    assert result["rejected_candidates"][0]["status"] == "unsupported_refit_topology"
    assert result["rejected_candidates"][0]["diagnostics"][0]["fold_index"] == 0
    assert result["rejected_candidates"][0]["diagnostics"][0]["topology"]["reason"] == "incomplete_refit_topology"
    assert len(result["sampled_macros"]) == 1
    assert all(record["comparable"] for record in result["sampled_macros"])
