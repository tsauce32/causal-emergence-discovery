"""Compatibility and fail-closed controls for serialized merge recipes."""

import copy
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from causal_emergence_discovery import MacroRefitTopologyError
from causal_emergence_discovery.macro import (
    kmeans_macro, merge_macro_states, quantile_macro, refit_macro,
)
from causal_emergence_discovery.search import greedy_macro_search
from causal_emergence_discovery.scoring import score_macro


def _frame():
    return pd.DataFrame({"x": [0.] * 4 + [10.] * 4 + [20.] * 4, "y": np.arange(12.)})


def test_legacy_merged_kmeans_rejects_uncertified_rank_identity():
    frame = _frame()
    merged = merge_macro_states(kmeans_macro(frame, ["x"], 3), 0, 1)
    legacy = copy.deepcopy(merged.encoder)
    legacy.pop("algorithm_version")
    legacy.pop("identity")
    with pytest.raises(MacroRefitTopologyError) as caught:
        refit_macro(replace(merged, encoder=legacy), frame)
    assert caught.value.diagnostics["reason"] == "unsupported_legacy_identity"
    assert caught.value.diagnostics["phase"] == "source"


def test_legacy_merged_quantile_rejects_missing_raw_occupancy_evidence():
    frame = _frame()
    merged = merge_macro_states(quantile_macro(frame, "x", 3), 0, 1)
    legacy = copy.deepcopy(merged.encoder)
    legacy.pop("fit_occupied_raw_states")
    legacy.pop("fit_observed_states")
    with pytest.raises(MacroRefitTopologyError) as caught:
        refit_macro(replace(merged, encoder=legacy), frame)
    assert caught.value.diagnostics["reason"] == "unsupported_source_topology"


def test_unmerged_legacy_kmeans_refit_migrates_without_reusing_learned_centers():
    frame = _frame()
    original = kmeans_macro(frame, ["x"], 3)
    legacy = copy.deepcopy(original.encoder)
    legacy.pop("algorithm_version")
    legacy.pop("identity")
    legacy["centers"] = [[999.], [1000.], [1001.]]
    fitted = refit_macro(replace(original, encoder=legacy), frame)
    assert fitted.encoder["centers"] == original.encoder["centers"]
    assert fitted.encoder["algorithm_version"] == 2
    assert fitted.encoder["identity"] == "canonical_lexicographic_centroid_rank"


@pytest.mark.parametrize("left,right", [(1, 2), (0, 1)])
def test_chained_history_rejects_either_operand_after_it_was_merged_away(left, right):
    frame = _frame()
    merged = merge_macro_states(kmeans_macro(frame, ["x"], 3), 0, 1)
    encoder = copy.deepcopy(merged.encoder)
    encoder["merges"].append({"left": left, "right": right})
    with pytest.raises(MacroRefitTopologyError) as caught:
        refit_macro(replace(merged, encoder=encoder), frame)
    assert caught.value.diagnostics["reason"] == "invalid_chained_merge_operands"
    assert caught.value.diagnostics["merge_index"] == 1


def test_search_with_unsupported_initial_merge_has_no_ranked_score_and_is_json_safe():
    frame = _frame()
    merged = merge_macro_states(kmeans_macro(frame, ["x"], 3), 0, 1)
    encoder_before = copy.deepcopy(merged.encoder)
    result = greedy_macro_search(
        frame, merged, outcome="y", target_column="y",
        micro_feature_columns=["x"], intervention_columns=[],
        validation_splits=[(range(4, 12), range(4)), (range(8), range(8, 12))],
    )
    assert result["status"] == "no_valid_candidate"
    assert result["best"] is None
    assert result["sampled_macros"] == []
    assert result["sampled_macro_count"] == 0
    failures = result["rejected_candidates"][0]["diagnostics"]
    assert [failure["fold_index"] for failure in failures] == [0, 1]
    assert all(failure["topology"]["observed_states"] == 2 for failure in failures)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert merged.encoder == encoder_before


def test_unmerged_reduced_folds_keep_their_topology_in_serialized_score():
    frame = _frame()
    candidate = kmeans_macro(frame, ["x"], 3)
    score = score_macro(
        frame, candidate, outcome="y", target_column="y",
        micro_feature_columns=["x"], intervention_columns=[],
        validation_splits=[(range(4, 12), range(4)), (range(8), range(8, 12))],
    )
    serialized = score.to_dict()
    topologies = serialized["fold_refit_topologies"]
    assert len(score.macro_fold_scores) == len(topologies) == 2
    assert [item["fold_index"] for item in topologies] == [0, 1]
    assert all(item["requested_states"] == 3 and item["observed_states"] == 2 for item in topologies)
    assert all(item["merged"] is False for item in topologies)
    assert json.loads(json.dumps(serialized, allow_nan=False)) == serialized
