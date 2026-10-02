"""Adversarial checks against the explicitly pinned integrated source.

Set DISCOVERY_BENCHMARK_SOURCE and DISCOVERY_BENCHMARK_EXPECTED_COMMIT when
running these tests. They skip when the integrated checkout is not ready; they
never import the benchmark checkout's own package source as a proxy.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.pipeline import baselines
from benchmarks.pipeline.dgps import generate_case
from benchmarks.pipeline.protocol import generate_seeds, get_profile, summarize_trials
from benchmarks.pipeline.runner import EXPECTED_BASE, _load_package, verify_source


def _integrated_source():
    raw_source = os.environ.get("DISCOVERY_BENCHMARK_SOURCE")
    expected = os.environ.get("DISCOVERY_BENCHMARK_EXPECTED_COMMIT")
    if not raw_source or not expected:
        pytest.skip("set DISCOVERY_BENCHMARK_SOURCE and DISCOVERY_BENCHMARK_EXPECTED_COMMIT to the ready integrated checkout")
    source = Path(raw_source).resolve()
    info = verify_source(source, expected)
    assert info["head"].lower() != EXPECTED_BASE.lower(), "the reviewed validation base is not the integrated benchmark source"
    return _load_package(source)


def _discovery_result(discovery, frame, spec, *, mode="entity_holdout", seed=31):
    config = discovery.DiscoveryConfig(
        outcome="y", lag=1, max_states=3, paths=1, branching_factor=1,
        folds=2, top_k=1, validation_mode=mode, holdout_fraction=0.25, seed=seed,
    )
    return discovery.run_discovery(frame, spec, config)


def _split_keys(discovery, frame, spec, result):
    from causal_emergence_discovery.panel import build_lagged_table

    lagged = build_lagged_table(frame, spec, outcomes=["y"], lag=1)
    plan = result["validation"]["outer"]
    return {
        (str(lagged.data.iloc[i][spec.id_column]),
         int(lagged.data.iloc[i][spec.time_column]),
         int(lagged.data.iloc[i][lagged.target_time_column]))
        for i in plan["test_indices"]
    }


def test_heldout_outcome_missingness_does_not_reselect_the_reserved_entities():
    discovery = _integrated_source()
    from causal_emergence_discovery.panel import build_lagged_table
    from causal_emergence_discovery.spec import StudySpec

    generated = generate_case("additive_signal", seed=31, n_entities=48, n_periods=16)
    frame = generated.frame.copy()
    spec = StudySpec.from_dict(generated.spec_dict)
    baseline = _discovery_result(discovery, frame, spec)
    baseline_lagged = build_lagged_table(frame, spec, outcomes=["y"], lag=1)
    outer = baseline["validation"]["outer"]
    target_keys = {
        (baseline_lagged.data.iloc[i][spec.id_column],
         int(baseline_lagged.data.iloc[i]["__lead_time"]))
        for i in outer["test_indices"]
    }
    assert target_keys

    poisoned = frame.copy()
    for entity, lead_time in target_keys:
        poisoned.loc[(poisoned[spec.id_column] == entity) &
                     (poisoned[spec.time_column] == lead_time), "y"] = np.nan
    mutated = _discovery_result(discovery, poisoned, spec)
    baseline_keys = _split_keys(discovery, frame, spec, baseline)
    mutated_keys = _split_keys(discovery, poisoned, spec, mutated)
    assert mutated_keys == baseline_keys
    baseline_top = [(item["macro_name"], item["ranking_score"]) for item in baseline["top_macros"]]
    mutated_top = [(item["macro_name"], item["ranking_score"]) for item in mutated["top_macros"]]
    assert mutated_top == baseline_top


def test_heldout_predictor_dtype_mutation_cannot_remove_a_declared_numeric_candidate():
    discovery = _integrated_source()
    from causal_emergence_discovery.panel import build_lagged_table
    from causal_emergence_discovery.spec import StudySpec

    generated = generate_case("additive_signal", seed=42, n_entities=48, n_periods=16)
    frame = generated.frame.copy()
    spec = StudySpec.from_dict(generated.spec_dict)
    baseline = _discovery_result(discovery, frame, spec)
    lagged = build_lagged_table(frame, spec, outcomes=["y"], lag=1)
    test_indices = baseline["validation"]["outer"]["test_indices"]
    predictor_row = lagged.data.iloc[test_indices[0]]
    poisoned = frame.copy()
    mask = ((poisoned[spec.id_column] == predictor_row[spec.id_column]) &
            (poisoned[spec.time_column] == predictor_row[spec.time_column]))
    poisoned["x1"] = poisoned["x1"].astype("object")
    poisoned.loc[mask, "x1"] = "malformed-heldout-value"
    mutated = _discovery_result(discovery, poisoned, spec)

    assert _split_keys(discovery, frame, spec, baseline) == _split_keys(discovery, poisoned, spec, mutated)
    baseline_names = [search["initial_macro"] for search in baseline["searches"]]
    mutated_names = [search["initial_macro"] for search in mutated["searches"]]
    assert any(":x1" in name for name in baseline_names)
    assert any(":x1" in name for name in mutated_names)
    assert [(item["macro_name"], item["ranking_score"]) for item in baseline["top_macros"]] == [
        (item["macro_name"], item["ranking_score"]) for item in mutated["top_macros"]
    ]


def _assert_refit_is_explicitly_unsupported_or_rejected(macro_module, macro, fold):
    try:
        refitted = macro_module.refit_macro(macro, fold)
    except (ValueError, RuntimeError) as exc:
        assert any(word in str(exc).lower() for word in ("refit", "merge", "state", "recipe", "support", "topology"))
        return
    diagnostics = refitted.metadata.get("refit_diagnostics", {})
    diagnostics = {**refitted.metadata, **diagnostics} if isinstance(diagnostics, dict) else refitted.metadata
    topology = refitted.metadata.get("refit_topology", {})
    topology_changed = isinstance(topology, dict) and any(
        topology.get(expected) is not None
        and topology.get(observed) is not None
        and topology.get(expected) != topology.get(observed)
        for expected, observed in (("expected_states", "observed_states"),
                                   ("requested_states", "observed_states"),
                                   ("expected_cutpoints", "observed_cutpoints"))
    )
    degenerate = (
        bool(refitted.metadata.get("refit_merge_degeneracies"))
        or bool(refitted.metadata.get("recipe_estimability_failure"))
        or diagnostics.get("recipe_estimable") is False
        or diagnostics.get("estimable") is False
        or topology_changed
    )
    assert degenerate, "ambiguous refit silently produced an apparently estimable recipe"


def test_merged_cluster_recipe_with_missing_population_is_not_silently_relabelled():
    _integrated_source()
    from causal_emergence_discovery import macro as macro_module

    full = pd.DataFrame({"x": np.repeat([0.0, 10.0, 20.0], 12)})
    assignment = macro_module.kmeans_macro(full, ["x"], 3)
    by_value = {
        value: int(assignment.labels.loc[full["x"] == value].mode().iloc[0])
        for value in (0.0, 10.0, 20.0)
    }
    merged = macro_module.merge_macro_states(assignment, by_value[0.0], by_value[10.0])
    fold = pd.DataFrame({"x": np.repeat([10.0, 20.0], 8)})
    _assert_refit_is_explicitly_unsupported_or_rejected(macro_module, merged, fold)


def test_tied_centers_and_collapsed_quantile_cutpoints_are_flagged_or_rejected():
    _integrated_source()
    from causal_emergence_discovery import macro as macro_module

    separated = pd.DataFrame({"x": np.repeat([0.0, 10.0, 20.0], 12)})
    clustered = macro_module.merge_macro_states(
        macro_module.kmeans_macro(separated, ["x"], 3), 0, 1
    )
    tied_training = pd.DataFrame({"x": np.repeat(10.0, 12)})
    _assert_refit_is_explicitly_unsupported_or_rejected(macro_module, clustered, tied_training)

    quantiled = macro_module.merge_macro_states(
        macro_module.quantile_macro(separated, "x", 3), 0, 1
    )
    collapsed_training = pd.DataFrame({"x": np.repeat([0.0, 10.0], [10, 2])})
    _assert_refit_is_explicitly_unsupported_or_rejected(macro_module, quantiled, collapsed_training)


def test_trial_summary_counts_failures_and_missing_macro_micro_pairs_separately():
    profile = get_profile("quick")
    seed0, seed1 = generate_seeds(profile)
    profile["cells"] = [{"case": "iid_null", "split": "entity_holdout"}]
    records = [
        {"case": "iid_null", "split": "entity_holdout", "seed": seed0,
         "status": "success", "paired_r2": {"selected_macro": 0.2, "restricted_micro": 0.1,
          "matched_micro": None}},
        {"case": "iid_null", "split": "entity_holdout", "seed": seed1,
         "status": "failed", "failure_reason": "unsupported refit",
         "paired_r2": {"selected_macro": None, "restricted_micro": None}},
    ]
    cell = summarize_trials(records, profile)["cells"]["iid_null::entity_holdout"]
    assert cell["n_planned"] == 2
    assert cell["n_records"] == 2
    assert cell["n_success"] == 1
    assert cell["n_failed_or_unsupported"] == 1
    assert cell["failure_rate_planned"] == 0.5
    assert cell["paired_r2_differences"]["selected_macro_minus_restricted_micro"]["n"] == 1
    assert cell["paired_r2_differences"]["selected_macro_minus_matched_micro"]["n"] == 0
