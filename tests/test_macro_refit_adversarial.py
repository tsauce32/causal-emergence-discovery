"""Independent adversarial checks for merged macro refits.

These tests treat canonical state rank as recipe identity, not proof that a
state has the same population meaning after the training distribution shifts.
"""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from causal_emergence_discovery.macro import (
    MacroRefitTopologyError,
    apply_macro,
    kmeans_macro,
    merge_macro_states,
    quantile_macro,
    refit_macro,
)


def _counterexample_frame():
    return pd.DataFrame({"x": [0.0] * 4 + [10.0] * 4 + [20.0] * 4})


def _labels_by_row(frame, labels, columns):
    return {
        tuple(row): int(label)
        for row, label in zip(frame[columns].itertuples(index=False, name=None), labels)
    }


def test_lost_original_kmeans_population_rejects_merge_replay():
    full = _counterexample_frame()
    merged = merge_macro_states(kmeans_macro(full, ["x"], 3), 0, 1)
    fold_train = pd.DataFrame({"x": [10.0] * 4 + [20.0] * 4})

    with pytest.raises(MacroRefitTopologyError) as caught:
        refit_macro(merged, fold_train)

    assert caught.value.recipe == "kmeans"
    assert caught.value.expected_states == 3
    assert caught.value.observed_states == 2
    assert caught.value.diagnostics["reason"] in {
        "incomplete_refit_topology",
        "refit_topology_changed",
    }


def test_kmeans_tie_rows_are_invariant_to_permutation_and_duplicates():
    rows = [(0, 0), (0, 1), (0, 8), (0, 9), (1, 0), (1, 1), (1, 8), (1, 9)]
    base = pd.DataFrame(rows, columns=["x", "z"])
    # Add repeated observations and exercise more than the review's one shuffle.
    duplicated = pd.concat([base, base.iloc[[0, 2, 2, 7]]], ignore_index=True)
    baseline = kmeans_macro(duplicated, ["x", "z"], 3)
    expected = _labels_by_row(duplicated, baseline.labels, ["x", "z"])
    expected_centers = np.asarray(baseline.encoder["centers"])

    for seed in range(20):
        permuted = duplicated.sample(frac=1, random_state=seed).reset_index(drop=True)
        fit = kmeans_macro(permuted, ["x", "z"], 3)
        assert _labels_by_row(permuted, fit.labels, ["x", "z"]) == expected
        np.testing.assert_array_equal(np.asarray(fit.encoder["centers"]), expected_centers)


def test_kmeans_ordered_sums_are_stable_under_large_magnitude_permutations():
    # These rows create cancellation-sensitive means while remaining finite.
    values = [1e16, 1.0, -1e16, 3.0, 1e16, -1e16, 7.0, 9.0]
    frame = pd.DataFrame({"x": values, "z": [0.0, 1.0, 8.0, 9.0, 0.0, 1.0, 8.0, 9.0]})
    baseline = kmeans_macro(frame, ["x", "z"], 3)

    for seed in range(12):
        permuted = frame.sample(frac=1, random_state=seed).reset_index(drop=True)
        fit = kmeans_macro(permuted, ["x", "z"], 3)
        assert _labels_by_row(permuted, fit.labels, ["x", "z"]) == _labels_by_row(
            frame, baseline.labels, ["x", "z"]
        )
        np.testing.assert_allclose(
            np.asarray(fit.encoder["centers"]),
            np.asarray(baseline.encoder["centers"]),
            rtol=0,
            atol=0,
        )


def test_quantile_merge_replay_rejects_a_collapsed_bin_topology():
    source = quantile_macro(pd.DataFrame({"x": np.arange(12, dtype=float)}), "x", 3)
    merged = merge_macro_states(source, 0, 1)
    # At the requested tertiles, this unbalanced two-point sample has only
    # one distinct cutpoint, so the three-bin recipe has collapsed topology.
    folded = pd.DataFrame({"x": [0.0] * 11 + [10.0]})

    with pytest.raises(MacroRefitTopologyError) as caught:
        refit_macro(merged, folded)

    assert caught.value.recipe == "quantile"
    assert caught.value.expected_cutpoints == 2
    assert caught.value.observed_cutpoints < 2


def test_kmeans_refit_preserves_nondefault_iteration_recipe():
    full = _counterexample_frame()
    base = kmeans_macro(full, ["x"], 3, max_iter=1)
    merged = merge_macro_states(base, 0, 1)

    refitted = refit_macro(merged, full)

    assert refitted.encoder["max_iter"] == 1


def test_source_merge_history_accepts_valid_chain_and_rejects_map_corruption():
    full = _counterexample_frame()
    base = kmeans_macro(full, ["x"], 3)
    chained = merge_macro_states(merge_macro_states(base, 0, 1), 0, 2)

    refitted = refit_macro(chained, full)
    assert refitted.state_count == 1
    assert refitted.encoder["merges"] == [{"left": 0, "right": 1}, {"left": 0, "right": 2}]

    corrupted = json.loads(json.dumps(chained.encoder))
    corrupted["state_map"]["2"] = 2
    with pytest.raises(MacroRefitTopologyError) as caught:
        refit_macro(replace(chained, encoder=corrupted), full)
    assert caught.value.diagnostics["reason"] == "corrupted_source_merge_map"


def test_same_rank_topology_with_shifted_distribution_is_allowed_without_population_claim():
    source = _counterexample_frame()
    merged = merge_macro_states(kmeans_macro(source, ["x"], 3), 0, 1)
    shifted = pd.DataFrame({"x": [100.0] * 4 + [1000.0] * 4 + [10000.0] * 4})

    refitted = refit_macro(merged, shifted)

    assert refitted.state_count == 2
    assert refitted.metadata["refit_topology"]["observed_states"] == 3
    # The encoder estimates changed. Matching canonical ranks preserves the
    # recipe's ordering rule; it cannot identify the same real-world groups.
    assert refitted.encoder["centers"] != merged.encoder["centers"]
    assert refitted.metadata["refit_topology"]["identity"] == "canonical_lexicographic_centroid_rank"


def test_frozen_application_does_not_mutate_fitted_parameters_or_leak_validation_values():
    full = _counterexample_frame()
    merged = merge_macro_states(kmeans_macro(full, ["x"], 3), 0, 1)
    train = pd.DataFrame({"x": [0.0] * 4 + [10.0] * 4 + [20.0] * 4})
    fitted = refit_macro(merged, train)
    before = json.loads(json.dumps(fitted.encoder))
    validation_a = pd.DataFrame({"x": [1.0, 11.0, 21.0], "outcome": [0.0, 1.0, 2.0]})
    validation_b = pd.DataFrame({"x": [1e9, -1e9, 21.0], "outcome": [1e12, -1e12, 2.0]})

    labels_a = apply_macro(fitted, validation_a).labels.tolist()
    labels_b = apply_macro(fitted, validation_b).labels.tolist()

    assert labels_a != labels_b  # application uses the changed validation features
    assert json.loads(json.dumps(fitted.encoder)) == before
    assert fitted.metadata["encoder"] == before
