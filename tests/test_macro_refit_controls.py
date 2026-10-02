"""Regression controls for refitting merged macro recipes."""

import numpy as np
import pandas as pd
import pytest

from causal_emergence_discovery.macro import (
    MacroRefitTopologyError,
    apply_macro,
    composite_quantile_macro,
    kmeans_macro,
    merge_macro_states,
    quantile_macro,
    refit_macro,
)


def _three_cluster_frame():
    return pd.DataFrame({"x": [0.0] * 4 + [10.0] * 4 + [20.0] * 4})


def test_kmeans_merged_refit_rejects_absent_original_cluster_instead_of_compacting_ids():
    selected = merge_macro_states(kmeans_macro(_three_cluster_frame(), ["x"], 3), 0, 1)
    fold_train = pd.DataFrame({"x": [10.0] * 4 + [20.0] * 4})

    with pytest.raises(MacroRefitTopologyError) as caught:
        refit_macro(selected, fold_train)

    assert caught.value.recipe == "kmeans"
    assert caught.value.expected_states == 3
    assert caught.value.observed_states == 2
    assert caught.value.diagnostics["reason"] == "incomplete_refit_topology"


def test_kmeans_merged_refit_rejects_empty_requested_cluster():
    selected = merge_macro_states(kmeans_macro(_three_cluster_frame(), ["x"], 3), 0, 2)
    # Only two unique points remain, so the requested third center cannot be occupied.
    fold_train = pd.DataFrame({"x": [0.0] * 3 + [10.0] * 3})

    with pytest.raises(MacroRefitTopologyError) as caught:
        refit_macro(selected, fold_train)

    assert caught.value.expected_states == 3
    assert caught.value.observed_states == 2


def test_quantile_merged_refit_rejects_fewer_cutpoints_from_ties():
    selected = merge_macro_states(quantile_macro(pd.DataFrame({"x": np.arange(5.0)}), "x", 4), 0, 1)
    tied_fold = pd.DataFrame({"x": [0.0] * 8})

    with pytest.raises(MacroRefitTopologyError) as caught:
        refit_macro(selected, tied_fold)

    assert caught.value.recipe == "quantile"
    assert caught.value.expected_cutpoints == 3
    assert caught.value.observed_cutpoints < 3


def test_quantile_merged_refit_rejects_missing_occupied_rank_even_when_cut_count_matches():
    selected = merge_macro_states(quantile_macro(pd.DataFrame({"x": np.arange(5.0)}), "x", 4), 0, 1)
    # The cutpoints remain distinct, but the exact middle threshold leaves raw rank 2 empty.
    fold_train = pd.DataFrame({"x": [0.0, 1.0, 2.0]})

    with pytest.raises(MacroRefitTopologyError) as caught:
        refit_macro(selected, fold_train)

    assert caught.value.expected_cutpoints == 3
    assert caught.value.observed_cutpoints == 3
    assert caught.value.expected_states == 4
    assert caught.value.observed_states == 3


def test_composite_quantile_merged_refit_rejects_reduced_topology():
    train = pd.DataFrame({"a": np.arange(5.0), "b": np.arange(5.0)})
    selected = merge_macro_states(composite_quantile_macro(train, ["a", "b"], 4), 0, 1)
    tied_fold = pd.DataFrame({"a": [0.0] * 8, "b": [0.0] * 8})

    with pytest.raises(MacroRefitTopologyError) as caught:
        refit_macro(selected, tied_fold)

    assert caught.value.recipe == "composite_quantile"
    assert caught.value.expected_cutpoints == 3
    assert caught.value.observed_cutpoints < 3


def test_chained_merges_replay_through_current_map_and_both_endpoints():
    original = quantile_macro(pd.DataFrame({"x": np.arange(8.0)}), "x", 4)
    first = merge_macro_states(original, 0, 1)
    second = merge_macro_states(first, 2, 3)
    chained = merge_macro_states(second, 0, 2)

    refitted = refit_macro(chained, pd.DataFrame({"x": np.arange(8.0)}))

    assert refitted.labels.tolist() == [0] * 8
    assert refitted.encoder["state_map"] == {"0": 0, "1": 0, "2": 0, "3": 0}
    # Applying the saved map to later data must preserve the full chained merge.
    assert apply_macro(refitted, pd.DataFrame({"x": [0.0, 2.0, 7.0]})).labels.tolist() == [0, 0, 0]


def test_quantile_merged_refit_rejects_source_with_unoccupied_requested_rank():
    # The source has three distinct cutpoints but only ranks 0, 1, and 3 populated.
    reduced_source = quantile_macro(pd.DataFrame({"x": [0.0, 1.0, 2.0]}), "x", 4)
    selected = merge_macro_states(reduced_source, 0, 1)

    with pytest.raises(MacroRefitTopologyError) as caught:
        refit_macro(selected, pd.DataFrame({"x": np.arange(5.0)}))

    assert caught.value.diagnostics["reason"] == "unsupported_source_topology"
    assert caught.value.diagnostics["phase"] == "source"


def test_kmeans_ids_are_canonical_center_ranks_and_frozen_apply_uses_saved_centers():
    train = pd.DataFrame({
        "x": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        "z": [0.0, 1.0, 8.0, 9.0, 0.0, 1.0, 8.0, 9.0],
    })
    macro = kmeans_macro(train, ["x", "z"], 3)
    permuted = kmeans_macro(train.sample(frac=1, random_state=4), ["x", "z"], 3)

    centers = np.asarray(macro.encoder["centers"])
    assert [tuple(row) for row in centers] == sorted(tuple(row) for row in centers)
    assert np.asarray(permuted.encoder["centers"]).tolist() == centers.tolist()
    assert apply_macro(macro, train).labels.tolist() == apply_macro(permuted, train).labels.tolist()

    one_dimensional = kmeans_macro(_three_cluster_frame(), ["x"], 3)
    assert apply_macro(one_dimensional, pd.DataFrame({"x": [0.0, 10.0, 20.0]})).labels.tolist() == [0, 1, 2]

    future = pd.DataFrame({"x": [0.0, 10.0, 20.0]})
    before = apply_macro(one_dimensional, future).labels
    # A refit learns new centers from only its supplied training rows.
    refitted = refit_macro(one_dimensional, pd.DataFrame({"x": [100.0] * 4 + [110.0] * 4 + [120.0] * 4}))
    assert refitted.encoder["preprocess"]["x"]["mean"] != one_dimensional.encoder["preprocess"]["x"]["mean"]
    assert apply_macro(one_dimensional, future).labels.equals(before)


@pytest.mark.parametrize("recipe", ["quantile", "composite_quantile", "kmeans"])
def test_unmerged_reduced_fit_is_allowed_and_reports_requested_vs_observed_topology(recipe):
    if recipe == "quantile":
        candidate = quantile_macro(pd.DataFrame({"x": np.arange(8.0)}), "x", 4)
        reduced_train = pd.DataFrame({"x": [0.0] * 4 + [1.0] * 4})
    elif recipe == "composite_quantile":
        candidate = composite_quantile_macro(pd.DataFrame({"a": np.arange(8.0), "b": np.arange(8.0)}), ["a", "b"], 4)
        reduced_train = pd.DataFrame({"a": [0.0] * 4 + [1.0] * 4, "b": [0.0] * 4 + [1.0] * 4})
    else:
        candidate = kmeans_macro(_three_cluster_frame(), ["x"], 3)
        reduced_train = pd.DataFrame({"x": [10.0] * 4 + [20.0] * 4})

    reduced = refit_macro(candidate, reduced_train)

    assert reduced.state_count == 2
    topology = reduced.metadata["refit_topology"]
    assert topology["recipe"] == recipe
    assert topology["requested_states"] == (3 if recipe == "kmeans" else 4)
    assert topology["observed_states"] == 2
    assert topology["merged"] is False
