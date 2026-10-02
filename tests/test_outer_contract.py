"""End-to-end holdout contract tests for missingness and schema poisoning.

These assert observable discovery behavior. They intentionally avoid private split
planner internals so they remain useful if the implementation changes.
"""

import numpy as np
import pandas as pd
import pytest

from causal_emergence_discovery.discovery import DiscoveryConfig, run_discovery
from causal_emergence_discovery.panel import build_lagged_table
from causal_emergence_discovery.spec import StudySpec

MODES = ("entity_holdout", "forward_time", "within_entity_interpolation")


def _panel(periods: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(421)
    rows = []
    for entity in range(16):
        previous = float(rng.normal())
        for time in range(periods):
            x = float(rng.normal() + entity / 10)
            z = float(rng.normal())
            rows.append({"entity": entity, "time": time, "x": x, "z": z, "y": previous})
            previous = 1.4 * x - 0.3 * z + float(rng.normal(scale=0.2))
    return pd.DataFrame(rows)


def _spec(*, declared_numeric: bool = True) -> StudySpec:
    columns = {
        "x": {"role": "context", **({"type": "numeric"} if declared_numeric else {})},
        "z": {"role": "context", "type": "numeric"},
        "y": {"role": "outcome", "type": "numeric", "allowed_as_cause": False,
              "allowed_as_adjustment": False},
    }
    return StudySpec.from_dict({
        "dataset": {"id_column": "entity", "time_column": "time"},
        "outcomes": ["y"],
        "columns": columns,
    })


def _config(mode: str, lag: int = 2) -> DiscoveryConfig:
    return DiscoveryConfig(
        outcome="y", lag=lag, validation_mode=mode, folds=2,
        holdout_fraction=0.2, max_states=3, paths=1,
        branching_factor=1, top_k=3, seed=19,
    )


def _run(frame: pd.DataFrame, spec: StudySpec, mode: str, lag: int = 2) -> dict:
    return run_discovery(frame, spec, _config(mode, lag))


def _split_signature(result: dict) -> tuple:
    """Membership and purge positions across outer and inner plans."""
    outer = result["validation"]["outer"]
    inner = result["validation"]["inner"]
    return tuple(
        (tuple(s["train_indices"]), tuple(s["test_indices"]),
         tuple(s["purged_indices"]), tuple(s["dropped_indices"]))
        for s in (outer, *inner)
    )


def _poison_reserved_targets(frame: pd.DataFrame, spec: StudySpec, result: dict, lag: int) -> pd.DataFrame:
    lagged = build_lagged_table(frame, spec, outcomes=["y"], lag=lag).data
    test = lagged.iloc[result["validation"]["outer"]["test_indices"]]
    poisoned = frame.copy()
    # Mutate only the actual target observations of reserved test rows. For
    # entity holdout, making every outcome missing for each reserved entity is
    # the strongest form of the original counterexample.
    if result["config"]["validation_mode"] == "entity_holdout":
        ids = set(test["entity"].tolist())
        poisoned.loc[poisoned.entity.isin(ids), "y"] = np.nan
    else:
        target_keys = set(zip(test.entity.tolist(), test["__lead_time"].tolist()))
        for entity, target_time in target_keys:
            poisoned.loc[(poisoned.entity == entity) & (poisoned.time == target_time), "y"] = np.nan
    return poisoned


def _poison_one_reserved_target(frame: pd.DataFrame, spec: StudySpec, result: dict, lag: int) -> pd.DataFrame:
    lagged = build_lagged_table(frame, spec, outcomes=["y"], lag=lag).data
    test = lagged.iloc[result["validation"]["outer"]["test_indices"]]
    row = test.iloc[0]
    poisoned = frame.copy()
    poisoned.loc[(poisoned.entity == row.entity) & (poisoned.time == row["__lead_time"]), "y"] = np.nan
    return poisoned


@pytest.mark.parametrize("mode", MODES)
def test_missing_reserved_targets_never_replace_outer_cohort(mode: str):
    frame, spec, lag = _panel(periods=14), _spec(), 2
    baseline = _run(frame, spec, mode, lag)
    poisoned = _poison_reserved_targets(frame, spec, baseline, lag)
    after = _run(poisoned, spec, mode, lag)

    # The contract protects both the reserved membership and its temporal
    # purges, even when the reserved labels become unusable.
    assert _split_signature(after) == _split_signature(baseline)
    assert after["searches"] == baseline["searches"]
    assert after["top_macros"] == baseline["top_macros"]
    outer_eval = after["outer_evaluation"]
    assert outer_eval["status"] == "not_evaluable"
    assert outer_eval["reason"]
    assert all(outer_eval[key] is None for key in ("macro_r2", "micro_r2", "predictive_r2_difference"))
    assert outer_eval["finite_test_targets"] < 2


@pytest.mark.parametrize("mode", MODES)
def test_one_missing_reserved_target_reduces_support_without_changing_split(mode: str):
    frame, spec, lag = _panel(periods=14), _spec(), 2
    baseline = _run(frame, spec, mode, lag)
    poisoned = _poison_one_reserved_target(frame, spec, baseline, lag)
    after = _run(poisoned, spec, mode, lag)

    assert _split_signature(after) == _split_signature(baseline)
    assert after["searches"] == baseline["searches"]
    assert after["outer_evaluation"]["status"] == baseline["outer_evaluation"]["status"]
    assert after["outer_evaluation"]["finite_test_targets"] == (
        baseline["outer_evaluation"]["finite_test_targets"] - 1
    )


@pytest.mark.parametrize("mode", MODES)
def test_reserved_predictor_magnitude_and_missingness_do_not_change_development_search(mode: str):
    frame, spec, lag = _panel(periods=14), _spec(), 2
    baseline = _run(frame, spec, mode, lag)
    lagged = build_lagged_table(frame, spec, outcomes=["y"], lag=lag).data
    row = lagged.iloc[baseline["validation"]["outer"]["test_indices"][0]]
    selector = (frame.entity == row.entity) & (frame.time == row.time)

    for value in (1e12, np.nan):
        changed = frame.copy()
        changed.loc[selector, "x"] = value
        after = _run(changed, spec, mode, lag)
        assert _split_signature(after) == _split_signature(baseline)
        assert after["searches"] == baseline["searches"]
        assert after["top_macros"] == baseline["top_macros"]


@pytest.mark.parametrize("mode", MODES)
def test_heldout_only_string_target_reduces_support_without_changing_search(mode: str):
    frame, spec, lag = _panel(periods=14), _spec(), 2
    baseline = _run(frame, spec, mode, lag)
    lagged = build_lagged_table(frame, spec, outcomes=["y"], lag=lag).data
    row = lagged.iloc[baseline["validation"]["outer"]["test_indices"][0]]
    changed = frame.copy()
    changed["y"] = changed.y.astype(object)
    changed.loc[(changed.entity == row.entity) & (changed.time == row["__lead_time"]), "y"] = "heldout-only-outcome"

    after = _run(changed, spec, mode, lag)

    assert _split_signature(after) == _split_signature(baseline)
    assert after["searches"] == baseline["searches"]
    assert after["top_macros"] == baseline["top_macros"]
    assert after["outer_evaluation"]["finite_test_targets"] == (
        baseline["outer_evaluation"]["finite_test_targets"] - 1
    )
    assert after["outer_evaluation"]["status"] == "evaluated"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("declared_numeric", [True, False], ids=["declared", "train-inferred"])
def test_heldout_only_string_preserves_numeric_feature_under_frozen_schema(mode: str, declared_numeric: bool):
    frame, spec, lag = _panel(periods=14), _spec(declared_numeric=declared_numeric), 2
    baseline = _run(frame, spec, mode, lag)
    lagged = build_lagged_table(frame, spec, outcomes=["y"], lag=lag).data
    row = lagged.iloc[baseline["validation"]["outer"]["test_indices"][0]]
    changed = frame.copy()
    changed["x"] = changed["x"].astype(object)
    changed.loc[(changed.entity == row.entity) & (changed.time == row.time), "x"] = "heldout-only-type"

    after = _run(changed, spec, mode, lag)

    assert _split_signature(after) == _split_signature(baseline)
    assert after["searches"] == baseline["searches"]
    assert after["top_macros"] == baseline["top_macros"]
    assert after["columns"]["macro_features"] == baseline["columns"]["macro_features"]
    assert any("x" in str(search["initial_macro"]) for search in after["searches"])
    schema = after["feature_schema"]
    assert schema["x"]["kind"] == "numeric"
    assert schema["x"]["source"] == ("declared" if declared_numeric else "outer_training_inference")
    assert schema["x"]["invalid_outer_values"] == 1


def test_lag_two_split_windows_remain_valid_after_outcome_poisoning():
    frame, spec, lag = _panel(periods=14), _spec(), 2
    for mode in MODES:
        result = _run(frame, spec, mode, lag)
        lagged = build_lagged_table(frame, spec, outcomes=["y"], lag=lag).data
        outer = result["validation"]["outer"]
        train = lagged.iloc[outer["train_indices"]]
        test = lagged.iloc[outer["test_indices"]]
        if mode == "entity_holdout":
            assert set(train.entity).isdisjoint(test.entity)
        elif mode == "forward_time":
            assert train["__lead_time"].max() < test.time.min()
            assert set(outer["purged_indices"])
        else:
            for entity in test.entity.unique():
                train_entity = train.loc[train.entity == entity]
                test_entity = test.loc[test.entity == entity]
                assert not set(train_entity["__lead_time"]) & set(test_entity.time)
                assert not set(train_entity.time) & set(test_entity["__lead_time"])
