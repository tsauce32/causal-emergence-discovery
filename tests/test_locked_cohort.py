"""Outcome availability must not alter the planned panel cohort."""

import numpy as np
import pandas as pd
import pytest

from causal_emergence_discovery.panel import build_lagged_table
from causal_emergence_discovery.spec import StudySpec
from causal_emergence_discovery.validation import build_validation_plan, target_support_audit


SPEC = StudySpec.from_dict(
    {
        "dataset": {"id_column": "entity", "time_column": "time"},
        "outcomes": ["y"],
    }
)


def panel(entities=12, periods=12):
    return pd.DataFrame(
        [
            {"entity": entity, "time": time, "x": entity + time / 10, "y": entity - time}
            for entity in range(entities)
            for time in range(periods)
        ]
    )


def plan_for(data, target_time_column, mode):
    return build_validation_plan(
        data,
        id_column="entity",
        time_column="time",
        target_time_column=target_time_column,
        mode=mode,
        folds=2,
        holdout_fraction=0.25,
        seed=17,
    )


@pytest.mark.parametrize(
    "mode", ["entity_holdout", "forward_time", "within_entity_interpolation"]
)
def test_missing_reserved_targets_do_not_replace_cohort(mode):
    source = panel()
    original = build_lagged_table(source, SPEC)
    original_plan = plan_for(original.data, original.target_time_column, mode)

    reserved = original.data.iloc[list(original_plan.outer.test_indices)]
    unavailable_targets = set(zip(reserved.entity, reserved[original.target_time_column]))
    masked_source = source.copy()
    mask = [
        (entity, time) in unavailable_targets
        for entity, time in zip(masked_source.entity, masked_source.time)
    ]
    masked_source.loc[mask, "y"] = np.nan

    masked = build_lagged_table(masked_source, SPEC)
    masked_plan = plan_for(masked.data, masked.target_time_column, mode)

    # All split positions and boundaries are fixed by IDs and time geometry.
    assert len(masked.data) == len(original.data)
    assert masked_plan.to_dict() == original_plan.to_dict()
    audit = target_support_audit(masked.data, masked_plan, masked.target_columns["y"])
    assert audit["outer"]["reserved_test_rows"] == len(original_plan.outer.test_indices)
    assert audit["outer"]["usable_test_rows"] == 0
    assert audit["outer"]["ineligible_test_indices"] == list(original_plan.outer.test_indices)


def test_lag_tail_is_removed_by_target_time_and_finite_eligibility_is_separate():
    source = panel(entities=3, periods=8)
    source.loc[(source.entity == 0) & (source.time == 2), "y"] = np.nan
    source.loc[(source.entity == 1) & (source.time == 4), "y"] = np.inf
    lagged = build_lagged_table(source, SPEC, lag=2)

    # Only the two positional lag tails per entity are omitted from the cohort.
    assert len(lagged.data) == 3 * (8 - 2)
    assert lagged.data.groupby("entity").size().to_dict() == {0: 6, 1: 6, 2: 6}
    eligible = lagged.target_eligibility["y"]
    target = lagged.target_columns["y"]
    assert not bool(lagged.data.loc[lagged.data[target].isna(), eligible].iloc[0])
    assert not bool(lagged.data.loc[np.isinf(lagged.data[target]), eligible].iloc[0])
    assert lagged.data[eligible].dtype == bool
