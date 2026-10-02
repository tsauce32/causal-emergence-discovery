import numpy as np
import pandas as pd
import pytest

from causal_emergence_discovery.validation import build_validation_plan


def test_entity_holdout_keeps_fingerprinted_entities_whole_and_is_deterministic():
    df = _lagged_panel(entity_count=30, periods=5)
    # This feature identifies the entity perfectly. A row split would put each
    # fingerprint in both sides, while entity holdout must keep them separated.
    first = build_validation_plan(
        df, id_column="entity", time_column="time", target_time_column="target_time",
        folds=4, seed=17,
    )
    second = build_validation_plan(
        df, id_column="entity", time_column="time", target_time_column="target_time",
        folds=4, seed=17,
    )

    assert first.to_dict() == second.to_dict()
    for split in (first.outer, *first.inner):
        train = set(df.iloc[list(split.train_indices)].entity)
        test = set(df.iloc[list(split.test_indices)].entity)
        assert train.isdisjoint(test)
    assert first.audit["outcomes_inspected_for_selection"] is False

    altered = df.assign(outcome=np.linspace(1e9, -1e9, len(df)))
    outcome_changed = build_validation_plan(
        altered, id_column="entity", time_column="time", target_time_column="target_time",
        folds=4, seed=17,
    )
    assert first.outer.train_indices == outcome_changed.outer.train_indices
    assert first.outer.test_indices == outcome_changed.outer.test_indices


def test_forward_time_uses_future_holdout_and_purges_crossing_targets():
    df = _lagged_panel(entity_count=4, periods=12)
    plan = build_validation_plan(
        df, id_column="entity", time_column="time", target_time_column="target_time",
        mode="forward_time", folds=3, holdout_fraction=0.25,
    )

    outer = plan.outer
    train = df.iloc[list(outer.train_indices)]
    test = df.iloc[list(outer.test_indices)]
    boundary = test.time.min()
    assert test.time.min() == 9
    assert train.time.max() < boundary
    assert train.target_time.max() < boundary
    # The predictor at t=8 targets t=9, so every such row must be purged.
    assert all(df.iloc[i].time == 8 for i in outer.purged_indices)
    assert len(outer.purged_indices) == 4

    for split in plan.inner:
        fold_train = df.iloc[list(outer.train_indices)].reset_index(drop=True).iloc[list(split.train_indices)]
        fold_test = df.iloc[list(outer.train_indices)].reset_index(drop=True).iloc[list(split.test_indices)]
        assert fold_train.time.max() < fold_test.time.min()
        assert fold_train.target_time.max() < fold_test.time.min()


def test_within_entity_interpolation_purges_overlapping_outcome_windows():
    df = _lagged_panel(entity_count=6, periods=14)
    plan = build_validation_plan(
        df, id_column="entity", time_column="time", target_time_column="target_time",
        mode="within_entity_interpolation", folds=2, holdout_fraction=0.15,
    )

    assert "already observed entities" in plan.audit["interpretation_limit"]
    for split, frame in [
        (plan.outer, df),
    ]:
        training = frame.iloc[list(split.train_indices)]
        validation = frame.iloc[list(split.test_indices)]
        assert set(training.entity) == set(validation.entity)
        for entity in set(validation.entity):
            v = validation.loc[validation.entity == entity]
            t = training.loc[training.entity == entity]
            forbidden = set(v.time) | set(v.target_time)
            assert not set(t.target_time) & forbidden
            assert not set(t.time) & set(v.target_time)
    inner_frame = df.iloc[list(plan.outer.train_indices)].reset_index(drop=True)
    for split in plan.inner:
        training = inner_frame.iloc[list(split.train_indices)]
        validation = inner_frame.iloc[list(split.test_indices)]
        assert set(training.entity) == set(validation.entity)
        for entity in set(validation.entity):
            v = validation.loc[validation.entity == entity]
            t = training.loc[training.entity == entity]
            forbidden = set(v.time) | set(v.target_time)
            assert not set(t.target_time) & forbidden
            assert not set(t.time) & set(v.target_time)


def test_planner_rejects_insufficient_entity_folds_instead_of_reducing_them():
    df = _lagged_panel(entity_count=3, periods=3)
    with pytest.raises(ValueError, match=r"folds \+ 1 \(4\).*outer-training entities"):
        build_validation_plan(
            df, id_column="entity", time_column="time", target_time_column="target_time",
            folds=3,
        )


def test_forward_planner_rejects_missing_target_timestamp_metadata():
    df = _lagged_panel(entity_count=5, periods=8).drop(columns="target_time")
    with pytest.raises(ValueError, match="Validation columns are missing"):
        build_validation_plan(
            df, id_column="entity", time_column="time", target_time_column="target_time",
            mode="forward_time", folds=2,
        )


def test_planner_rejects_non_forward_target_timestamps():
    df = _lagged_panel(entity_count=5, periods=8)
    df.loc[0, "target_time"] = df.loc[0, "time"]
    with pytest.raises(ValueError, match="strictly after"):
        build_validation_plan(
            df, id_column="entity", time_column="time", target_time_column="target_time",
            folds=2,
        )


def test_datetime_and_shuffled_irregular_rows_keep_true_future_order():
    rows = []
    times = [
        "2025-01-01", "2025-01-02", "2025-01-10", "2025-02-01", "2025-03-15",
        "2025-04-01", "2025-04-17", "2025-05-03", "2025-06-20", "2025-08-01",
    ]
    for entity, count in enumerate([10, 9, 8, 10, 7, 9]):
        for index, time in enumerate(times[:count]):
            target_index = min(index + 1, count - 1)
            if target_index == index:
                continue
            rows.append({
                "entity": entity,
                "time": pd.Timestamp(time, tz="UTC"),
                "target_time": pd.Timestamp(times[target_index], tz="UTC"),
            })
    df = pd.DataFrame(rows).sample(frac=1, random_state=71).reset_index(drop=True)
    plan = build_validation_plan(
        df, id_column="entity", time_column="time", target_time_column="target_time",
        mode="forward_time", folds=2, holdout_fraction=0.3,
    )

    train = df.iloc[list(plan.outer.train_indices)]
    test = df.iloc[list(plan.outer.test_indices)]
    assert train.time.max() < test.time.min()
    assert train.target_time.max() < test.time.min()
    assert plan.audit["outer"]["train_target_time_max"] < plan.audit["outer"]["test_predictor_time_min"]
    assert set(plan.outer.dropped_indices).isdisjoint(plan.outer.purged_indices)


def test_forward_inner_advances_warmup_until_purge_leaves_support():
    # The first expanding-window cut would purge every warmup row because
    # target_time equals that cut. The planner advances the initial window.
    df = _lagged_panel(entity_count=2, periods=9)
    plan = build_validation_plan(
        df, id_column="entity", time_column="time", target_time_column="target_time",
        mode="forward_time", folds=2, holdout_fraction=0.2,
    )
    inner_frame = df.iloc[list(plan.outer.train_indices)].reset_index(drop=True)
    for split in plan.inner:
        train = inner_frame.iloc[list(split.train_indices)]
        test = inner_frame.iloc[list(split.test_indices)]
        assert len(train) >= 2
        assert train.target_time.max() < test.time.min()


def test_interpolation_uses_distinct_interior_rows_for_each_fold():
    df = _lagged_panel(entity_count=5, periods=8)
    plan = build_validation_plan(
        df, id_column="entity", time_column="time", target_time_column="target_time",
        mode="within_entity_interpolation", folds=2, holdout_fraction=0.2,
    )
    inner = [set(split.test_indices) for split in plan.inner]
    assert inner[0].isdisjoint(inner[1])


def test_interpolation_support_failure_is_explicit_for_short_trajectories():
    df = _lagged_panel(entity_count=5, periods=5)
    with pytest.raises(ValueError, match="at least three training rows per entity|too few interior observations"):
        build_validation_plan(
            df, id_column="entity", time_column="time", target_time_column="target_time",
            mode="within_entity_interpolation", folds=2,
        )


def _lagged_panel(entity_count: int, periods: int) -> pd.DataFrame:
    rows = []
    for entity in range(entity_count):
        for time in range(periods):
            rows.append(
                {
                    "entity": entity,
                    "fingerprint": entity * 1000 + time,
                    "time": time,
                    "target_time": time + 1,
                    "outcome": 5.0 * entity + time,
                }
            )
    return pd.DataFrame(rows)
