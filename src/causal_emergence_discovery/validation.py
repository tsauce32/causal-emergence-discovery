"""Deterministic, auditable train/test plans for longitudinal panels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataSplit:
    """Row positions for one split; all positions are local to its source frame."""

    name: str
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    dropped_indices: tuple[int, ...] = ()
    purged_indices: tuple[int, ...] = ()
    audit: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "train_indices": list(self.train_indices),
            "test_indices": list(self.test_indices),
            "dropped_indices": list(self.dropped_indices),
            "purged_indices": list(self.purged_indices),
            "audit": self.audit,
        }


@dataclass(frozen=True)
class ValidationPlan:
    """Outer holdout and inner folds, with indices local to documented frames."""

    outer: DataSplit
    inner: tuple[DataSplit, ...]
    mode: str
    audit: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "outer": self.outer.to_dict(),
            "inner": [split.to_dict() for split in self.inner],
            "audit": self.audit,
        }


_MODES = {"entity_holdout", "forward_time", "within_entity_interpolation"}


def build_validation_plan(
    df: pd.DataFrame,
    *,
    id_column: str,
    time_column: str,
    target_time_column: str,
    mode: str = "entity_holdout",
    folds: int = 5,
    holdout_fraction: float = 0.2,
    seed: int = 0,
) -> ValidationPlan:
    """Plan outer and inner splits without inspecting outcome or predictor values.

    Outer indices address ``df``. Inner indices address a reset-index copy of
    the outer training rows. Only entity IDs and the predictor/target timestamps
    are read while choosing rows.
    """
    if mode not in _MODES:
        raise ValueError(f"Unknown validation mode {mode!r}; choose from {sorted(_MODES)}.")
    if not isinstance(folds, int) or folds < 2:
        raise ValueError("folds must be an integer of at least 2.")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be strictly between 0 and 1.")
    missing = [c for c in (id_column, time_column, target_time_column) if c not in df.columns]
    if missing:
        raise ValueError(f"Validation columns are missing: {missing}.")
    if len(df) < 4:
        raise ValueError("Validation needs at least four rows to form train and test sets.")
    if df[id_column].isna().any():
        raise ValueError(f"id_column {id_column!r} contains missing values.")
    if df[time_column].isna().any() or df[target_time_column].isna().any():
        raise ValueError("Predictor and target timestamps must be complete.")

    frame = df.reset_index(drop=True)
    predictor_keys = _time_keys(frame[time_column], time_column)
    target_keys = _time_keys(frame[target_time_column], target_time_column)
    if any(target <= predictor for predictor, target in zip(predictor_keys, target_keys)):
        raise ValueError("Every target timestamp must be strictly after its predictor timestamp.")

    if mode == "entity_holdout":
        outer = _entity_outer(frame, id_column, holdout_fraction, seed)
        inner_frame = frame.iloc[list(outer.train_indices)].reset_index(drop=True)
        inner = _entity_inner(inner_frame, id_column, folds, seed + 1)
        limitation = None
    elif mode == "forward_time":
        outer = _forward_outer(frame, predictor_keys, target_keys, holdout_fraction)
        inner_frame = frame.iloc[list(outer.train_indices)].reset_index(drop=True)
        inner_predictor = [predictor_keys[i] for i in outer.train_indices]
        inner_target = [target_keys[i] for i in outer.train_indices]
        inner = _forward_inner(inner_frame, inner_predictor, inner_target, folds)
        limitation = None
    else:
        outer = _interpolation_split(
            frame, predictor_keys, target_keys, id_column, holdout_fraction, "outer"
        )
        inner_frame = frame.iloc[list(outer.train_indices)].reset_index(drop=True)
        inner_predictor = [predictor_keys[i] for i in outer.train_indices]
        inner_target = [target_keys[i] for i in outer.train_indices]
        inner = _interpolation_inner(
            inner_frame, inner_predictor, inner_target, id_column, folds
        )
        limitation = (
            "This estimates interpolation for already observed entities and time ranges. "
            "It does not measure transfer to unseen entities or forecasting beyond observed time."
        )

    _require_minimum(outer, "outer")
    for split in inner:
        _require_minimum(split, split.name)

    outer.audit.update(_split_coverage(frame, outer, id_column, time_column, target_time_column))
    for split in inner:
        split.audit.update(
            _split_coverage(inner_frame, split, id_column, time_column, target_time_column)
        )
    all_times = [_display_time(v) for v in predictor_keys]
    audit = {
        "index_scope": {
            "outer": "positions in the supplied lagged table",
            "inner": "positions in outer training rows after reset_index(drop=True)",
        },
        "row_count": int(len(frame)),
        "entity_count": int(frame[id_column].nunique()),
        "predictor_time_coverage": {
            "min": min(all_times), "max": max(all_times),
            "unique_count": int(len(set(predictor_keys))),
        },
        "outer": outer.audit,
        "inner": [split.audit for split in inner],
        "minimum_train_rows": int(min([len(outer.train_indices), *(len(s.train_indices) for s in inner)])),
        "minimum_test_rows": int(min([len(outer.test_indices), *(len(s.test_indices) for s in inner)])),
        "finite_timestamp_support": True,
        "selection_columns": [id_column, time_column, target_time_column],
        "outcomes_inspected_for_selection": False,
    }
    if limitation:
        audit["interpretation_limit"] = limitation
    return ValidationPlan(outer=outer, inner=tuple(inner), mode=mode, audit=audit)


def _entity_sort_key(value: Any) -> tuple[str, str, str, str]:
    """Break textual-ID ties by type, independent of first-seen row order.

    Ordinary scalar IDs keep their existing textual order. Distinct values
    such as integer 1 and string "1" receive stable, different sort keys.
    """
    kind = type(value)
    return str(value), kind.__module__, kind.__qualname__, repr(value)


def _entity_outer(df: pd.DataFrame, id_column: str, fraction: float, seed: int) -> DataSplit:
    entities = np.asarray(sorted(pd.unique(df[id_column]).tolist(), key=_entity_sort_key), dtype=object)
    test_count = max(1, int(np.ceil(len(entities) * fraction)))
    train_count = len(entities) - test_count
    if train_count < 1:
        raise ValueError("entity_holdout needs at least two distinct entities.")
    shuffled = np.random.default_rng(seed).permutation(entities)
    test_entities = set(shuffled[:test_count].tolist())
    entity_values = df[id_column].tolist()
    test = tuple(int(i) for i, entity in enumerate(entity_values) if entity in test_entities)
    test_set = set(test)
    train = tuple(i for i in range(len(df)) if i not in test_set)
    split = DataSplit("outer", train, test)
    _require_minimum(split, "entity_holdout outer split")
    return split


def _entity_inner(df: pd.DataFrame, id_column: str, folds: int, seed: int) -> list[DataSplit]:
    entities = np.asarray(sorted(pd.unique(df[id_column]).tolist(), key=_entity_sort_key), dtype=object)
    if len(entities) < folds + 1:
        raise ValueError(
            f"entity_holdout needs at least folds + 1 ({folds + 1}) outer-training entities; "
            f"found {len(entities)}. Reduce folds or provide more entities."
        )
    shuffled = np.random.default_rng(seed).permutation(entities)
    buckets = np.array_split(shuffled, folds)
    splits = []
    for fold, bucket in enumerate(buckets):
        test_entities = set(bucket.tolist())
        entity_values = df[id_column].tolist()
        test = tuple(i for i, entity in enumerate(entity_values) if entity in test_entities)
        test_set = set(test)
        train = tuple(i for i in range(len(df)) if i not in test_set)
        split = DataSplit(f"inner_{fold + 1}", train, test)
        _require_minimum(split, split.name)
        splits.append(split)
    return splits


def _forward_outer(
    df: pd.DataFrame, predictor: list[Any], target: list[Any], fraction: float
) -> DataSplit:
    times = sorted(set(predictor))
    test_count = max(1, int(np.ceil(len(times) * fraction)))
    if len(times) - test_count < 1:
        raise ValueError("forward_time needs predictor timestamps on both sides of a boundary.")
    boundary = times[-test_count]
    test = tuple(i for i, value in enumerate(predictor) if value >= boundary)
    raw_train = {i for i, value in enumerate(predictor) if value < boundary}
    train = tuple(i for i in sorted(raw_train) if target[i] < boundary)
    purged = tuple(sorted(raw_train - set(train)))
    excluded = set(train) | set(test) | set(purged)
    dropped = tuple(i for i in range(len(df)) if i not in excluded)
    split = DataSplit("outer", train, test, dropped, purged)
    _require_minimum(split, "forward_time outer split")
    return split


def _forward_inner(
    df: pd.DataFrame, predictor: list[Any], target: list[Any], folds: int
) -> list[DataSplit]:
    times = sorted(set(predictor))
    latest_start = len(times) - folds
    initial = next(
        (
            boundary_index
            for boundary_index in range(1, latest_start + 1)
            if sum(
                predictor[i] < times[boundary_index] and target[i] < times[boundary_index]
                for i in range(len(df))
            ) >= 2
        ),
        None,
    )
    if initial is None:
        raise ValueError(
            f"forward_time cannot form {folds} inner validation windows with at least two "
            "training rows after timestamp-boundary purging. Reduce folds or provide a longer panel."
        )
    test_blocks = np.array_split(np.asarray(times[initial:], dtype=object), folds)
    splits = []
    for fold, block in enumerate(test_blocks):
        if len(block) == 0:
            raise ValueError(f"forward_time cannot form inner fold {fold + 1} with these timestamps.")
        boundary = block[0]
        block_set = set(block.tolist())
        test = tuple(i for i, value in enumerate(predictor) if value in block_set)
        raw_train = {i for i, value in enumerate(predictor) if value < boundary}
        train = tuple(i for i in sorted(raw_train) if target[i] < boundary)
        purged = tuple(sorted(raw_train - set(train)))
        used = set(train) | set(test)
        dropped = tuple(i for i in range(len(df)) if i not in used | set(purged))
        split = DataSplit(f"inner_{fold + 1}", train, test, dropped, purged)
        _require_minimum(split, split.name)
        splits.append(split)
    return splits


def _interpolation_split(
    df: pd.DataFrame,
    predictor: list[Any],
    target: list[Any],
    id_column: str,
    fraction: float,
    name: str,
) -> DataSplit:
    test_candidates: set[int] = set()
    group_rows = _entity_rows(df, id_column, predictor)
    for indices in group_rows.values():
        n = len(indices)
        count = max(1, int(np.ceil(n * fraction)))
        if n < count + 3:
            raise ValueError(
                "within_entity_interpolation needs at least three training rows per entity "
                "after selecting a test block and purging overlap; provide longer trajectories."
            )
        start = max(1, (n - count) // 2)
        chosen = indices[start : start + count]
        if len(chosen) != count or start + count > n - 1:
            raise ValueError("within_entity_interpolation cannot place an interior test block.")
        test_candidates.update(chosen)
    train, purged = _purge_interpolation(df, predictor, target, id_column, test_candidates)
    test = tuple(sorted(test_candidates))
    dropped = tuple(i for i in range(len(df)) if i not in set(train) | set(test) | set(purged))
    split = DataSplit(name, train, test, dropped, purged)
    for entity, indices in group_rows.items():
        if not any(i in train for i in indices):
            raise ValueError(
                f"within_entity_interpolation cannot retain entity {entity!r} in training "
                "after purging overlapping outcome windows."
            )
    return split


def _interpolation_inner(
    df: pd.DataFrame,
    predictor: list[Any],
    target: list[Any],
    id_column: str,
    folds: int,
) -> list[DataSplit]:
    groups = _entity_rows(df, id_column, predictor)
    splits: list[DataSplit] = []
    for fold in range(folds):
        selected: set[int] = set()
        for indices in groups.values():
            n = len(indices)
            interior = indices[1:-1]
            if interior:
                buckets = np.array_split(np.asarray(interior, dtype=int), folds)
                bucket = buckets[fold]
                if len(bucket):
                    selected.add(int(bucket[len(bucket) // 2]))
        if len(selected) < 2:
            raise ValueError(
                f"within_entity_interpolation cannot form inner fold {fold + 1}: "
                "too few interior observations. Reduce folds or provide longer trajectories."
            )
        train, purged = _purge_interpolation(df, predictor, target, id_column, selected)
        for entity, indices in groups.items():
            if not any(i in train for i in indices):
                raise ValueError(
                    f"within_entity_interpolation inner fold {fold + 1} cannot retain "
                    f"entity {entity!r} in training after overlap purging."
                )
        test = tuple(sorted(selected))
        dropped = tuple(i for i in range(len(df)) if i not in set(train) | set(test) | set(purged))
        split = DataSplit(f"inner_{fold + 1}", train, test, dropped, purged)
        _require_minimum(split, split.name)
        splits.append(split)
    return splits


def _purge_interpolation(
    df: pd.DataFrame,
    predictor: list[Any],
    target: list[Any],
    id_column: str,
    test: set[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    by_entity: dict[Any, list[int]] = {}
    for i, entity in enumerate(df[id_column].tolist()):
        by_entity.setdefault(entity, []).append(i)
    purge: set[int] = set()
    for entity, indices in by_entity.items():
        test_rows = [i for i in indices if i in test]
        if not test_rows:
            continue
        test_start = min(predictor[i] for i in test_rows)
        test_end = max(target[i] for i in test_rows)
        for i in indices:
            if i in test:
                continue
            # Samples whose predictor-to-target interval touches a held-out
            # target window cannot contribute to training.
            if predictor[i] <= test_end and target[i] >= test_start:
                purge.add(i)
    train = tuple(i for i in range(len(df)) if i not in test and i not in purge)
    return train, tuple(sorted(purge))


def _entity_rows(df: pd.DataFrame, id_column: str, predictor: list[Any]) -> dict[Any, list[int]]:
    result: dict[Any, list[int]] = {}
    for i, entity in enumerate(df[id_column].tolist()):
        result.setdefault(entity, []).append(i)
    for indices in result.values():
        indices.sort(key=lambda i: (predictor[i], i))
    return result


def _split_coverage(
    df: pd.DataFrame,
    split: DataSplit,
    id_column: str,
    time_column: str,
    target_time_column: str,
) -> dict[str, Any]:
    train_ids = df.iloc[list(split.train_indices)][id_column]
    test_ids = df.iloc[list(split.test_indices)][id_column]
    train_time = df.iloc[list(split.train_indices)][time_column]
    test_time = df.iloc[list(split.test_indices)][time_column]
    return {
        "name": split.name,
        "train_rows": int(len(split.train_indices)),
        "test_rows": int(len(split.test_indices)),
        "dropped_rows": int(len(split.dropped_indices)),
        "purged_rows": int(len(split.purged_indices)),
        "train_entities": int(train_ids.nunique()),
        "test_entities": int(test_ids.nunique()),
        "entity_overlap": int(len(set(train_ids.tolist()) & set(test_ids.tolist()))),
        "train_predictor_time_min": _display_time(min(_time_keys(train_time, time_column))) if len(train_time) else None,
        "train_predictor_time_max": _display_time(max(_time_keys(train_time, time_column))) if len(train_time) else None,
        "train_target_time_min": _display_time(min(_time_keys(df.iloc[list(split.train_indices)][target_time_column], target_time_column))) if split.train_indices else None,
        "train_target_time_max": _display_time(max(_time_keys(df.iloc[list(split.train_indices)][target_time_column], target_time_column))) if split.train_indices else None,
        "test_predictor_time_min": _display_time(min(_time_keys(test_time, time_column))) if len(test_time) else None,
        "test_predictor_time_max": _display_time(max(_time_keys(test_time, time_column))) if len(test_time) else None,
        "test_target_time_min": _display_time(min(_time_keys(df.iloc[list(split.test_indices)][target_time_column], target_time_column))) if split.test_indices else None,
        "test_target_time_max": _display_time(max(_time_keys(df.iloc[list(split.test_indices)][target_time_column], target_time_column))) if split.test_indices else None,
        "target_time_column": target_time_column,
    }


def _require_minimum(split: DataSplit, label: str) -> None:
    if len(split.train_indices) < 2 or len(split.test_indices) < 2:
        raise ValueError(
            f"{label} needs at least 2 training and 2 test rows; "
            f"got {len(split.train_indices)} train and {len(split.test_indices)} test rows."
        )


def _time_keys(values: pd.Series, name: str) -> list[Any]:
    if pd.api.types.is_datetime64_any_dtype(values):
        parsed = pd.to_datetime(values, errors="coerce", utc=True)
        if not parsed.notna().all():
            raise ValueError(f"Timestamp column {name!r} must contain finite values.")
        return [pd.Timestamp(value) for value in parsed.tolist()]
    if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"Timestamp column {name!r} must contain finite values.")
        return numeric.tolist()
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"Timestamp column {name!r} must contain finite values.")
        return numeric.tolist()
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if parsed.notna().all():
        return [pd.Timestamp(value) for value in parsed.tolist()]
    raise ValueError(
        f"Timestamp column {name!r} must be numeric or parseable as datetimes."
    )


def _display_time(value: Any) -> str | float | int:
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return str(value)
