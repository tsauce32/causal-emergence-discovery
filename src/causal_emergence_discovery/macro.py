"""Candidate macro-variable construction."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MacroAssignment:
    """A discrete macro variable assigned to each modeling row."""

    name: str
    labels: pd.Series
    method: str
    feature_columns: tuple[str, ...]
    metadata: dict[str, object]

    @property
    def state_count(self) -> int:
        return int(self.labels.nunique(dropna=True))

    def with_labels(self, labels: pd.Series, *, name: str | None = None, method: str | None = None) -> "MacroAssignment":
        return replace(
            self,
            name=name or self.name,
            labels=_canonical_labels(labels),
            method=method or self.method,
        )


def generate_candidate_macros(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    max_states: int = 6,
) -> list[MacroAssignment]:
    """Generate a compact set of candidate macro variables."""
    numeric = [
        column
        for column in feature_columns
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column])
    ]
    macros: list[MacroAssignment] = []
    for column in numeric:
        for bins in (2, 3):
            if bins <= max_states:
                candidate = quantile_macro(df, column, bins)
                if candidate.state_count > 1:
                    macros.append(candidate)

    if len(numeric) >= 2:
        for bins in (2, 3):
            if bins <= max_states:
                candidate = composite_quantile_macro(df, numeric, bins)
                if candidate.state_count > 1:
                    macros.append(candidate)

    for states in range(2, max_states + 1):
        if len(numeric) >= 1:
            candidate = kmeans_macro(df, numeric, states)
            if candidate.state_count > 1:
                macros.append(candidate)
    return _dedupe_macros(macros)


def quantile_macro(df: pd.DataFrame, column: str, bins: int) -> MacroAssignment:
    values = pd.to_numeric(df[column], errors="coerce")
    labels = _quantile_labels(values, bins)
    return MacroAssignment(
        name=f"q{bins}:{column}",
        labels=labels,
        method="quantile",
        feature_columns=(column,),
        metadata={"bins": bins},
    )


def composite_quantile_macro(df: pd.DataFrame, columns: list[str], bins: int) -> MacroAssignment:
    matrix = _standardized_numeric_matrix(df, columns)
    composite = pd.Series(np.nanmean(matrix, axis=1), index=df.index)
    labels = _quantile_labels(composite, bins)
    return MacroAssignment(
        name=f"q{bins}:composite",
        labels=labels,
        method="composite_quantile",
        feature_columns=tuple(columns),
        metadata={"bins": bins},
    )


def kmeans_macro(
    df: pd.DataFrame,
    columns: list[str],
    states: int,
    *,
    max_iter: int = 50,
) -> MacroAssignment:
    matrix = _standardized_numeric_matrix(df, columns)
    unique_rows = np.unique(matrix, axis=0)
    state_count = min(states, len(unique_rows), len(matrix))
    if state_count < 2:
        labels = pd.Series(np.zeros(len(df), dtype=int), index=df.index)
    else:
        centers = _initial_centers(matrix, state_count)
        labels_array = np.zeros(len(matrix), dtype=int)
        for _ in range(max_iter):
            distances = ((matrix[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            next_labels = distances.argmin(axis=1)
            if np.array_equal(next_labels, labels_array):
                break
            labels_array = next_labels
            for state in range(state_count):
                members = matrix[labels_array == state]
                if len(members):
                    centers[state] = members.mean(axis=0)
        labels = pd.Series(labels_array, index=df.index)
    return MacroAssignment(
        name=f"kmeans:{state_count}",
        labels=_canonical_labels(labels),
        method="kmeans",
        feature_columns=tuple(columns),
        metadata={"requested_states": states, "states": state_count},
    )


def merge_macro_states(macro: MacroAssignment, left: int, right: int) -> MacroAssignment:
    labels = macro.labels.copy()
    labels = labels.where(labels != right, left)
    merged_name = f"{macro.name}|merge:{left}+{right}"
    return macro.with_labels(labels, name=merged_name, method=f"{macro.method}_merged")


def possible_state_merges(macro: MacroAssignment) -> list[MacroAssignment]:
    states = sorted(int(state) for state in macro.labels.dropna().unique())
    return [
        merge_macro_states(macro, left, right)
        for left_index, left in enumerate(states)
        for right in states[left_index + 1 :]
    ]


def _standardized_numeric_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not columns:
        raise ValueError("At least one numeric column is required.")
    frame = pd.DataFrame(index=df.index)
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce").astype(float)
        fill = float(values.median()) if values.notna().any() else 0.0
        values = values.fillna(fill)
        std = float(values.std(ddof=0))
        frame[column] = 0.0 if std == 0.0 else (values - float(values.mean())) / std
    return frame.to_numpy(dtype=float)


def _initial_centers(matrix: np.ndarray, states: int) -> np.ndarray:
    order = np.argsort(matrix[:, 0], kind="mergesort")
    positions = np.linspace(0, len(order) - 1, states).round().astype(int)
    return matrix[order[positions]].copy()


def _quantile_labels(values: pd.Series, bins: int) -> pd.Series:
    filled = values.fillna(float(values.median()) if values.notna().any() else 0.0)
    ranks = filled.rank(method="first")
    try:
        labels = pd.qcut(ranks, q=bins, labels=False, duplicates="drop")
    except ValueError:
        labels = pd.Series(np.zeros(len(values), dtype=int), index=values.index)
    return _canonical_labels(pd.Series(labels, index=values.index))


def _canonical_labels(labels: pd.Series) -> pd.Series:
    mapping: dict[object, int] = {}
    canonical = []
    for value in labels.tolist():
        key = value if pd.notna(value) else "__missing__"
        if key not in mapping:
            mapping[key] = len(mapping)
        canonical.append(mapping[key])
    return pd.Series(canonical, index=labels.index, dtype=int)


def _dedupe_macros(macros: list[MacroAssignment]) -> list[MacroAssignment]:
    seen: set[tuple[int, ...]] = set()
    unique = []
    for macro in macros:
        signature = tuple(int(value) for value in macro.labels.tolist())
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(macro)
    return unique
