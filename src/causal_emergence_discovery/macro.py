"""Candidate macro-variable construction with fitted, reusable encoders."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MacroAssignment:
    """A discrete macro assignment and its training-fitted transform."""

    name: str
    labels: pd.Series
    method: str
    feature_columns: tuple[str, ...]
    metadata: dict[str, object]
    encoder: dict[str, Any] | None = None

    @property
    def state_count(self) -> int:
        return int(self.labels.nunique(dropna=True))

    def with_labels(self, labels: pd.Series, *, name: str | None = None, method: str | None = None) -> "MacroAssignment":
        metadata = dict(self.metadata)
        metadata.pop("encoder", None)
        return replace(self, name=name or self.name, labels=_canonical_labels(labels), method=method or self.method, metadata=metadata, encoder=None)


def generate_candidate_macros(df: pd.DataFrame, feature_columns: list[str], *, max_states: int = 6) -> list[MacroAssignment]:
    """Generate candidate macros, fitting every encoder on ``df`` only."""
    numeric = [c for c in feature_columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
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
        if numeric:
            candidate = kmeans_macro(df, numeric, states)
            if candidate.state_count > 1:
                macros.append(candidate)
    return _dedupe_macros(macros)


def quantile_macro(df: pd.DataFrame, column: str, bins: int) -> MacroAssignment:
    _validate_bins(bins)
    values = _numeric_column(df, column)
    if not values.notna().any():
        raise ValueError(f"Feature {column!r} has no finite numeric values.")
    median = _median(values)
    filled = values.fillna(median)
    cuts = _quantile_cuts(filled.to_numpy(), bins)
    encoder = {"version": 1, "recipe": "quantile", "columns": [column], "bins": int(bins), "medians": {column: median}, "cutpoints": cuts, "state_map": {str(i): i for i in range(len(cuts) + 1)}}
    return _assignment_from_encoder(df, f"q{bins}:{column}", "quantile", (column,), encoder, {"bins": bins})


def composite_quantile_macro(df: pd.DataFrame, columns: list[str], bins: int) -> MacroAssignment:
    _validate_bins(bins)
    stats = _fit_standardization(df, columns)
    matrix = _transform_matrix(df, columns, stats)
    composite = matrix.mean(axis=1)
    cuts = _quantile_cuts(composite, bins)
    encoder = {"version": 1, "recipe": "composite_quantile", "columns": list(columns), "bins": int(bins), "preprocess": stats, "cutpoints": cuts, "state_map": {str(i): i for i in range(len(cuts) + 1)}}
    return _assignment_from_encoder(df, f"q{bins}:composite", "composite_quantile", tuple(columns), encoder, {"bins": bins})


def kmeans_macro(df: pd.DataFrame, columns: list[str], states: int, *, max_iter: int = 50) -> MacroAssignment:
    if states < 2:
        raise ValueError("states must be at least 2.")
    stats = _fit_standardization(df, columns)
    matrix = _transform_matrix(df, columns, stats)
    if len(matrix) < 2:
        raise ValueError("At least two rows are required to fit k-means.")
    unique_rows = np.unique(matrix, axis=0)
    state_count = min(int(states), len(unique_rows), len(matrix))
    if state_count < 2:
        centers = matrix[:1].copy()
    else:
        centers = _initial_centers(matrix, state_count)
        labels = np.full(len(matrix), -1, dtype=int)
        for _ in range(max_iter):
            next_labels = _nearest_centers(matrix, centers)
            if np.array_equal(next_labels, labels):
                break
            labels = next_labels
            updated = centers.copy()
            for state in range(len(centers)):
                members = matrix[labels == state]
                if len(members):
                    updated[state] = members.mean(axis=0)
            centers = updated
        labels = _nearest_centers(matrix, centers)
        used = sorted(np.unique(labels).tolist())
        centers = centers[used]
        state_count = len(centers)
    encoder = {"version": 1, "recipe": "kmeans", "columns": list(columns), "requested_states": int(states), "states": int(state_count), "preprocess": stats, "centers": centers.tolist(), "state_map": {str(i): i for i in range(state_count)}}
    return _assignment_from_encoder(df, f"kmeans:{state_count}", "kmeans", tuple(columns), encoder, {"requested_states": states, "states": state_count})


def merge_macro_states(macro: MacroAssignment, left: int, right: int) -> MacroAssignment:
    """Merge two fitted states while preserving their mapping for future rows."""
    left, right = int(left), int(right)
    if left == right or left not in set(macro.labels.unique()) or right not in set(macro.labels.unique()):
        raise ValueError("left and right must be distinct states present in the macro.")
    labels = macro.labels.where(macro.labels != right, left).astype(int)
    encoder = _copy_encoder(macro.encoder)
    if encoder is not None:
        state_map = encoder.setdefault("state_map", {})
        for raw, mapped in list(state_map.items()):
            if int(mapped) == right:
                state_map[raw] = left
        encoder.setdefault("merges", []).append({"left": left, "right": right})
    metadata = dict(macro.metadata)
    if encoder is not None:
        metadata["encoder"] = encoder
    return replace(macro, name=f"{macro.name}|merge:{left}+{right}", labels=labels, method=f"{macro.method}_merged", metadata=metadata, encoder=encoder)


def possible_state_merges(macro: MacroAssignment) -> list[MacroAssignment]:
    states = sorted(int(state) for state in macro.labels.dropna().unique())
    return [merge_macro_states(macro, left, right) for i, left in enumerate(states) for right in states[i + 1:]]


def refit_macro(macro: MacroAssignment, train_df: pd.DataFrame) -> MacroAssignment:
    """Refit the same candidate recipe and state merges on training rows."""
    if macro.encoder is None:
        raise ValueError("MacroAssignment has no fitted encoder to refit.")
    recipe = macro.encoder.get("recipe")
    columns = list(macro.feature_columns)
    params = macro.encoder
    if recipe == "quantile":
        fresh = quantile_macro(train_df, columns[0], int(params["bins"]))
    elif recipe == "composite_quantile":
        fresh = composite_quantile_macro(train_df, columns, int(params["bins"]))
    elif recipe == "kmeans":
        fresh = kmeans_macro(train_df, columns, int(params["requested_states"]))
    else:
        raise ValueError(f"Unsupported fitted macro recipe: {recipe!r}.")
    encoder = _copy_encoder(fresh.encoder)
    assert encoder is not None
    degeneracies = []
    state_map = encoder.setdefault("state_map", {})
    replayed_merges = []
    for merge in params.get("merges", []):
        left, right = int(merge["left"]), int(merge["right"])
        present = set(int(label) for label in fresh.labels.unique())
        if right not in present:
            degeneracies.append({"left": left, "right": right, "reason": "right_state_absent_in_refit_training"})
        for raw, mapped in list(state_map.items()):
            if int(mapped) == right:
                state_map[raw] = left
        replayed_merges.append({"left": left, "right": right})
    if replayed_merges:
        encoder["merges"] = replayed_merges
    metadata = dict(fresh.metadata)
    metadata["encoder"] = _copy_encoder(encoder)
    if degeneracies:
        metadata["refit_merge_degeneracies"] = degeneracies
    return replace(fresh, name=macro.name, method=macro.method, labels=_apply_encoder(train_df, encoder), encoder=encoder, metadata=metadata)


def apply_macro(macro: MacroAssignment, validation_df: pd.DataFrame) -> MacroAssignment:
    """Apply training-fitted parameters without learning from validation rows."""
    if macro.encoder is None:
        raise ValueError("MacroAssignment has no fitted encoder to apply.")
    labels = _apply_encoder(validation_df, macro.encoder)
    metadata = dict(macro.metadata)
    metadata["encoder"] = _copy_encoder(macro.encoder)
    return replace(macro, labels=labels, metadata=metadata)


def _assignment_from_encoder(df: pd.DataFrame, name: str, method: str, columns: tuple[str, ...], encoder: dict[str, Any], metadata: dict[str, object]) -> MacroAssignment:
    labels = _apply_encoder(df, encoder)
    audit = dict(metadata)
    audit["encoder"] = _copy_encoder(encoder)
    return MacroAssignment(name, labels, method, columns, audit, encoder)


def _apply_encoder(df: pd.DataFrame, encoder: dict[str, Any]) -> pd.Series:
    columns = list(encoder["columns"])
    absent = [c for c in columns if c not in df.columns]
    if absent:
        raise ValueError(f"Cannot apply macro; missing feature columns: {absent}.")
    recipe = encoder["recipe"]
    if recipe == "quantile":
        column = columns[0]
        values = _numeric_column(df, column).fillna(float(encoder["medians"][column])).to_numpy()
        raw = np.searchsorted(np.asarray(encoder["cutpoints"], dtype=float), values, side="left")
    elif recipe == "composite_quantile":
        matrix = _transform_matrix(df, columns, encoder["preprocess"])
        raw = np.searchsorted(np.asarray(encoder["cutpoints"], dtype=float), matrix.mean(axis=1), side="left")
    elif recipe == "kmeans":
        matrix = _transform_matrix(df, columns, encoder["preprocess"])
        raw = _nearest_centers(matrix, np.asarray(encoder["centers"], dtype=float))
    else:
        raise ValueError(f"Unsupported fitted macro recipe: {recipe!r}.")
    mapping = {int(k): int(v) for k, v in encoder.get("state_map", {}).items()}
    mapped = np.asarray([mapping.get(int(state), int(state)) for state in raw], dtype=int)
    return pd.Series(mapped, index=df.index, dtype=int)


def _fit_standardization(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    if not columns:
        raise ValueError("At least one numeric column is required.")
    stats: dict[str, dict[str, float]] = {}
    for column in columns:
        values = _numeric_column(df, column)
        if not values.notna().any():
            raise ValueError(f"Feature {column!r} has no finite numeric values.")
        median = _median(values)
        filled = values.fillna(median)
        mean = float(filled.mean())
        scale = float(filled.std(ddof=0))
        stats[column] = {"median": median, "mean": mean, "scale": scale if scale > 0 else 1.0}
    return stats


def _transform_matrix(df: pd.DataFrame, columns: list[str], stats: dict[str, dict[str, float]]) -> np.ndarray:
    transformed = []
    for column in columns:
        values = _numeric_column(df, column).fillna(float(stats[column]["median"]))
        scale = float(stats[column]["scale"])
        transformed.append(((values.to_numpy(dtype=float) - float(stats[column]["mean"])) / scale).reshape(-1, 1))
    return np.hstack(transformed).astype(float)


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        raise ValueError(f"Feature column {column!r} is missing.")
    values = pd.to_numeric(df[column], errors="coerce").astype(float)
    return values.where(np.isfinite(values), np.nan)


def _median(values: pd.Series) -> float:
    return float(values.median()) if values.notna().any() else 0.0


def _quantile_cuts(values: np.ndarray, bins: int) -> list[float]:
    if bins < 2:
        raise ValueError("bins must be at least 2.")
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("At least one finite value is required to fit quantile bins.")
    cuts = np.quantile(finite, np.arange(1, bins) / bins, method="linear")
    return sorted(set(float(x) for x in cuts if np.isfinite(x)))


def _initial_centers(matrix: np.ndarray, states: int) -> np.ndarray:
    order = np.argsort(matrix[:, 0], kind="mergesort")
    positions = np.linspace(0, len(order) - 1, states).round().astype(int)
    return matrix[order[positions]].copy()


def _nearest_centers(matrix: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return ((matrix[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)


def _validate_bins(bins: int) -> None:
    if bins < 2:
        raise ValueError("bins must be at least 2.")


def _copy_encoder(encoder: dict[str, Any] | None) -> dict[str, Any] | None:
    if encoder is None:
        return None
    # JSON round-tripping ensures the audit representation uses plain serializable types.
    import json
    return json.loads(json.dumps(encoder))


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
