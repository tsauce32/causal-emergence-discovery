"""Candidate macro-variable construction with fitted, reusable encoders."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from causal_emergence_discovery.schema import FeatureSchema, declared_feature_types, numeric_values, resolve_feature_schema


class MacroRefitTopologyError(ValueError):
    """A merged macro cannot be replayed because its refit topology changed.

    ``diagnostics`` is a JSON-serializable explanation of the requested and
    observed recipe topology. The convenience attributes mirror its common
    fields so callers can branch on the failure without parsing the message.
    """

    def __init__(self, message: str, diagnostics: dict[str, object]):
        super().__init__(message)
        self.diagnostics = dict(diagnostics)
        self.recipe = diagnostics.get("recipe")
        self.expected_states = diagnostics.get("expected_states")
        self.observed_states = diagnostics.get("observed_states")
        self.expected_cutpoints = diagnostics.get("expected_cutpoints")
        self.observed_cutpoints = diagnostics.get("observed_cutpoints")


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


def generate_candidate_macros(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    max_states: int = 6,
    column_specs: dict[str, object] | None = None,
    schema: FeatureSchema | None = None,
) -> list[MacroAssignment]:
    """Generate candidate macros, fitting every encoder on ``df`` only."""
    available = [c for c in feature_columns if c in df.columns]
    resolved = schema or resolve_feature_schema(df, available, column_specs)
    numeric = [c for c in available if resolved.kind(c) == "numeric"]
    macros: list[MacroAssignment] = []
    for column in numeric:
        for bins in (2, 3):
            if bins <= max_states:
                candidate = quantile_macro(df, column, bins, column_specs=column_specs, schema=resolved)
                if candidate.state_count > 1:
                    macros.append(candidate)
    if len(numeric) >= 2:
        for bins in (2, 3):
            if bins <= max_states:
                candidate = composite_quantile_macro(df, numeric, bins, column_specs=column_specs, schema=resolved)
                if candidate.state_count > 1:
                    macros.append(candidate)
    for states in range(2, max_states + 1):
        if numeric:
            candidate = kmeans_macro(df, numeric, states, column_specs=column_specs, schema=resolved)
            if candidate.state_count > 1:
                macros.append(candidate)
    return _dedupe_macros(macros)


def quantile_macro(df: pd.DataFrame, column: str, bins: int, *, column_specs: dict[str, object] | None = None, schema: FeatureSchema | None = None) -> MacroAssignment:
    _validate_bins(bins)
    resolved = _numeric_schema(df, [column], column_specs, schema)
    values = _numeric_column(df, column)
    if not values.notna().any():
        raise ValueError(f"Feature {column!r} has no finite numeric values.")
    median = _median(values)
    filled = values.fillna(median)
    cuts = _quantile_cuts(filled.to_numpy(), bins)
    encoder = {"version": 1, "algorithm_version": 2, "recipe": "quantile", "identity": "bin_rank", "columns": [column], "bins": int(bins), "medians": {column: median}, "cutpoints": cuts, "state_map": {str(i): i for i in range(len(cuts) + 1)}}
    encoder.update(_schema_audit(resolved, column_specs, [column]))
    return _assignment_from_encoder(df, f"q{bins}:{column}", "quantile", (column,), encoder, {"bins": bins})


def composite_quantile_macro(df: pd.DataFrame, columns: list[str], bins: int, *, column_specs: dict[str, object] | None = None, schema: FeatureSchema | None = None) -> MacroAssignment:
    _validate_bins(bins)
    resolved = _numeric_schema(df, columns, column_specs, schema)
    stats = _fit_standardization(df, columns)
    matrix = _transform_matrix(df, columns, stats)
    composite = matrix.mean(axis=1)
    cuts = _quantile_cuts(composite, bins)
    encoder = {"version": 1, "algorithm_version": 2, "recipe": "composite_quantile", "identity": "bin_rank", "columns": list(columns), "bins": int(bins), "preprocess": stats, "cutpoints": cuts, "state_map": {str(i): i for i in range(len(cuts) + 1)}}
    encoder.update(_schema_audit(resolved, column_specs, columns))
    return _assignment_from_encoder(df, f"q{bins}:composite", "composite_quantile", tuple(columns), encoder, {"bins": bins})


def kmeans_macro(df: pd.DataFrame, columns: list[str], states: int, *, max_iter: int = 50, column_specs: dict[str, object] | None = None, schema: FeatureSchema | None = None) -> MacroAssignment:
    if states < 2:
        raise ValueError("states must be at least 2.")
    resolved = _numeric_schema(df, columns, column_specs, schema)
    stats = _fit_standardization(df, columns)
    matrix = _transform_matrix(df, columns, stats)
    if len(matrix) < 2:
        raise ValueError("At least two rows are required to fit k-means.")
    unique_rows = np.unique(matrix, axis=0)
    state_count = min(int(states), len(unique_rows), len(matrix))
    if state_count < 2:
        centers = matrix[:1].copy()
    else:
        # Sorting full rows before summing makes Lloyd updates independent of
        # input row order, including rows tied on the first feature.
        fit_matrix = matrix[_lexicographic_order(matrix)]
        centers = _initial_centers(fit_matrix, state_count)
        labels = np.full(len(matrix), -1, dtype=int)
        for _ in range(max_iter):
            next_labels = _nearest_centers(fit_matrix, centers)
            if np.array_equal(next_labels, labels):
                break
            labels = next_labels
            updated = centers.copy()
            for state in range(len(centers)):
                members = fit_matrix[labels == state]
                if len(members):
                    updated[state] = members.mean(axis=0)
            centers = updated
        labels = _nearest_centers(fit_matrix, centers)
        used = sorted(np.unique(labels).tolist())
        centers = centers[used]
        centers = centers[_lexicographic_order(centers)]
        state_count = len(centers)
    encoder = {"version": 1, "algorithm_version": 2, "algorithm": "canonical_lexicographic_lloyd", "recipe": "kmeans", "identity": "canonical_lexicographic_centroid_rank", "columns": list(columns), "requested_states": int(states), "max_iter": int(max_iter), "states": int(state_count), "preprocess": stats, "centers": centers.tolist(), "state_map": {str(i): i for i in range(state_count)}}
    encoder.update(_schema_audit(resolved, column_specs, columns))
    return _assignment_from_encoder(df, f"kmeans:{state_count}", "kmeans", tuple(columns), encoder, {"requested_states": states, "states": state_count, "algorithm_version": 2, "state_identity": "canonical_lexicographic_centroid_rank"})


def merge_macro_states(macro: MacroAssignment, left: int, right: int) -> MacroAssignment:
    """Merge two fitted states while preserving their mapping for future rows."""
    left, right = int(left), int(right)
    if left == right or left not in set(macro.labels.unique()) or right not in set(macro.labels.unique()):
        raise ValueError("left and right must be distinct states present in the macro.")
    labels = macro.labels.where(macro.labels != right, left).astype(int)
    encoder = _copy_encoder(macro.encoder)
    if encoder is not None:
        state_map = encoder.setdefault("state_map", {})
        mapped_states = {int(mapped) for mapped in state_map.values()}
        if left not in mapped_states or right not in mapped_states:
            raise ValueError("left and right must both be current mapped states in the fitted encoder.")
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


def refit_macro(macro: MacroAssignment, train_df: pd.DataFrame, *, column_specs: dict[str, object] | None = None) -> MacroAssignment:
    """Refit the same candidate recipe and state merges on training rows."""
    if macro.encoder is None:
        raise ValueError("MacroAssignment has no fitted encoder to refit.")
    recipe = macro.encoder.get("recipe")
    columns = list(macro.feature_columns)
    params = macro.encoder
    stored_types = dict(params.get("declared_types", {}))
    stored_types.update(declared_feature_types(column_specs, columns))
    refit_specs = {name: {"type": kind} for name, kind in stored_types.items()}
    if recipe == "quantile":
        fresh = quantile_macro(train_df, columns[0], int(params["bins"]), column_specs=refit_specs)
    elif recipe == "composite_quantile":
        fresh = composite_quantile_macro(train_df, columns, int(params["bins"]), column_specs=refit_specs)
    elif recipe == "kmeans":
        fresh = kmeans_macro(train_df, columns, int(params["requested_states"]), max_iter=int(params.get("max_iter", 50)), column_specs=refit_specs)
    else:
        raise ValueError(f"Unsupported fitted macro recipe: {recipe!r}.")
    encoder = _copy_encoder(fresh.encoder)
    assert encoder is not None
    merges = list(params.get("merges", []))
    requested_states, observed_states, expected_cutpoints, observed_cutpoints = _recipe_topology(encoder, fresh.labels)
    source_expected, source_observed, source_expected_cuts, source_observed_cuts = _recipe_topology(params)
    if merges:
        _require_replay_topology(params, source_expected, source_observed, source_expected_cuts, source_observed_cuts, "source")
        _require_replay_topology(encoder, requested_states, observed_states, expected_cutpoints, observed_cutpoints, "refit")
        _validate_source_merge_history(params, merges, source_expected)
    state_map = encoder.setdefault("state_map", {})
    replayed_merges = []
    for merge in merges:
        left, right = int(merge["left"]), int(merge["right"])
        current_groups = {int(mapped) for mapped in state_map.values()}
        if left == right or left not in current_groups or right not in current_groups:
            diagnostics = {
                "recipe": recipe,
                "reason": "invalid_chained_merge_operands",
                "merge_index": len(replayed_merges),
                "left": left,
                "right": right,
                "current_groups": sorted(current_groups),
                "expected_states": requested_states,
                "observed_states": observed_states,
                "expected_cutpoints": expected_cutpoints,
                "observed_cutpoints": observed_cutpoints,
            }
            raise MacroRefitTopologyError("Merge history refers to absent or already-collapsed states.", diagnostics)
        for raw, mapped in list(state_map.items()):
            if int(mapped) == right:
                state_map[raw] = left
        replayed_merges.append({"left": left, "right": right})
    if replayed_merges:
        encoder["merges"] = replayed_merges
    metadata = dict(fresh.metadata)
    metadata["refit_topology"] = {
        "recipe": recipe,
        "requested_states": requested_states,
        "observed_states": observed_states,
        "expected_cutpoints": expected_cutpoints,
        "observed_cutpoints": observed_cutpoints,
        "merged": bool(merges),
        "identity": encoder.get("identity"),
        "algorithm_version": encoder.get("algorithm_version", 1),
    }
    metadata["encoder"] = _copy_encoder(encoder)
    return replace(fresh, name=macro.name, method=macro.method, labels=_apply_encoder(train_df, encoder), encoder=encoder, metadata=metadata)


def apply_macro(macro: MacroAssignment, validation_df: pd.DataFrame) -> MacroAssignment:
    """Apply training-fitted parameters without learning from validation rows."""
    if macro.encoder is None:
        raise ValueError("MacroAssignment has no fitted encoder to apply.")
    labels, diagnostics = _apply_encoder_with_report(validation_df, macro.encoder)
    metadata = dict(macro.metadata)
    metadata["encoder"] = _copy_encoder(macro.encoder)
    metadata["application_diagnostics"] = diagnostics
    return replace(macro, labels=labels, metadata=metadata)


def _assignment_from_encoder(df: pd.DataFrame, name: str, method: str, columns: tuple[str, ...], encoder: dict[str, Any], metadata: dict[str, object]) -> MacroAssignment:
    labels, diagnostics = _apply_encoder_with_report(df, encoder)
    # Preserve pre-merge occupancy so later refits can distinguish a complete
    # source recipe from an already-reduced merged label set.
    encoder["fit_observed_states"] = int(labels.nunique(dropna=True))
    encoder["fit_occupied_raw_states"] = sorted(int(value) for value in labels.dropna().unique())
    audit = dict(metadata)
    audit["encoder"] = _copy_encoder(encoder)
    audit["fit_transform_diagnostics"] = diagnostics
    return MacroAssignment(name, labels, method, columns, audit, encoder)


def _apply_encoder(df: pd.DataFrame, encoder: dict[str, Any]) -> pd.Series:
    return _apply_encoder_with_report(df, encoder)[0]


def _apply_encoder_with_report(df: pd.DataFrame, encoder: dict[str, Any]) -> tuple[pd.Series, dict[str, dict[str, int]]]:
    columns = list(encoder["columns"])
    absent = [c for c in columns if c not in df.columns]
    if absent:
        raise ValueError(f"Cannot apply macro; missing feature columns: {absent}.")
    diagnostics = {}
    for column in columns:
        values, invalid = numeric_values(df[column])
        diagnostics[column] = {"missing": int(df[column].isna().sum()), "invalid_numeric": invalid, "unknown_category": 0}
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
    return pd.Series(mapped, index=df.index, dtype=int), diagnostics


def _fit_standardization(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    if not columns:
        raise ValueError("At least one numeric column is required.")
    stats: dict[str, dict[str, float]] = {}
    for column in columns:
        values = _numeric_column(df, column)
        if not values.notna().any():
            raise ValueError(f"Feature {column!r} has no finite numeric values.")
        median = _median(values)
        # Sum in a canonical value order so a row permutation cannot alter
        # floating-point reductions through a different accumulation order.
        ordered = np.sort(values.fillna(median).to_numpy(dtype=float))
        mean = float(np.mean(ordered))
        scale = float(np.sqrt(np.mean((ordered - mean) ** 2)))
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


def _numeric_schema(df: pd.DataFrame, columns: list[str], column_specs: dict[str, object] | None, schema: FeatureSchema | None) -> FeatureSchema:
    resolved = schema or resolve_feature_schema(df, columns, column_specs)
    nonnumeric = [column for column in columns if resolved.kind(column) != "numeric"]
    if nonnumeric:
        raise ValueError(f"Macro features must have numeric variable types: {nonnumeric}.")
    return resolved


def _schema_audit(schema: FeatureSchema, column_specs: dict[str, object] | None, columns: list[str]) -> dict[str, object]:
    return {
        "feature_schema": {column: schema.kind(column) for column in columns},
        "declared_types": declared_feature_types(column_specs, columns),
    }


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
    unique_rows = np.unique(matrix, axis=0)
    positions = np.linspace(0, len(unique_rows) - 1, states).round().astype(int)
    return unique_rows[positions].copy()


def _lexicographic_order(matrix: np.ndarray) -> np.ndarray:
    """Return row indices sorted by every column, with column zero primary."""
    if matrix.shape[1] <= 1:
        return np.argsort(matrix[:, 0], kind="mergesort")
    return np.lexsort(tuple(matrix[:, column] for column in range(matrix.shape[1] - 1, -1, -1)))


def _recipe_topology(encoder: dict[str, Any], labels: pd.Series | None = None) -> tuple[int, int, int | None, int | None]:
    recipe = encoder.get("recipe")
    if recipe in {"quantile", "composite_quantile"}:
        requested = int(encoder["bins"])
        expected_cuts = requested - 1
        observed_cuts = len(encoder.get("cutpoints", []))
        observed = int(labels.nunique(dropna=True)) if labels is not None else int(encoder.get("fit_observed_states", encoder.get("states", observed_cuts + 1)))
        return requested, observed, expected_cuts, observed_cuts
    if recipe == "kmeans":
        requested = int(encoder["requested_states"])
        observed = int(labels.nunique(dropna=True)) if labels is not None else int(encoder.get("fit_observed_states", encoder.get("states", len(encoder.get("centers", [])))))
        return requested, observed, None, None
    raise ValueError(f"Unsupported fitted macro recipe: {recipe!r}.")


def _require_replay_topology(
    encoder: dict[str, Any],
    expected_states: int,
    observed_states: int,
    expected_cutpoints: int | None,
    observed_cutpoints: int | None,
    phase: str,
) -> None:
    recipe = str(encoder.get("recipe"))
    if recipe == "kmeans" and (
        int(encoder.get("algorithm_version", 1)) < 2
        or encoder.get("identity") != "canonical_lexicographic_centroid_rank"
    ):
        diagnostics = {
            "recipe": recipe,
            "reason": "unsupported_legacy_identity",
            "phase": phase,
            "algorithm_version": encoder.get("algorithm_version", 1),
            "identity": encoder.get("identity"),
            "expected_states": expected_states,
            "observed_states": observed_states,
            "expected_cutpoints": expected_cutpoints,
            "observed_cutpoints": observed_cutpoints,
        }
        raise MacroRefitTopologyError("Merged legacy k-means recipe has no stable centroid-rank identity.", diagnostics)
    if phase == "source" and recipe in {"quantile", "composite_quantile"}:
        occupied = encoder.get("fit_occupied_raw_states")
        expected_occupied = list(range(expected_states))
        if occupied is None or [int(value) for value in occupied] != expected_occupied:
            diagnostics = {
                "recipe": recipe,
                "reason": "unsupported_source_topology",
                "phase": phase,
                "fit_occupied_raw_states": occupied,
                "expected_states": expected_states,
                "observed_states": observed_states,
                "expected_cutpoints": expected_cutpoints,
                "observed_cutpoints": observed_cutpoints,
            }
            raise MacroRefitTopologyError("Merged legacy quantile recipe has no complete source occupancy record.", diagnostics)
    cuts_complete = expected_cutpoints is None or observed_cutpoints == expected_cutpoints
    if observed_states != expected_states or not cuts_complete:
        diagnostics = {
            "recipe": recipe,
            "reason": "incomplete_refit_topology",
            "phase": phase,
            "expected_states": expected_states,
            "observed_states": observed_states,
            "expected_cutpoints": expected_cutpoints,
            "observed_cutpoints": observed_cutpoints,
        }
        raise MacroRefitTopologyError("Merged macro recipe does not retain its requested distinct occupied topology.", diagnostics)


def _validate_source_merge_history(encoder: dict[str, Any], merges: list[dict[str, Any]], expected_states: int) -> None:
    """Verify stored merge operations produce the encoder's current state map."""
    actual_map = {str(raw): int(mapped) for raw, mapped in encoder.get("state_map", {}).items()}
    raw_ranks = {str(rank) for rank in range(expected_states)}
    if set(actual_map) != raw_ranks:
        diagnostics = {
            "recipe": encoder.get("recipe"),
            "reason": "corrupted_source_merge_map",
            "phase": "source",
            "expected_raw_ranks": sorted(raw_ranks),
            "observed_raw_ranks": sorted(actual_map),
        }
        raise MacroRefitTopologyError("Source encoder state map does not contain the complete recipe rank domain.", diagnostics)
    replay_map = {raw: int(raw) for raw in raw_ranks}
    for index, merge in enumerate(merges):
        left, right = int(merge["left"]), int(merge["right"])
        groups = set(replay_map.values())
        if left == right or left not in groups or right not in groups:
            diagnostics = {
                "recipe": encoder.get("recipe"),
                "reason": "invalid_chained_merge_operands",
                "phase": "source",
                "merge_index": index,
                "left": left,
                "right": right,
                "current_groups": sorted(groups),
            }
            raise MacroRefitTopologyError("Source merge history refers to absent or already-collapsed states.", diagnostics)
        for raw, mapped in list(replay_map.items()):
            if mapped == right:
                replay_map[raw] = left
    if replay_map != actual_map:
        diagnostics = {
            "recipe": encoder.get("recipe"),
            "reason": "corrupted_source_merge_map",
            "phase": "source",
            "expected_state_map": replay_map,
            "observed_state_map": actual_map,
        }
        raise MacroRefitTopologyError("Source encoder state map disagrees with its stored merge history.", diagnostics)


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
