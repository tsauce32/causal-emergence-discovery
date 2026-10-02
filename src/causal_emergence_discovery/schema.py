"""Training-fitted feature types shared by macro and regression encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSchema:
    """Frozen variable kinds resolved from declarations or training values."""

    kinds: tuple[tuple[str, str], ...]

    def kind(self, column: str) -> str:
        try:
            return dict(self.kinds)[column]
        except KeyError as exc:
            raise ValueError(f"Feature {column!r} is not present in the fitted schema.") from exc

    def to_dict(self) -> dict[str, str]:
        return dict(self.kinds)


@dataclass(frozen=True)
class TransformReport:
    """Counts describing how a frozen encoder handled supplied rows."""

    columns: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {name: dict(counts) for name, counts in self.columns.items()}


def resolve_feature_schema(
    train_df: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
    column_specs: Mapping[str, Any] | None = None,
) -> FeatureSchema:
    """Resolve declared kinds first and otherwise inspect only training values.

    Object dtype is deliberately not used as evidence: a string in an outer
    holdout can change the dtype of the source column before rows are split.
    """
    kinds: list[tuple[str, str]] = []
    for column in dict.fromkeys(columns):
        if column not in train_df:
            raise ValueError(f"Feature column {column!r} is missing from training data.")
        raw_type = _declared_type((column_specs or {}).get(column))
        kind = _normalize_kind(raw_type) if raw_type is not None else _infer_kind(train_df[column])
        kinds.append((column, kind))
    return FeatureSchema(tuple(kinds))


def declared_feature_types(column_specs: Mapping[str, Any] | None, columns: list[str] | tuple[str, ...]) -> dict[str, str]:
    """Return normalized explicit declarations for recipe refits."""
    declared = {}
    for column in dict.fromkeys(columns):
        raw_type = _declared_type((column_specs or {}).get(column))
        if raw_type is not None:
            declared[column] = _normalize_kind(raw_type)
    return declared


def numeric_values(series: pd.Series) -> tuple[pd.Series, int]:
    """Convert values to finite numbers and count malformed non-missing input."""
    converted = pd.to_numeric(series, errors="coerce").astype(float)
    finite = converted.where(np.isfinite(converted), np.nan)
    invalid = int((series.notna() & finite.isna()).sum())
    return finite, invalid


def _declared_type(spec: Any) -> str | None:
    if spec is None:
        return None
    if isinstance(spec, Mapping):
        value = spec.get("variable_type", spec.get("type"))
    else:
        value = getattr(spec, "variable_type", None)
    if value is None or not str(value).strip():
        return None
    return str(value).strip().lower()


def _normalize_kind(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"numeric", "number", "continuous", "integer", "int", "float", "real", "ordinal_numeric"}:
        return "numeric"
    if normalized in {"categorical", "category", "string", "text", "nominal", "boolean", "bool", "binary"}:
        return "categorical"
    raise ValueError(f"Unsupported declared variable type {value!r}; use numeric or categorical.")


def _infer_kind(values: pd.Series) -> str:
    observed = values.loc[values.notna()]
    if observed.empty or all(isinstance(value, (bool, np.bool_)) for value in observed.tolist()):
        return "categorical"
    converted = pd.to_numeric(observed, errors="coerce")
    return "numeric" if bool(converted.notna().all()) else "categorical"
