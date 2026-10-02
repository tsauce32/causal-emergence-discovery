"""Panel data validation and lagged table construction."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from causal_emergence_discovery.schema import declared_feature_types
from causal_emergence_discovery.spec import StudySpec


@dataclass(frozen=True)
class PanelSummary:
    row_count: int
    entity_count: int
    observed_columns: tuple[str, ...]
    duplicate_entity_times: int
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class LaggedDataset:
    data: pd.DataFrame
    outcome_columns: tuple[str, ...]
    target_columns: dict[str, str]
    lag: int
    id_column: str
    time_column: str
    target_time_column: str = "__lead_time"
    target_eligibility: dict[str, str] = field(default_factory=dict)


def load_panel_csv(path: str, spec: StudySpec | None = None) -> pd.DataFrame:
    """Load a panel CSV, preserving declared categorical tokens when supplied.

    Without a spec, retain pandas' legacy type inference. With a spec, declared
    categorical columns are read as strings so values such as ``"01"`` are
    not collapsed into the numeric token ``"1"`` before development-only
    schema fitting. Categorical aliases are normalized by the shared schema
    helper.
    """
    dtype: dict[str, str] = {}
    if spec is not None and spec.columns:
        columns = list(dict.fromkeys([
            *spec.feature_columns(list(spec.columns)),
            *spec.interventions,
            *(spec.adjustment.columns if spec.adjustment is not None else ()),
        ]))
        declared = declared_feature_types(spec.columns, columns)
        categorical = {name: "string" for name, kind in declared.items() if kind == "categorical"}
        if categorical:
            available = set(pd.read_csv(path, nrows=0).columns)
            dtype = {name: kind for name, kind in categorical.items() if name in available}
    return pd.read_csv(path, dtype=dtype or None)


def validate_panel(df: pd.DataFrame, spec: StudySpec) -> PanelSummary:
    """Validate the minimal longitudinal table contract."""
    missing = [name for name in (spec.id_column, spec.time_column) if name not in df.columns]
    missing.extend(name for name in spec.outcomes if name not in df.columns)
    missing.extend(name for name in spec.interventions if name not in df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(set(missing))}")

    warnings: list[str] = []
    if df[spec.id_column].isna().any():
        raise ValueError(f"id_column {spec.id_column!r} contains missing values.")
    if df[spec.time_column].isna().any():
        raise ValueError(f"time_column {spec.time_column!r} contains missing values.")

    duplicate_count = int(df.duplicated([spec.id_column, spec.time_column]).sum())
    if duplicate_count:
        warnings.append(
            f"{duplicate_count} rows share the same entity id and time; "
            "lag construction will keep their original row order."
        )

    entity_count = int(df[spec.id_column].nunique())
    if entity_count < 2:
        warnings.append("Only one entity was found; robustness checks will be limited.")
    return PanelSummary(
        row_count=len(df),
        entity_count=entity_count,
        observed_columns=tuple(df.columns),
        duplicate_entity_times=duplicate_count,
        warnings=tuple(warnings),
    )


def build_lagged_table(
    df: pd.DataFrame,
    spec: StudySpec,
    *,
    outcomes: list[str] | None = None,
    lag: int = 1,
) -> LaggedDataset:
    """Return rows at time t with requested outcomes shifted to t + lag."""
    if lag < 1:
        raise ValueError("lag must be at least 1.")
    validate_panel(df, spec)
    selected_outcomes = tuple(outcomes or spec.outcome_columns(list(df.columns)))
    if not selected_outcomes:
        raise ValueError("At least one outcome column is required.")
    missing_outcomes = [name for name in selected_outcomes if name not in df.columns]
    if missing_outcomes:
        raise ValueError(f"Outcome columns not found: {missing_outcomes}")
    generated_columns = {
        "__lead_time",
        "__ced_sort_time",
        *(f"{name}__lead{lag}" for name in selected_outcomes),
        *(f"__target_eligible__{name}" for name in selected_outcomes),
    }
    collisions = sorted(name for name in df.columns if name in generated_columns or name.startswith("__ced_"))
    if collisions:
        raise ValueError(f"Input columns collide with generated modeling columns: {collisions}")

    ordered = _sort_panel(df, spec)
    result = ordered.copy()
    target_columns: dict[str, str] = {}
    grouped = result.groupby(spec.id_column, sort=False)
    for outcome in selected_outcomes:
        target = f"{outcome}__lead{lag}"
        target_columns[outcome] = target
        result[target] = grouped[outcome].shift(-lag)
    target_eligibility: dict[str, str] = {}
    for outcome, target in target_columns.items():
        eligible = f"__target_eligible__{outcome}"
        target_eligibility[outcome] = eligible
        numeric_target = pd.to_numeric(result[target], errors="coerce").replace(
            [float("inf"), float("-inf")], pd.NA
        )
        result[eligible] = numeric_target.notna()
    # Nullable integers preserve exact timestamp precision across the trailing
    # missing lead, instead of promoting large integer timestamps to float.
    timestamps = result[spec.time_column]
    if pd.api.types.is_integer_dtype(timestamps):
        timestamps = timestamps.astype("UInt64" if pd.api.types.is_unsigned_integer_dtype(timestamps) else "Int64")
    result["__lead_time"] = timestamps.groupby(result[spec.id_column], sort=False).shift(-lag)
    # Lock the candidate cohort on panel geometry alone. A missing or nonfinite
    # outcome remains a row with an ineligible target, so a later split cannot
    # replace it with a different entity or time block.
    result = result.loc[result["__lead_time"].notna()].reset_index(drop=True)
    return LaggedDataset(
        data=result,
        outcome_columns=selected_outcomes,
        target_columns=target_columns,
        target_eligibility=target_eligibility,
        lag=lag,
        id_column=spec.id_column,
        time_column=spec.time_column,
        target_time_column="__lead_time",
    )


def _sort_panel(df: pd.DataFrame, spec: StudySpec) -> pd.DataFrame:
    raw_time = df[spec.time_column]
    if pd.api.types.is_datetime64_any_dtype(raw_time):
        time_values = pd.to_datetime(raw_time, errors="coerce", utc=True)
    else:
        numeric = pd.to_numeric(raw_time, errors="coerce")
        time_values = numeric if numeric.notna().all() else pd.to_datetime(raw_time, errors="coerce", utc=True)
    if time_values.isna().any():
        raise ValueError(f"Timestamp column {spec.time_column!r} must be numeric or parseable as datetimes.")
    ordered = df.assign(__ced_sort_time=time_values)
    return (
        ordered.sort_values([spec.id_column, "__ced_sort_time"], kind="mergesort")
        .drop(columns=["__ced_sort_time"])
        .reset_index(drop=True)
    )
