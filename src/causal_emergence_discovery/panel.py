"""Panel data validation and lagged table construction."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

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


def load_panel_csv(path: str) -> pd.DataFrame:
    """Load a CSV as a pandas DataFrame."""
    return pd.read_csv(path)


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

    ordered = _sort_panel(df, spec)
    result = ordered.copy()
    target_columns: dict[str, str] = {}
    grouped = result.groupby(spec.id_column, sort=False)
    for outcome in selected_outcomes:
        target = f"{outcome}__lead{lag}"
        target_columns[outcome] = target
        result[target] = grouped[outcome].shift(-lag)
    result["__lead_time"] = grouped[spec.time_column].shift(-lag)
    result = result.dropna(subset=list(target_columns.values())).reset_index(drop=True)
    return LaggedDataset(
        data=result,
        outcome_columns=selected_outcomes,
        target_columns=target_columns,
        lag=lag,
        id_column=spec.id_column,
        time_column=spec.time_column,
    )


def _sort_panel(df: pd.DataFrame, spec: StudySpec) -> pd.DataFrame:
    parsed_time = pd.to_datetime(df[spec.time_column], errors="coerce")
    time_values = parsed_time if parsed_time.notna().all() else df[spec.time_column]
    ordered = df.assign(__ced_sort_time=time_values)
    return (
        ordered.sort_values([spec.id_column, "__ced_sort_time"], kind="mergesort")
        .drop(columns=["__ced_sort_time"])
        .reset_index(drop=True)
    )
