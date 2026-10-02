"""Study specification objects for longitudinal tabular discovery."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CAUSE_ROLES = {
    "intervention",
    "exposure",
    "environment",
    "individual",
    "context",
    "measurement",
    "outcome",
    "covariate",
    "unknown",
}
EFFECT_ROLES = {"outcome", "measurement", "unknown"}
ADJUSTMENT_ROLES = {
    "environment",
    "individual",
    "context",
    "measurement",
    "outcome",
    "covariate",
    "unknown",
}


@dataclass(frozen=True)
class ColumnSpec:
    """Metadata for one observed column."""

    name: str
    role: str = "unknown"
    variable_type: str | None = None
    tags: tuple[str, ...] = ()
    allowed_as_cause: bool | None = None
    allowed_as_effect: bool | None = None
    allowed_as_adjustment: bool | None = None

    @classmethod
    def from_mapping(cls, name: str, data: dict[str, Any] | str | None) -> "ColumnSpec":
        if data is None:
            return cls(name=name)
        if isinstance(data, str):
            return cls(name=name, role=data)
        tags = data.get("tags", ())
        if isinstance(tags, str):
            tags = (tags,)
        return cls(
            name=name,
            role=data.get("role", "unknown"),
            variable_type=data.get("type") or data.get("variable_type"),
            tags=tuple(tags),
            allowed_as_cause=data.get("allowed_as_cause"),
            allowed_as_effect=data.get("allowed_as_effect"),
            allowed_as_adjustment=data.get("allowed_as_adjustment"),
        )

    def can_cause(self) -> bool:
        if self.allowed_as_cause is not None:
            return self.allowed_as_cause
        return self.role in CAUSE_ROLES

    def can_be_effect(self) -> bool:
        if self.allowed_as_effect is not None:
            return self.allowed_as_effect
        return self.role in EFFECT_ROLES

    def can_adjust(self) -> bool:
        if self.allowed_as_adjustment is not None:
            return self.allowed_as_adjustment
        return self.role in ADJUSTMENT_ROLES


@dataclass(frozen=True)
class AdjustmentSpec:
    """Explicitly declared adjustment variables and their rationale."""

    columns: tuple[str, ...]
    rationale: str
    include_macro: bool = False

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "AdjustmentSpec | None":
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("adjustment must be an object with columns and rationale.")
        raw_columns = value.get("columns", ())
        if isinstance(raw_columns, str) or not isinstance(raw_columns, (list, tuple)):
            raise ValueError("adjustment.columns must be a list of column names.")
        if any(not isinstance(column, str) or not column.strip() for column in raw_columns):
            raise ValueError("adjustment.columns must contain non-empty string column names.")
        columns = tuple(raw_columns)
        if len(set(columns)) != len(columns):
            raise ValueError("adjustment.columns must not contain duplicates.")
        rationale = value.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("An explicit adjustment requires a non-empty rationale.")
        include_macro = value.get("include_macro", False)
        if not isinstance(include_macro, bool):
            raise ValueError("adjustment.include_macro must be a boolean.")
        return cls(columns=columns, rationale=rationale.strip(), include_macro=include_macro)

    def to_dict(self) -> dict[str, Any]:
        return {"columns": list(self.columns), "rationale": self.rationale, "include_macro": self.include_macro}


@dataclass(frozen=True)
class StudySpec:
    """A minimal, domain-agnostic contract for a longitudinal table."""

    id_column: str
    time_column: str
    columns: dict[str, ColumnSpec] = field(default_factory=dict)
    outcomes: tuple[str, ...] = ()
    interventions: tuple[str, ...] = ()
    environments: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    adjustment: AdjustmentSpec | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StudySpec":
        dataset = data.get("dataset", data)
        id_column = dataset.get("id_column") or data.get("id_column")
        time_column = dataset.get("time_column") or data.get("time_column")
        if not id_column or not time_column:
            raise ValueError("Study spec requires id_column and time_column.")

        raw_columns = data.get("columns", {})
        columns = {
            name: ColumnSpec.from_mapping(name, value)
            for name, value in raw_columns.items()
        }
        outcomes = tuple(data.get("outcomes", ()))
        interventions = tuple(data.get("interventions", ()))
        environments = tuple(data.get("environments", ()))
        exclude = tuple(data.get("exclude", ()))
        adjustment = AdjustmentSpec.from_mapping(data.get("adjustment"))

        columns = _with_role_defaults(columns, outcomes, "outcome")
        columns = _with_role_defaults(columns, interventions, "intervention")
        columns = _with_role_defaults(columns, environments, "environment")
        columns = _with_role_defaults(columns, exclude, "exclude")

        return cls(
            id_column=id_column,
            time_column=time_column,
            columns=columns,
            outcomes=outcomes,
            interventions=interventions,
            environments=environments,
            exclude=exclude,
            metadata=data.get("metadata", {}),
            adjustment=adjustment,
        )

    def __post_init__(self) -> None:
        if self.adjustment is None:
            return
        forbidden = {self.id_column, self.time_column, "__lead_time", *self.outcomes, *self.interventions}
        forbidden.update(
            name for name, column in self.columns.items()
            if column.role in {"outcome", "intervention", "exposure", "exclude"}
        )
        invalid = [
            name for name in self.adjustment.columns
            if name in forbidden or name.startswith("__lead")
        ]
        if invalid:
            raise ValueError(f"Adjustment columns cannot include identifiers, time, treatments, or outcomes: {invalid}.")
        excluded = [
            name for name in self.adjustment.columns
            if name in self.exclude or self.column_spec(name).role == "exclude"
        ]
        if excluded:
            raise ValueError(f"Excluded columns cannot be adjustment variables: {excluded}.")
        disallowed = [name for name in self.adjustment.columns if not self.column_spec(name).can_adjust()]
        if disallowed:
            raise ValueError(f"Columns marked allowed_as_adjustment=false cannot be adjustment variables: {disallowed}.")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "dataset": {"id_column": self.id_column, "time_column": self.time_column},
            "columns": {
                name: {
                    "role": spec.role,
                    **({"type": spec.variable_type} if spec.variable_type is not None else {}),
                    **({"tags": list(spec.tags)} if spec.tags else {}),
                    **({"allowed_as_cause": spec.allowed_as_cause} if spec.allowed_as_cause is not None else {}),
                    **({"allowed_as_effect": spec.allowed_as_effect} if spec.allowed_as_effect is not None else {}),
                    **({"allowed_as_adjustment": spec.allowed_as_adjustment} if spec.allowed_as_adjustment is not None else {}),
                }
                for name, spec in self.columns.items()
            },
            "outcomes": list(self.outcomes),
            "interventions": list(self.interventions),
            "environments": list(self.environments),
            "exclude": list(self.exclude),
            "metadata": self.metadata,
        }
        if self.adjustment is not None:
            result["adjustment"] = self.adjustment.to_dict()
        return result

    def column_spec(self, name: str) -> ColumnSpec:
        return self.columns.get(name, ColumnSpec(name=name))

    def role_columns(self, role: str, available_columns: list[str] | None = None) -> list[str]:
        columns = [
            name
            for name, spec in self.columns.items()
            if spec.role == role and name not in self.exclude
        ]
        if role == "outcome":
            columns.extend(name for name in self.outcomes if name not in columns)
        if role == "intervention":
            columns.extend(name for name in self.interventions if name not in columns)
        if role == "environment":
            columns.extend(name for name in self.environments if name not in columns)
        if available_columns is not None:
            allowed = set(available_columns)
            columns = [name for name in columns if name in allowed]
        return columns

    def outcome_columns(self, available_columns: list[str] | None = None) -> list[str]:
        return self.role_columns("outcome", available_columns)

    def intervention_columns(self, available_columns: list[str] | None = None) -> list[str]:
        return self.role_columns("intervention", available_columns)

    def environment_columns(self, available_columns: list[str] | None = None) -> list[str]:
        return self.role_columns("environment", available_columns)

    def feature_columns(self, available_columns: list[str]) -> list[str]:
        reserved = {self.id_column, self.time_column, *self.exclude}
        features = []
        for name in available_columns:
            if name in reserved:
                continue
            spec = self.column_spec(name)
            if spec.role == "exclude":
                continue
            if spec.can_cause() or spec.can_adjust():
                features.append(name)
        return features

    def macro_feature_columns(self, available_columns: list[str]) -> list[str]:
        excluded_roles = {"intervention", "exposure", "exclude"}
        return [
            name
            for name in self.feature_columns(available_columns)
            if self.column_spec(name).role not in excluded_roles
        ]


def load_spec(path: str | Path) -> StudySpec:
    """Load a study spec from JSON.

    YAML is intentionally not required for the MVP to keep installation light.
    JSON is a strict subset of YAML, so the same structure can later be loaded
    with PyYAML without changing the spec contract.
    """
    spec_path = Path(path)
    text = spec_path.read_text(encoding="utf-8")
    if spec_path.suffix.lower() != ".json":
        raise ValueError("Only JSON study specs are supported in the MVP.")
    return StudySpec.from_dict(json.loads(text))


def _with_role_defaults(
    columns: dict[str, ColumnSpec],
    names: tuple[str, ...],
    role: str,
) -> dict[str, ColumnSpec]:
    updated = dict(columns)
    for name in names:
        if name not in updated:
            updated[name] = ColumnSpec(name=name, role=role)
    return updated
