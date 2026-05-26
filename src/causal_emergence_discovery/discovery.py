"""End-to-end causal-emergence discovery workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from causal_emergence_discovery.macro import MacroAssignment, generate_candidate_macros, merge_macro_states
from causal_emergence_discovery.models import effect_estimates
from causal_emergence_discovery.panel import build_lagged_table, load_panel_csv, validate_panel
from causal_emergence_discovery.scoring import MACRO_COLUMN
from causal_emergence_discovery.search import greedy_macro_search
from causal_emergence_discovery.spec import StudySpec, load_spec


@dataclass(frozen=True)
class DiscoveryConfig:
    """Configuration for one outcome-oriented discovery run."""

    outcome: str | None = None
    lag: int = 1
    max_states: int = 6
    paths: int = 10
    branching_factor: int = 2
    folds: int = 5
    top_k: int = 10


def discover_from_csv(
    csv_path: str | Path,
    spec_path: str | Path,
    config: DiscoveryConfig | None = None,
) -> dict[str, Any]:
    """Load a CSV and study spec, then run discovery."""
    df = load_panel_csv(str(csv_path))
    spec = load_spec(spec_path)
    return run_discovery(df, spec, config or DiscoveryConfig())


def run_discovery(
    df: pd.DataFrame,
    spec: StudySpec,
    config: DiscoveryConfig | None = None,
) -> dict[str, Any]:
    """Run the MVP causal-emergence discovery workflow."""
    config = config or DiscoveryConfig()
    panel_summary = validate_panel(df, spec)
    outcome = _resolve_outcome(spec, df, config.outcome)
    lagged = build_lagged_table(df, spec, outcomes=[outcome], lag=config.lag)
    target_column = lagged.target_columns[outcome]

    available_columns = list(df.columns)
    micro_features = spec.feature_columns(available_columns)
    intervention_columns = spec.intervention_columns(available_columns)
    environment_columns = spec.environment_columns(available_columns)
    macro_features = spec.macro_feature_columns(available_columns)
    if not macro_features:
        macro_features = [name for name in micro_features if name not in intervention_columns]
    if not macro_features:
        raise ValueError("No candidate macro feature columns are available.")

    initial_macros = generate_candidate_macros(
        lagged.data,
        macro_features,
        max_states=config.max_states,
    )
    if not initial_macros:
        raise ValueError("No candidate macro variables could be generated.")

    searches = [
        greedy_macro_search(
            lagged.data,
            macro,
            outcome=outcome,
            target_column=target_column,
            micro_feature_columns=micro_features,
            intervention_columns=intervention_columns,
            folds=config.folds,
            n_paths=config.paths,
            branching_factor=config.branching_factor,
        )
        for macro in initial_macros
    ]
    top_macros = sorted(
        (search["best"] for search in searches),
        key=lambda item: (
            float(item["score"]),
            float(item["emergence_delta"]),
            float(item["specificity"]),
            int(item["state_count"]),
            str(item["macro_name"]),
        ),
        reverse=True,
    )[: config.top_k]

    best_macro = _find_macro_for_record(initial_macros, searches, top_macros[0])
    pathway_table = lagged.data.copy()
    pathway_table[MACRO_COLUMN] = best_macro.labels.to_numpy()
    adjustment_columns = [MACRO_COLUMN, *environment_columns]
    pathways = effect_estimates(
        pathway_table,
        intervention_columns,
        target_column,
        adjustment_columns,
    )

    return {
        "library": "causal-emergence-discovery",
        "status": "experimental_hypothesis_generation",
        "assumptions": [
            "Rows are longitudinal observations of the same entities over time.",
            "Only earlier rows are used to explain later outcomes at the requested lag.",
            "Intervention coefficients are adjustment-based observational estimates, not proof of causal effects.",
            "Macro variables are scored by outcome clarity, specificity, stability, and compression.",
        ],
        "panel": {
            "row_count": panel_summary.row_count,
            "entity_count": panel_summary.entity_count,
            "duplicate_entity_times": panel_summary.duplicate_entity_times,
            "warnings": list(panel_summary.warnings),
        },
        "config": {
            "outcome": outcome,
            "lag": config.lag,
            "max_states": config.max_states,
            "paths": config.paths,
            "branching_factor": config.branching_factor,
            "folds": config.folds,
            "top_k": config.top_k,
        },
        "columns": {
            "micro_features": micro_features,
            "macro_features": macro_features,
            "interventions": intervention_columns,
            "environments": environment_columns,
            "target": target_column,
        },
        "top_macros": top_macros,
        "candidate_pathways": pathways,
        "searches": searches,
        "warnings": _warnings(intervention_columns, environment_columns),
    }


def _resolve_outcome(spec: StudySpec, df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested outcome {requested!r} is not in the dataset.")
        return requested
    outcomes = spec.outcome_columns(list(df.columns))
    if len(outcomes) != 1:
        raise ValueError(
            "Specify an outcome when the study spec has zero or multiple outcome columns."
        )
    return outcomes[0]


def _find_macro_for_record(
    initial_macros: list[MacroAssignment],
    searches: list[dict[str, Any]],
    record: dict[str, object],
) -> MacroAssignment:
    name = str(record["macro_name"])
    for macro in initial_macros:
        if name == macro.name:
            return macro
        prefix = f"{macro.name}|"
        if name.startswith(prefix):
            return _apply_merge_name(macro, name.removeprefix(prefix))
    return initial_macros[0]


def _apply_merge_name(macro: MacroAssignment, suffix: str) -> MacroAssignment:
    current = macro
    for token in suffix.split("|"):
        if not token.startswith("merge:"):
            continue
        left_text, right_text = token.removeprefix("merge:").split("+", maxsplit=1)
        current = merge_macro_states(current, int(left_text), int(right_text))
    return current


def _warnings(interventions: list[str], environments: list[str]) -> list[str]:
    warnings = [
        "Discovery results are causal hypotheses and require design review, sensitivity checks, and domain validation."
    ]
    if not interventions:
        warnings.append("No intervention columns were declared, so pathway estimates are omitted.")
    if not environments:
        warnings.append("No environment columns were declared, so invariance checks are not yet available.")
    return warnings
