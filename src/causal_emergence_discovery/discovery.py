"""End-to-end causal-emergence discovery workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from causal_emergence_discovery.macro import MacroAssignment, apply_macro, generate_candidate_macros, merge_macro_states, refit_macro
from causal_emergence_discovery.models import fit_design_encoder, fit_linear_model, predict_r2
from causal_emergence_discovery.panel import build_lagged_table, load_panel_csv, validate_panel
from causal_emergence_discovery.scoring import MACRO_COLUMN, SCORE_SCHEMA_VERSION
from causal_emergence_discovery.search import greedy_macro_search
from causal_emergence_discovery.schema import resolve_feature_schema
from causal_emergence_discovery.spec import ColumnSpec, StudySpec, load_spec
from causal_emergence_discovery.validation import build_validation_plan, target_support_audit


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
    validation_mode: str = "entity_holdout"
    holdout_fraction: float = 0.2
    seed: int = 0


def discover_from_csv(
    csv_path: str | Path,
    spec_path: str | Path,
    config: DiscoveryConfig | None = None,
) -> dict[str, Any]:
    """Load a CSV and study spec, then run discovery."""
    spec = load_spec(spec_path)
    df = load_panel_csv(str(csv_path), spec=spec)
    return run_discovery(df, spec, config or DiscoveryConfig())


def run_discovery(
    df: pd.DataFrame,
    spec: StudySpec,
    config: DiscoveryConfig | None = None,
) -> dict[str, Any]:
    """Run the MVP causal-emergence discovery workflow."""
    config = config or DiscoveryConfig()
    if config.top_k < 1 or config.max_states < 2:
        raise ValueError("top_k must be positive and max_states must be at least two.")
    panel_summary = validate_panel(df, spec)
    outcome = _resolve_outcome(spec, df, config.outcome)
    lagged = build_lagged_table(df, spec, outcomes=[outcome], lag=config.lag)
    target_column = lagged.target_columns[outcome]
    plan = build_validation_plan(
        lagged.data,
        id_column=spec.id_column,
        time_column=spec.time_column,
        target_time_column=lagged.target_time_column,
        mode=config.validation_mode,
        folds=config.folds,
        holdout_fraction=config.holdout_fraction,
        seed=config.seed,
    )
    development = lagged.data.iloc[list(plan.outer.train_indices)].reset_index(drop=True)
    holdout = lagged.data.iloc[list(plan.outer.test_indices)].reset_index(drop=True)
    target_support = target_support_audit(lagged.data, plan, target_column)
    _require_development_target_support(development, plan.inner, target_column)

    available_columns = list(df.columns)
    micro_features = spec.feature_columns(available_columns)
    intervention_columns = spec.intervention_columns(available_columns)
    environment_columns = spec.environment_columns(available_columns)
    macro_features = spec.macro_feature_columns(available_columns)
    if not macro_features:
        macro_features = [name for name in micro_features if name not in intervention_columns]
    if not macro_features:
        raise ValueError("No candidate macro feature columns are available.")
    schema_columns = list(dict.fromkeys([
        *micro_features, *intervention_columns, *macro_features,
        *(spec.adjustment.columns if spec.adjustment is not None else ()),
    ]))
    feature_schema = resolve_feature_schema(development, schema_columns, column_specs=spec.columns)
    feature_application = fit_design_encoder(development, schema_columns, schema=feature_schema).transform_with_report(holdout)[1]
    schema_report = {
        name: {
            "kind": kind,
            "source": "declared" if spec.column_spec(name).variable_type else "outer_training_inference",
            "invalid_outer_values": feature_application[name]["invalid_numeric"],
            "unknown_outer_categories": feature_application[name]["unknown_category"],
            "missing_outer_values": feature_application[name]["missing"],
        }
        for name, kind in feature_schema.kinds
    }

    initial_macros = generate_candidate_macros(
        development,
        macro_features,
        max_states=config.max_states,
        column_specs=spec.columns,
    )
    if not initial_macros:
        raise ValueError("No candidate macro variables could be generated.")

    searches = [
        greedy_macro_search(
            development,
            macro,
            outcome=outcome,
            target_column=target_column,
            micro_feature_columns=micro_features,
            intervention_columns=intervention_columns,
            folds=config.folds,
            n_paths=config.paths,
            branching_factor=config.branching_factor,
            validation_splits=plan.inner,
            column_specs=spec.columns,
        )
        for macro in initial_macros
    ]
    top_macros = sorted(
        (search["best"] for search in searches if search["best"] is not None),
        key=lambda item: (
            float(item["ranking_score"]),
            float(item["specificity"]),
            int(item["state_count"]),
            str(item["macro_name"]),
        ),
        reverse=True,
    )[: config.top_k]
    if not top_macros:
        raise ValueError("No candidate macro can be scored on every required development fold.")
    for record in top_macros:
        record["evaluation_scope"] = "development_selection_only"

    best_macro = refit_macro(_find_macro_for_record(initial_macros, searches, top_macros[0]), development, column_specs=spec.columns)
    evaluation = _evaluate_selected_macro(
        development, holdout, best_macro, target_column, micro_features, intervention_columns,
        column_specs=spec.columns,
    )
    pathway_table = development.copy()
    pathway_table[MACRO_COLUMN] = best_macro.labels.astype(str).to_numpy()
    adjustment = _adjustment_report(pathway_table, spec, best_macro, intervention_columns, target_column)
    pathways = adjustment["pathways"]

    return {
        "library": "causal-emergence-discovery",
        "schema_version": SCORE_SCHEMA_VERSION,
        "status": "experimental_hypothesis_generation",
        "assumptions": [
            "Rows are longitudinal observations of the same entities over time.",
            "Lag means the requested number of observations within an entity, not necessarily calendar intervals.",
            "The declared validation mode determines the predictive estimand; training lead outcomes respect split boundaries.",
            "Macro recipes and ranking use development data only; the selected recipe is evaluated once on the reserved holdout.",
            "Intervention coefficients are adjustment-based observational estimates, not proof of causal effects.",
            "Macro variables are ranked by heuristic development prediction, validation specificity, fold dispersion, and compression preferences.",
        ],
        "score_semantics": {
            "schema_version": 2,
            "ranking_score": "heuristic preference for candidate ranking; not an emergence statistic",
            "specificity": "mean_positive_clipped_validation_r2_of_training_fitted_state_means",
            "outcome_specificity": "descriptive in-sample between-state outcome variance share",
            "predictive_comparison": "raw paired macro-minus-micro R2; caller scope labels do not certify split independence",
            "predictive_provenance": "computed_training_only_refit",
            "emergence_evidence_status": "not_assessed",
        },
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
            "validation_mode": config.validation_mode,
            "holdout_fraction": config.holdout_fraction,
            "seed": config.seed,
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
        "adjustment": adjustment,
        "validation": {**plan.to_dict(), "target_support": target_support},
        "feature_schema": schema_report,
        "outer_evaluation": evaluation,
        "searches": searches,
        "warnings": [
            *_warnings(intervention_columns, environment_columns),
            *adjustment.get("warnings", []),
            *([evaluation["reason"]] if evaluation["status"] == "not_evaluable" else []),
        ],
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
    # Recipe names can collide (e.g. a column literally named 'composite').
    # Recover the originating search, not the first matching display name.
    for macro, search in zip(initial_macros, searches):
        if search["best"] is record or search["best"] == record:
            if name == macro.name:
                return macro
            prefix = f"{macro.name}|"
            if name.startswith(prefix):
                return _apply_merge_name(macro, name.removeprefix(prefix))
    raise ValueError(f"Selected macro recipe could not be recovered: {name}")


def _evaluate_selected_macro(
    train: pd.DataFrame,
    test: pd.DataFrame,
    macro: MacroAssignment,
    target_column: str,
    micro_features: list[str],
    interventions: list[str],
    *,
    column_specs=None,
) -> dict[str, Any]:
    """Evaluate one locked recipe; no holdout-driven selection or specificity."""
    train = train.copy()
    test = test.copy()
    test_macro = apply_macro(macro, test)
    train[MACRO_COLUMN] = macro.labels.astype(str).to_numpy()
    test[MACRO_COLUMN] = test_macro.labels.astype(str).to_numpy()
    macro_columns = list(dict.fromkeys([*interventions, MACRO_COLUMN]))
    micro_columns = list(dict.fromkeys([*interventions, *micro_features]))
    model_specs = {**(column_specs or {}), MACRO_COLUMN: ColumnSpec(name=MACRO_COLUMN, variable_type="categorical")}
    macro_fit = fit_linear_model(train, macro_columns, target_column, column_specs=model_specs)
    micro_fit = fit_linear_model(train, micro_columns, target_column, column_specs=model_specs)
    test_target = pd.to_numeric(test[target_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    seen_states = set(macro.labels.tolist())
    result = {
        "scope": "untouched_outer_holdout_selected_macro",
        "macro_name": macro.name,
        "status": "not_evaluable",
        "macro_r2": None,
        "micro_r2": None,
        "predictive_r2_difference": None,
        "train_rows": len(train),
        "test_rows": len(test),
        "finite_train_targets": macro_fit.nobs,
        "finite_test_targets": int(test_target.notna().sum()),
        "nonfinite_test_targets_excluded": int(test_target.isna().sum()),
        "metric_weighting": "Rows contribute equally within a split; development fold R2 values are averaged equally.",
        "training_state_count": macro.state_count,
        "holdout_state_sizes": {str(int(state)): int(count) for state, count in test_macro.labels.value_counts().sort_index().items()},
        "unseen_state_rows": int((~test_macro.labels.isin(seen_states)).sum()),
        "encoder": macro.metadata,
        "feature_application": {
            "macro_encoder": test_macro.metadata.get("application_diagnostics", {}),
            "macro_design": macro_fit.encoder.transform_with_report(test)[1],
            "micro_design": micro_fit.encoder.transform_with_report(test)[1],
        },
        "note": "Predictive metrics for the selected recipe, not a causal effect or a composite emergence score. Negative R2 is retained.",
    }
    usable = test_target.dropna()
    if len(usable) < 2:
        result["reason"] = (
            f"Reserved outer holdout has {len(usable)} finite target rows among {len(test)} reserved rows; "
            "evaluation requires at least two. The reserved cohort was not replaced."
        )
        return result
    if usable.nunique() < 2:
        result["reason"] = "Reserved outer holdout target is constant; R2 is undefined. The reserved cohort was not replaced."
        return result
    macro_r2 = predict_r2(macro_fit, test, macro_columns, target_column)
    micro_r2 = predict_r2(micro_fit, test, micro_columns, target_column)
    result.update(
        status="evaluated", macro_r2=macro_r2, micro_r2=micro_r2,
        predictive_r2_difference=macro_r2 - micro_r2,
    )
    return result


def _require_development_target_support(development, inner_splits, target_column):
    """Reject unusable development targets without adapting the locked plan."""
    target = pd.to_numeric(development[target_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for split in inner_splits:
        for side, indices in (("train", split.train_indices), ("test", split.test_indices)):
            values = target.iloc[list(indices)].dropna()
            if len(values) < 2 or (side == "test" and values.nunique() < 2):
                raise ValueError(
                    f"Reserved development split {split.name!r} {side} targets cannot support "
                    f"evaluation: {len(values)} finite targets among {len(indices)} reserved rows "
                    f"and {values.nunique()} distinct targets. The locked split was not replaced."
                )


def _adjustment_report(df, spec, macro, interventions, target_column):
    # Kept separate from predictive evaluation: never fit pathways on holdout outcomes.
    from causal_emergence_discovery.adjustment import adjustment_sufficiency_audit
    return adjustment_sufficiency_audit(df, spec, macro, interventions=interventions, target_column=target_column)


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
