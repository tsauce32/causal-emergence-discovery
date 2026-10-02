"""Explicit adjustment declarations and exploratory sufficiency diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from causal_emergence_discovery.macro import MacroAssignment
from causal_emergence_discovery.spec import ColumnSpec, StudySpec


def resolve_adjustment(
    spec: StudySpec,
    available_columns: list[str],
    *,
    interventions: list[str],
    target_column: str,
) -> dict[str, Any]:
    """Resolve only explicitly declared covariates; never infer from column roles.

    This returns a compact audit object useful to CLI/reporting callers. Roles such as
    ``context`` and ``environment`` do not opt a variable into adjustment.
    """
    declaration = spec.adjustment
    requested = [] if declaration is None else list(declaration.columns)
    if declaration is None:
        return {
            "columns": [],
            "declared": False,
            "rationale": None,
            "include_macro": False,
            "audit": "No adjustment variables were explicitly declared; no context or environment columns were inferred.",
        }

    forbidden = {spec.id_column, spec.time_column, "__lead_time", target_column, *interventions, *spec.outcomes, *spec.interventions}
    forbidden.update(name for name, column in spec.columns.items() if column.role in {"outcome", "intervention", "exposure", "exclude"})
    invalid = [column for column in requested if column in forbidden or column.startswith("__lead")]
    if invalid:
        raise ValueError(f"Adjustment columns cannot include identifiers, time, treatments, or outcomes: {invalid}.")
    missing = [column for column in requested if column not in available_columns]
    if missing:
        raise ValueError(f"Declared adjustment columns are unavailable: {missing}.")
    disallowed = [column for column in requested if not spec.column_spec(column).can_adjust()]
    if disallowed:
        raise ValueError(f"Columns marked allowed_as_adjustment=false cannot be adjustment variables: {disallowed}.")
    excluded = [column for column in requested if column in spec.exclude or spec.column_spec(column).role == "exclude"]
    if excluded:
        raise ValueError(f"Excluded columns cannot be adjustment variables: {excluded}.")
    if target_column in spec.exclude or target_column == spec.time_column:
        raise ValueError("The target cannot be an excluded or time column.")
    return {
        "columns": requested,
        "declared": True,
        "rationale": declaration.rationale,
        "include_macro": declaration.include_macro,
        "audit": "Adjustment variables come only from the explicit study declaration.",
    }


def adjustment_sufficiency_audit(
    df: pd.DataFrame,
    spec: StudySpec,
    macro: MacroAssignment | None = None,
    *,
    interventions: list[str],
    target_column: str,
    estimand: str | None = None,
) -> dict[str, Any]:
    """Compare exploratory treatment coefficients with declared covariates.

    ``df`` should contain development data only. By default, all declared
    interventions enter one joint conditional model. ``estimand="marginal"`` fits
    each intervention separately with the explicit adjustment covariates. When the
    keyword is omitted, a declared ``adjustment.estimand`` is used if present.
    Coefficients and IID standard errors are descriptive and do not establish
    exchangeability, correct temporal ordering, absence of selection bias, or a
    causal effect. A coarse macro is a representation of its source features, not a
    substitute for declared continuous covariates.
    """
    resolved = resolve_adjustment(spec, list(df.columns), interventions=interventions, target_column=target_column)
    effective_estimand = _resolve_estimand(spec, estimand)
    if resolved["include_macro"] and macro is None:
        raise ValueError("adjustment.include_macro=true requires a fitted MacroAssignment for the audit.")
    working = df.copy()
    macro_column = "__ced_adjustment_macro"
    macro_features: list[str] = []
    if macro is not None:
        if len(macro.labels) != len(working):
            raise ValueError("Macro labels must align one-to-one with the supplied development rows.")
        working[macro_column] = macro.labels.astype("string").to_numpy()
        macro_features = list(macro.feature_columns)
    macro_covariates = [macro_column] if macro is not None else []
    declared_columns = list(resolved["columns"])
    declared_model_adjustments = declared_columns + (macro_covariates if resolved["include_macro"] else [])

    model_specs = {
        **spec.columns,
        macro_column: ColumnSpec(name=macro_column, variable_type="categorical"),
    }
    macro_only = _coefficient_comparison(
        working, interventions, target_column, macro_covariates,
        estimand=effective_estimand, column_specs=model_specs,
    )
    declared_fit = _coefficient_comparison(
        working, interventions, target_column, declared_model_adjustments,
        estimand=effective_estimand, column_specs=model_specs,
    )
    source_overlap = [column for column in declared_columns if column in macro_features]
    omitted = [column for column in declared_columns if column not in macro_features]
    coarsened = []
    if macro is not None:
        for column in source_overlap:
            observed = int(working[column].nunique(dropna=True)) if column in working else 0
            if macro.state_count < observed:
                coarsened.append({"column": column, "observed_values": observed, "macro_states": macro.state_count})
    covariate_residuals = _covariate_residual_diagnostics(working, declared_columns, interventions, macro_column if macro is not None else None)
    macro_by_term = {(item.get("intervention"), item.get("comparison_term")): item for item in _term_records(macro_only)}
    adjusted_by_term = {(item.get("intervention"), item.get("comparison_term")): item for item in _term_records(declared_fit)}
    coefficient_changes = []
    for intervention, term in dict.fromkeys([*macro_by_term, *adjusted_by_term]):
        before = macro_by_term.get((intervention, term), {}).get("coefficient")
        after = adjusted_by_term.get((intervention, term), {}).get("coefficient")
        coefficient_changes.append({
            "intervention": intervention,
            "term": term,
            "term_metadata": {
                key: source.get(key)
                for key in ("kind", "level", "reference_level", "feature_name")
                if (source := adjusted_by_term.get((intervention, term), macro_by_term.get((intervention, term)))) is not None and key in source
            },
            "macro_only_coefficient": before,
            "declared_adjustment_coefficient": after,
            "difference_adjusted_minus_macro_only": None if before is None or after is None else _finite_or_none(float(after) - float(before)),
        })
    warnings = [
        "Associational regression diagnostics do not certify causal identification or confounder sufficiency.",
        "IID standard errors do not account for repeated entities; panel-robust uncertainty is unsupported.",
        "Within-macro residual diagnostics describe remaining information in this development sample; they are not causal tests.",
    ]
    if not resolved["declared"]:
        warnings.append("No adjustment covariates were explicitly declared; pathways are conditional only on any co-treatments included by the selected estimand.")
    return {
        "target": target_column,
        "interventions": list(interventions),
        "estimand": effective_estimand,
        "fit_specification": {
            "estimand": effective_estimand,
            "joint_conditional_terms": list(dict.fromkeys([*interventions, *declared_model_adjustments])),
            "explicit_covariates": declared_columns,
            "macro_only_covariates": macro_covariates,
            "declared_fit_covariates": declared_model_adjustments,
            "marginal_terms_by_intervention": {
                intervention: list(dict.fromkeys([intervention, *declared_model_adjustments]))
                for intervention in interventions
            },
            "coefficient_interpretation": (
                "Associational linear-regression coefficients conditional on the listed model terms; "
                "they do not establish causal effects. No mediators, colliders, or context variables are "
                "automatically selected for adjustment."
            ),
        },
        "uncertainty": {
            "standard_errors": "IID OLS",
            "panel_uncertainty_status": "unsupported",
            "panel_uncertainty_note": "Reported standard errors treat rows as IID and do not account for repeated entities.",
        },
        "adjustment": resolved,
        "macro": None if macro is None else {"name": macro.name, "states": macro.state_count, "feature_columns": macro_features},
        "comparison": {"macro_only": macro_only, "declared_adjustment": declared_fit, "coefficient_changes": coefficient_changes},
        "pathways": declared_fit,
        "warnings": warnings,
        "evidence_scope": "development_only",
        "declared_covariate_diagnostics": {
            "represented_as_macro_features": source_overlap,
            "omitted_from_macro_features": omitted,
            "coarsened_by_macro": coarsened,
            "within_macro_residual_information": covariate_residuals,
        },
        "interpretation": (
            "Exploratory observational regression comparison. The coefficients are not causally certified; "
            "a coarse macro does not replace declared continuous covariates, and this report does not establish "
            "confounder sufficiency or causal identification. IID standard errors do not account for repeated entities; "
            "panel-robust uncertainty is unsupported."
        ),
        "data_scope": "Use development rows only; do not use final holdout outcomes for this audit.",
        "context_interpretation": "Context and environment roles are descriptive metadata and do not imply adjustment unless named explicitly.",
    }


def _coefficient_comparison(
    df: pd.DataFrame,
    interventions: list[str],
    target_column: str,
    adjustment_columns: list[str],
    *,
    estimand: str = "joint_conditional",
    column_specs=None,
) -> list[dict[str, Any]]:
    from causal_emergence_discovery.models import treatment_effect_records

    records = treatment_effect_records(
        df, interventions, target_column, adjustment_columns,
        estimand=estimand, column_specs=column_specs,
    )
    return [{"intervention": item.get("treatment"), **item} for item in records]


def _resolve_estimand(spec: StudySpec, requested: str | None) -> str:
    if requested is None:
        declaration = spec.adjustment
        requested = getattr(declaration, "estimand", None) or "joint_conditional"
    if not isinstance(requested, str) or requested not in {"joint_conditional", "marginal"}:
        raise ValueError("estimand must be 'joint_conditional' or 'marginal'.")
    return requested


def _term_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten intervention records for comparing like-for-like terms."""
    flattened = []
    for record in records:
        terms = record.get("terms")
        if not isinstance(terms, list) or not terms:
            flattened.append(record)
            continue
        for term in terms:
            if isinstance(term, dict):
                flattened.append({
                    **record,
                    **term,
                    "comparison_term": term.get("feature_name", term.get("term")),
                    "terms": terms,
                })
    return flattened


def _covariate_residual_diagnostics(
    df: pd.DataFrame,
    covariates: list[str],
    interventions: list[str],
    macro_column: str | None,
) -> list[dict[str, Any]]:
    """Describe numeric confounder information remaining after macro state means."""
    if macro_column is None:
        group_design = np.ones((len(df), 1), dtype=float)
    else:
        groups = df[macro_column].astype("string")
        codes, _ = pd.factorize(groups, sort=True, use_na_sentinel=False)
        levels = int(codes.max()) + 1 if len(codes) else 0
        dummies = (
            np.column_stack([(codes == code).astype(float) for code in range(1, levels)])
            if levels > 1 else np.empty((len(df), 0), dtype=float)
        )
        group_design = np.column_stack([np.ones(len(df)), dummies])
    diagnostics = []
    for column in covariates:
        values = pd.to_numeric(df[column], errors="coerce").astype(float).to_numpy()
        valid = np.isfinite(values)
        if valid.sum() < 2:
            diagnostics.append({"column": column, "status": "insufficient_numeric_values"})
            continue
        design = group_design[valid]
        covariate = values[valid]
        centered = covariate - covariate.mean()
        total_variance = float(np.mean(centered ** 2))
        cov_residual = covariate - design @ np.linalg.lstsq(design, covariate, rcond=None)[0]
        remaining = float(np.mean(cov_residual ** 2))
        correlations = []
        for intervention in interventions:
            treatment = pd.to_numeric(df[intervention], errors="coerce").astype(float).to_numpy()
            treatment_valid = valid & np.isfinite(treatment)
            if treatment_valid.sum() < 2:
                correlations.append({"intervention": intervention, "residual_correlation": None, "nobs": int(treatment_valid.sum())})
                continue
            local_design = group_design[treatment_valid]
            x = values[treatment_valid]
            z = treatment[treatment_valid]
            x_residual = x - local_design @ np.linalg.lstsq(local_design, x, rcond=None)[0]
            z_residual = z - local_design @ np.linalg.lstsq(local_design, z, rcond=None)[0]
            denom = float(np.std(x_residual) * np.std(z_residual))
            correlation = None if denom == 0.0 else float(np.mean((x_residual - x_residual.mean()) * (z_residual - z_residual.mean())) / denom)
            correlations.append({"intervention": intervention, "residual_correlation": correlation, "nobs": int(treatment_valid.sum())})
        total_variance_out = _finite_or_none(total_variance)
        remaining_out = _finite_or_none(remaining)
        fraction = None if total_variance == 0.0 else _finite_or_none(remaining / total_variance)
        correlations = [
            {**item, "residual_correlation": _finite_or_none(item["residual_correlation"])}
            for item in correlations
        ]
        diagnostics.append({
            "column": column,
            "status": "ok" if total_variance_out is not None and remaining_out is not None else "nonfinite_diagnostic",
            "numeric": True,
            "total_variance": total_variance_out,
            "within_macro_variance": remaining_out,
            "within_macro_variance_fraction_remaining": fraction,
            "treatment_covariate_residual_correlations": correlations,
        })
    return diagnostics


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None
