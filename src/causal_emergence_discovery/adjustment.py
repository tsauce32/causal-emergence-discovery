"""Explicit adjustment declarations and exploratory sufficiency diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from causal_emergence_discovery.macro import MacroAssignment
from causal_emergence_discovery.models import fit_linear_model
from causal_emergence_discovery.spec import StudySpec


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
) -> dict[str, Any]:
    """Compare exploratory treatment coefficients with declared covariates.

    ``df`` should contain development data only. The comparison is descriptive: OLS
    coefficients and IID standard errors do not establish exchangeability, correct
    temporal ordering, absence of selection bias, or a causal effect. A coarse macro
    is reported as a representation of its source features, not as a substitute for
    the declared continuous covariates.
    """
    resolved = resolve_adjustment(spec, list(df.columns), interventions=interventions, target_column=target_column)
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

    macro_only = _coefficient_comparison(working, interventions, target_column, macro_covariates)
    declared_fit = _coefficient_comparison(working, interventions, target_column, declared_model_adjustments)
    source_overlap = [column for column in declared_columns if column in macro_features]
    omitted = [column for column in declared_columns if column not in macro_features]
    coarsened = []
    if macro is not None:
        for column in source_overlap:
            observed = int(working[column].nunique(dropna=True)) if column in working else 0
            if macro.state_count < observed:
                coarsened.append({"column": column, "observed_values": observed, "macro_states": macro.state_count})
    covariate_residuals = _covariate_residual_diagnostics(working, declared_columns, interventions, macro_column if macro is not None else None)
    macro_by_intervention = {item["intervention"]: item for item in macro_only}
    adjusted_by_intervention = {item["intervention"]: item for item in declared_fit}
    coefficient_changes = []
    for intervention in interventions:
        before = macro_by_intervention.get(intervention, {}).get("coefficient")
        after = adjusted_by_intervention.get(intervention, {}).get("coefficient")
        coefficient_changes.append({
            "intervention": intervention,
            "macro_only_coefficient": before,
            "declared_adjustment_coefficient": after,
            "difference_adjusted_minus_macro_only": None if before is None or after is None else float(after) - float(before),
        })
    warnings = [
        "Associational regression diagnostics do not certify causal identification or confounder sufficiency.",
        "IID standard errors do not account for repeated entities.",
        "Within-macro residual diagnostics describe remaining information in this development sample; they are not causal tests.",
    ]
    if not resolved["declared"]:
        warnings.append("No adjustment variables were explicitly declared; reported pathways are unadjusted treatment associations.")
    return {
        "target": target_column,
        "interventions": list(interventions),
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
            "confounder sufficiency or causal identification. IID standard errors do not account for repeated entities."
        ),
        "data_scope": "Use development rows only; do not use final holdout outcomes for this audit.",
        "context_interpretation": "Context and environment roles are descriptive metadata and do not imply adjustment unless named explicitly.",
    }


def _coefficient_comparison(
    df: pd.DataFrame,
    interventions: list[str],
    target_column: str,
    adjustment_columns: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for intervention in interventions:
        features = [intervention, *[column for column in adjustment_columns if column != intervention]]
        try:
            fit = fit_linear_model(df, features, target_column)
        except (ValueError, KeyError) as exc:
            records.append({"intervention": intervention, "coefficient": None, "stderr": None, "nobs": 0, "error": str(exc)})
            continue
        if intervention not in fit.feature_names:
            records.append({"intervention": intervention, "coefficient": None, "stderr": None, "nobs": fit.nobs, "note": "intervention was not represented as a single numeric term"})
            continue
        index = fit.feature_names.index(intervention)
        coefficient = float(fit.coefficients[index])
        stderr = float(fit.stderr[index])
        records.append({
            "intervention": intervention,
            "target": target_column,
            "coefficient": coefficient,
            "stderr": stderr,
            "stderr_iid": stderr,
            "t_stat": None if stderr == 0.0 else coefficient / stderr,
            "nobs": fit.nobs,
            "model_r2": float(fit.r2),
        })
    return records


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
        groups = df[macro_column].astype("string").fillna("__missing__")
        dummies = pd.get_dummies(groups, drop_first=True, dtype=float)
        group_design = np.column_stack([np.ones(len(df)), dummies.to_numpy(dtype=float)])
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
        diagnostics.append({
            "column": column,
            "status": "ok",
            "numeric": True,
            "total_variance": total_variance,
            "within_macro_variance": remaining,
            "within_macro_variance_fraction_remaining": None if total_variance == 0.0 else remaining / total_variance,
            "treatment_covariate_residual_correlations": correlations,
        })
    return diagnostics
