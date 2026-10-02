"""Small modeling utilities used by the discovery MVP."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from causal_emergence_discovery.schema import FeatureSchema, numeric_values, resolve_feature_schema


@dataclass(frozen=True)
class DesignEncoder:
    """Training-only variable kinds, imputation and categorical levels."""

    columns: tuple[str, ...]
    numeric_fills: dict[str, float]
    categories: dict[str, tuple[str, ...]]
    category_keys: dict[str, tuple[tuple[bool, str], ...]]
    references: dict[str, tuple[bool, str]]
    feature_names: tuple[str, ...]
    term_metadata: tuple[dict, ...]
    schema: FeatureSchema

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.transform_with_report(df)[0]

    def transform_with_report(self, df: pd.DataFrame) -> tuple[np.ndarray, dict[str, dict[str, int]]]:
        frames = []
        diagnostics: dict[str, dict[str, int]] = {}
        for column in self.columns:
            if column in self.numeric_fills:
                values, invalid = numeric_values(df[column])
                diagnostics[column] = {
                    "missing": int(df[column].isna().sum()),
                    "invalid_numeric": invalid,
                    "unknown_category": 0,
                }
                frames.append(values.fillna(self.numeric_fills[column]).to_numpy(dtype=float).reshape(-1, 1))
            else:
                raw = df[column]
                values = raw.astype("string")
                missing = raw.isna().to_numpy()
                categories = values.fillna("").astype(str).to_numpy()
                keys = [(True, "") if is_missing else (False, value) for is_missing, value in zip(missing, categories)]
                levels = self.category_keys[column]
                known = set(levels)
                unknown = sum((not is_missing) and ((False, value) not in known) for is_missing, value in keys)
                frames.append(np.column_stack([
                    np.fromiter((key == level for key in keys), dtype=float, count=len(keys))
                    for level in levels[1:]
                ]) if len(levels) > 1 else np.empty((len(df), 0)))
                diagnostics[column] = {
                    "missing": int(missing.sum()),
                    "invalid_numeric": 0,
                    "unknown_category": int(unknown),
                }
        matrix = np.hstack(frames) if frames else np.empty((len(df), 0))
        return matrix, diagnostics


def fit_design_encoder(
    df: pd.DataFrame,
    columns: list[str],
    column_specs=None,
    schema: FeatureSchema | None = None,
) -> DesignEncoder:
    """Fit medians and levels without consulting validation rows."""
    resolved_schema = schema or resolve_feature_schema(df, columns, column_specs)
    numeric_fills: dict[str, float] = {}
    categories: dict[str, tuple[str, ...]] = {}
    category_keys: dict[str, tuple[tuple[bool, str], ...]] = {}
    references: dict[str, tuple[bool, str]] = {}
    names: list[str] = []
    used_names: set[str] = set()
    metadata: list[dict] = []
    for column in dict.fromkeys(columns):
        values = df[column]
        if resolved_schema.kind(column) == "numeric":
            numeric, _ = numeric_values(values)
            numeric_fills[column] = float(numeric.median()) if numeric.notna().any() else 0.0
            names.append(column)
            used_names.add(column)
            metadata.append({"term": column, "kind": "numeric", "feature_name": column})
        else:
            strings = values.astype("string")
            distinct = {(False, value) for value in strings.dropna().astype(str).unique().tolist()}
            if strings.isna().any():
                distinct.add((True, ""))
            levels = tuple(sorted(distinct, key=lambda item: (item[0], item[1])))
            category_keys[column] = levels
            # Keep the public value-level list free of the null sentinel; the
            # richer category_keys separately preserves actual nulls without
            # colliding with a literal category named "__missing__".
            categories[column] = tuple(value for is_missing, value in levels if not is_missing)
            if levels:
                references[column] = levels[0]
                ref = {"is_missing": levels[0][0], "value": None if levels[0][0] else levels[0][1]}
                for is_missing, value in levels[1:]:
                    label = "<missing>" if is_missing else repr(value)
                    base_name = f"{column}={label}"
                    name = base_name
                    suffix = 2
                    while name in used_names:
                        name = f"{base_name}#{suffix}"
                        suffix += 1
                    used_names.add(name)
                    names.append(name)
                    metadata.append({"term": column, "kind": "categorical", "feature_name": name,
                                     "level": {"is_missing": is_missing, "value": None if is_missing else value},
                                     "reference_level": ref})
    return DesignEncoder(
        tuple(dict.fromkeys(columns)), numeric_fills, categories, category_keys,
        references, tuple(names), tuple(metadata), resolved_schema,
    )


@dataclass(frozen=True)
class LinearFit:
    feature_names: tuple[str, ...]
    coefficients: np.ndarray
    stderr: np.ndarray
    r2: float
    nobs: int
    encoder: DesignEncoder
    prediction_coefficients: np.ndarray = field(repr=False)
    estimable_mask: np.ndarray
    rank: int
    design_columns: int
    residual_df: int
    rank_tolerance: float
    condition_number: float | None
    aliased_terms: tuple[str, ...]
    term_metadata: tuple[dict, ...]


def design_matrix(
    df: pd.DataFrame,
    columns: list[str],
    column_specs=None,
    schema: FeatureSchema | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Fit and encode a standalone table; predictions use the fit's encoder."""
    encoder = fit_design_encoder(df, columns, column_specs, schema)
    return encoder.transform(df), encoder.feature_names


def fit_linear_model(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    column_specs=None,
    schema: FeatureSchema | None = None,
) -> LinearFit:
    """Fit OLS with SVD rank and coordinate-level estimability diagnostics."""
    # Resolve undeclared types on the entire fit-training partition before
    # filtering by target availability. Numeric fills and category levels are
    # then fitted only on rows with usable targets under this frozen contract.
    resolved_schema = schema or resolve_feature_schema(df, feature_columns, column_specs)
    clean = df.dropna(subset=[target_column]).reset_index(drop=True)
    y = pd.to_numeric(clean[target_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    clean = clean.loc[y.notna()].reset_index(drop=True)
    y_values = y.loc[y.notna()].to_numpy(dtype=float)
    if len(clean) < 2:
        raise ValueError("At least two non-missing target rows are required.")

    encoder = fit_design_encoder(clean, feature_columns, schema=resolved_schema)
    x = np.column_stack([np.ones(len(clean)), encoder.transform(clean)])
    names = ("__intercept__", *encoder.feature_names)
    metadata = (
        {"term": None, "kind": "internal_intercept", "feature_name": "__intercept__", "coefficient_index": 0},
        *({**item, "coefficient_index": index + 1} for index, item in enumerate(encoder.term_metadata)),
    )

    # Column scaling makes rank and condition diagnostics meaningful when
    # features use different units. Solve the scaled system directly by SVD.
    maxima = np.max(np.abs(x), axis=0)
    scales = np.asarray([
        0.0 if maximum == 0.0 else maximum * float(np.sqrt(np.sum((x[:, j] / maximum) ** 2)))
        for j, maximum in enumerate(maxima)
    ])
    if not np.isfinite(scales).all():
        raise ValueError("The regression design has a non-finite column scale.")
    scales[scales == 0] = 1.0
    z = x / scales
    try:
        u, singular_values, vt = np.linalg.svd(z, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("The regression design could not be decomposed safely.") from exc
    tolerance = (singular_values[0] * max(z.shape) * np.finfo(float).eps) if singular_values.size else 0.0
    rank = int(np.sum(singular_values > tolerance))
    gamma = np.zeros(z.shape[1], dtype=float)
    if rank:
        gamma = vt[:rank, :].T @ ((u[:, :rank].T @ y_values) / singular_values[:rank])
    beta = gamma / scales
    if not np.isfinite(beta).all():
        raise ValueError("The fitted regression coefficients are not finite; rescale or simplify the design.")
    fitted = x @ beta
    residuals = y_values - fitted
    total = y_values - _stable_mean(y_values)
    total_norm = _scaled_norm_parts(total)
    resid_norm = _scaled_norm_parts(residuals)
    if total_norm[0] == 0.0:
        r2 = 0.0
    else:
        ratio = (resid_norm[0] / total_norm[0]) * (resid_norm[1] / total_norm[1])
        if not np.isfinite(ratio) or ratio > np.sqrt(np.finfo(float).max):
            raise ValueError("Regression R2 is not finite for this target scale.")
        r2 = float(1.0 - ratio * ratio)
    dof = len(y_values) - rank
    rowspace_projection = np.sum(vt[:rank, :] ** 2, axis=0) if rank else np.zeros(x.shape[1])
    estimable = (1.0 - rowspace_projection) <= 1e-10
    stderr = np.full(x.shape[1], np.nan, dtype=float)
    if dof > 0 and rank:
        # Scaled least-squares variance diagonal from retained singular vectors.
        variance_factors = (vt[:rank, :] / scales.reshape(1, -1)) ** 2
        covariance_diag = np.sum(variance_factors / (singular_values[:rank, None] ** 2), axis=0)
        sigma = _stable_norm(residuals) / np.sqrt(dof)
        stderr = np.sqrt(covariance_diag) * sigma
    stderr[~estimable | ~np.isfinite(stderr)] = np.nan
    public_beta = beta.copy()
    public_beta[~estimable] = np.nan
    aliased_terms = tuple(names[index] for index in np.flatnonzero(~estimable))
    condition = None
    if rank and rank == x.shape[1]:
        condition = float(singular_values[0] / singular_values[rank - 1])
    return LinearFit(
        feature_names=names,
        coefficients=public_beta,
        stderr=stderr,
        r2=float(r2),
        nobs=len(clean),
        encoder=encoder,
        prediction_coefficients=beta,
        estimable_mask=estimable,
        rank=rank,
        design_columns=x.shape[1],
        residual_df=dof,
        rank_tolerance=float(tolerance),
        condition_number=condition,
        aliased_terms=aliased_terms,
        term_metadata=tuple(metadata),
    )


def _scaled_norm_parts(values: np.ndarray) -> tuple[float, float]:
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    if scale == 0.0:
        return 0.0, 0.0
    if not np.isfinite(scale):
        raise ValueError("Regression values contain non-finite values.")
    return scale, float(np.sqrt(np.sum((values / scale) ** 2)))


def _stable_mean(values: np.ndarray) -> float:
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    if scale == 0.0:
        return 0.0
    result = scale * float(np.mean(values / scale))
    if not np.isfinite(result):
        raise ValueError("The target mean is not finite at this scale.")
    return result


def _stable_norm(values: np.ndarray) -> float:
    scale, unit_norm = _scaled_norm_parts(values)
    result = scale * unit_norm
    if not np.isfinite(result):
        raise ValueError("Regression uncertainty is not finite for this target scale.")
    return result


def cross_validated_r2(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    *,
    folds: int = 5,
    splits=None,
    column_specs=None,
    schema: FeatureSchema | None = None,
) -> tuple[float, list[float]]:
    """Evaluate declared splits, or legacy row interpolation when omitted.

    Omitted splits answer a within-observed-rows interpolation question; they
    do not estimate performance for new entities or future periods. Discovery
    always supplies an auditable ValidationPlan's development splits.
    """
    clean = df.reset_index(drop=True)
    if splits is None:
        if folds < 2:
            raise ValueError("At least two folds are required.")
        indices = np.arange(len(clean))
        splits = [(indices[indices % folds != fold], indices[indices % folds == fold]) for fold in range(folds)]
    scores = []
    for split in splits:
        train_indices, test_indices = (split.train_indices, split.test_indices) if hasattr(split, "train_indices") else split
        train = clean.iloc[list(train_indices)].reset_index(drop=True)
        test = clean.iloc[list(test_indices)].reset_index(drop=True)
        fit = fit_linear_model(train, feature_columns, target_column, column_specs, schema)
        score = predict_r2(fit, test, feature_columns, target_column)
        scores.append(score)
    if not scores:
        raise ValueError("No validation splits were supplied.")
    return float(np.mean(scores)), [float(score) for score in scores]


def predict_linear_model(fit: LinearFit, df: pd.DataFrame) -> np.ndarray:
    """Predict with frozen training preprocessing; unknown categories are zero."""
    return predict_linear_model_with_report(fit, df)[0]


def predict_linear_model_with_report(
    fit: LinearFit,
    df: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, dict[str, int]]]:
    """Predict and report missing, malformed numeric, and unseen category counts."""
    transformed, diagnostics = fit.encoder.transform_with_report(df)
    x = np.column_stack([np.ones(len(df)), transformed])
    return x @ fit.prediction_coefficients, diagnostics


def predict_r2(
    fit: LinearFit,
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> float:
    clean = df.dropna(subset=[target_column]).reset_index(drop=True)
    y = pd.to_numeric(clean[target_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    clean = clean.loc[y.notna()].reset_index(drop=True)
    y_values = y.loc[y.notna()].to_numpy(dtype=float)
    if len(clean) < 2:
        raise ValueError("Validation requires at least two finite target rows.")
    if tuple(dict.fromkeys(feature_columns)) != fit.encoder.columns:
        raise ValueError("Prediction features must match the fitted design columns.")
    predictions = predict_linear_model(fit, clean)
    centered = y_values - _stable_mean(y_values)
    total_norm = _scaled_norm_parts(centered)
    if total_norm[0] == 0.0:
        raise ValueError("Validation R2 is undefined for a constant target.")
    residuals = y_values - predictions
    resid_norm = _scaled_norm_parts(residuals)
    ratio = (resid_norm[0] / total_norm[0]) * (resid_norm[1] / total_norm[1])
    if not np.isfinite(ratio) or ratio > np.sqrt(np.finfo(float).max):
        raise ValueError("Validation R2 is not finite for this target scale.")
    return float(1.0 - ratio * ratio)


def treatment_effect_records(
    df: pd.DataFrame,
    interventions: list[str],
    target_column: str,
    adjustment_columns: list[str],
    *,
    estimand: str = "joint_conditional",
    column_specs=None,
    schema: FeatureSchema | None = None,
) -> list[dict]:
    """Return term-aware intervention records under a declared OLS estimand.

    ``joint_conditional`` includes every declared intervention in every fit.
    ``marginal`` fits one intervention at a time with the declared adjustment
    columns. Uncertainty is ordinary IID-row uncertainty; it is not panel-aware.
    """
    if not isinstance(estimand, str) or estimand not in {"joint_conditional", "marginal"}:
        raise ValueError("estimand must be 'joint_conditional' or 'marginal'.")
    interventions = list(dict.fromkeys(interventions))
    adjustment_columns = list(dict.fromkeys(adjustment_columns))
    if set(interventions) & set(adjustment_columns):
        raise ValueError("Adjustment columns cannot overlap declared interventions.")
    estimates: list[dict] = []
    fitted_models: dict[tuple[str, ...], LinearFit | Exception] = {}
    for intervention in interventions:
        treatment_columns = interventions if estimand == "joint_conditional" else [intervention]
        columns = [*treatment_columns, *[name for name in adjustment_columns if name not in treatment_columns]]
        record = {
            "intervention": intervention,
            "treatment": intervention,
            "target": target_column,
            "estimand": estimand,
            "coefficient": None,
            "stderr": None,
            "stderr_iid": None,
            "t_stat": None,
            "terms": [],
            "nobs": 0,
            "model_r2": None,
            "estimable": False,
            "status": "fit_error",
            "uncertainty_scope": "iid_rows",
            "rank": None,
            "design_rank": None,
            "design_columns": None,
            "residual_df": None,
            "rank_tolerance": None,
            "condition_number": None,
            "aliased_terms": [],
            "uncertainty_status": "unavailable",
        }
        try:
            fit_key = tuple(columns)
            if fit_key not in fitted_models:
                fitted_models[fit_key] = fit_linear_model(
                    df, columns, target_column, column_specs=column_specs, schema=schema
                )
            fit = fitted_models[fit_key]
            if isinstance(fit, Exception):
                raise fit
        except (ValueError, KeyError) as exc:
            fitted_models[tuple(columns)] = exc
            record["error"] = str(exc)
            estimates.append(record)
            continue
        record.update({
            "nobs": fit.nobs,
            "model_r2": fit.r2,
            "rank": fit.rank,
            "design_rank": fit.rank,
            "design_columns": fit.design_columns,
            "residual_df": fit.residual_df,
            "rank_tolerance": fit.rank_tolerance,
            "condition_number": fit.condition_number,
            "aliased_terms": list(fit.aliased_terms),
        })
        treatment_metadata = [item for item in fit.term_metadata if item["term"] == intervention]
        treatment_indices = [item["coefficient_index"] for item in treatment_metadata]
        if len(treatment_indices) == 1 and treatment_metadata[0]["kind"] == "numeric":
            index = treatment_indices[0]
            estimable = bool(fit.estimable_mask[index])
            coefficient = _finite_or_none(fit.coefficients[index]) if estimable else None
            stderr = _finite_or_none(fit.stderr[index]) if estimable else None
            record.update({
                "coefficient": coefficient,
                "stderr": stderr,
                "stderr_iid": stderr,
                "t_stat": _safe_t_stat(coefficient, stderr),
                "estimable": estimable,
                "status": "estimable" if estimable else "non_estimable",
                "uncertainty_status": _uncertainty_status(fit, index),
                "terms": [{
                    "term": intervention, "feature_name": fit.feature_names[index], "kind": "numeric",
                    "coefficient": coefficient, "stderr": stderr, "estimable": estimable,
                    "status": "estimable" if estimable else "non_estimable",
                    "uncertainty_status": _uncertainty_status(fit, index),
                }],
            })
        elif treatment_metadata:
            terms = []
            for item, index in zip(treatment_metadata, treatment_indices):
                estimable = bool(fit.estimable_mask[index])
                coefficient = _finite_or_none(fit.coefficients[index]) if estimable else None
                stderr = _finite_or_none(fit.stderr[index]) if estimable else None
                terms.append({
                    "term": intervention,
                    "feature_name": item["feature_name"],
                    "kind": "categorical",
                    "level": item["level"],
                    "reference_level": item["reference_level"],
                    "coefficient": coefficient,
                    "stderr": stderr,
                    "estimable": estimable,
                    "status": "estimable" if estimable else "non_estimable",
                    "uncertainty_status": _uncertainty_status(fit, index),
                })
            record.update({"terms": terms, "estimable": all(term["estimable"] for term in terms), "status": "categorical"})
            record["note"] = "intervention was not represented as a single numeric term; categorical levels are reported as reference contrasts"
        else:
            record["status"] = "categorical"
            record["note"] = "intervention was not represented as a single numeric term; it has no non-reference estimable design term"
        estimates.append(record)
    return estimates


def _finite_or_none(value) -> float | None:
    result = float(value)
    return result if np.isfinite(result) else None


def _uncertainty_status(fit: LinearFit, index: int) -> str:
    if not fit.estimable_mask[index]:
        return "aliased_term"
    if fit.residual_df <= 0:
        return "residual_df_zero"
    if not np.isfinite(fit.stderr[index]):
        return "nonfinite_uncertainty"
    return "iid_available"


def _safe_t_stat(coefficient: float | None, stderr: float | None) -> float | None:
    if coefficient is None or stderr is None or stderr == 0.0:
        return None
    value = coefficient / stderr
    return float(value) if np.isfinite(value) else None


def effect_estimates(
    df: pd.DataFrame,
    interventions: list[str],
    target_column: str,
    adjustment_columns: list[str],
    *,
    estimand: str = "joint_conditional",
    column_specs=None,
    schema: FeatureSchema | None = None,
) -> list[dict]:
    """Compatibility wrapper routed through term-aware treatment records."""
    records = treatment_effect_records(
        df, interventions, target_column, adjustment_columns, estimand=estimand,
        column_specs=column_specs, schema=schema,
    )
    estimates = []
    for record in records:
        estimates.append({
            **record,
            "intervention": record["treatment"],
        })
    return sorted(estimates, key=lambda item: abs(float(item["t_stat"] or 0.0)), reverse=True)
