"""Small modeling utilities used by the discovery MVP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LinearFit:
    feature_names: tuple[str, ...]
    coefficients: np.ndarray
    stderr: np.ndarray
    r2: float
    nobs: int


def design_matrix(df: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, tuple[str, ...]]:
    """Encode mixed tabular columns as a numeric design matrix."""
    if not columns:
        return np.empty((len(df), 0)), ()
    frames = []
    feature_names: list[str] = []
    for column in columns:
        values = df[column]
        if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            numeric = pd.to_numeric(values, errors="coerce").astype(float)
            fill = float(numeric.median()) if numeric.notna().any() else 0.0
            frames.append(numeric.fillna(fill).to_numpy().reshape(-1, 1))
            feature_names.append(column)
        else:
            dummies = pd.get_dummies(values.astype("string").fillna("__missing__"), prefix=column)
            frames.append(dummies.to_numpy(dtype=float))
            feature_names.extend(str(name) for name in dummies.columns)
    return np.hstack(frames), tuple(feature_names)


def fit_linear_model(df: pd.DataFrame, feature_columns: list[str], target_column: str) -> LinearFit:
    """Fit an ordinary least squares model with a small ridge fallback."""
    clean = df.dropna(subset=[target_column]).reset_index(drop=True)
    y = pd.to_numeric(clean[target_column], errors="coerce")
    clean = clean.loc[y.notna()].reset_index(drop=True)
    y_values = y.loc[y.notna()].to_numpy(dtype=float)
    if len(clean) < 2:
        raise ValueError("At least two non-missing target rows are required.")

    x, feature_names = design_matrix(clean, feature_columns)
    x = np.column_stack([np.ones(len(clean)), x])
    names = ("intercept", *feature_names)
    xtx = x.T @ x
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ x.T @ y_values
    fitted = x @ beta
    residuals = y_values - fitted
    total = y_values - y_values.mean()
    ss_total = float(total @ total)
    ss_resid = float(residuals @ residuals)
    r2 = 0.0 if ss_total == 0.0 else 1.0 - ss_resid / ss_total
    dof = max(len(y_values) - x.shape[1], 1)
    sigma2 = ss_resid / dof
    stderr = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    return LinearFit(
        feature_names=names,
        coefficients=beta,
        stderr=stderr,
        r2=float(r2),
        nobs=len(clean),
    )


def cross_validated_r2(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    *,
    folds: int = 5,
) -> tuple[float, list[float]]:
    """Compute deterministic K-fold R2 for a linear model."""
    clean = df.dropna(subset=[target_column]).reset_index(drop=True)
    if len(clean) < 4:
        return 0.0, [0.0]
    fold_count = max(2, min(folds, len(clean)))
    scores = []
    indices = np.arange(len(clean))
    for fold in range(fold_count):
        test_mask = indices % fold_count == fold
        train = clean.loc[~test_mask].reset_index(drop=True)
        test = clean.loc[test_mask].reset_index(drop=True)
        if len(test) < 2 or len(train) < 2:
            continue
        fit = fit_linear_model(train, feature_columns, target_column)
        score = predict_r2(fit, test, feature_columns, target_column)
        scores.append(score)
    if not scores:
        return 0.0, [0.0]
    return float(np.mean(scores)), [float(score) for score in scores]


def predict_r2(
    fit: LinearFit,
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> float:
    clean = df.dropna(subset=[target_column]).reset_index(drop=True)
    y = pd.to_numeric(clean[target_column], errors="coerce")
    clean = clean.loc[y.notna()].reset_index(drop=True)
    y_values = y.loc[y.notna()].to_numpy(dtype=float)
    if len(clean) < 2:
        return 0.0
    x_raw, feature_names = design_matrix(clean, feature_columns)
    aligned = np.zeros((len(clean), len(fit.feature_names) - 1))
    target_names = list(fit.feature_names[1:])
    for source_index, name in enumerate(feature_names):
        if name in target_names:
            aligned[:, target_names.index(name)] = x_raw[:, source_index]
    x = np.column_stack([np.ones(len(clean)), aligned])
    predictions = x @ fit.coefficients
    centered = y_values - y_values.mean()
    ss_total = float(centered @ centered)
    if ss_total == 0.0:
        return 0.0
    residuals = y_values - predictions
    return float(1.0 - float(residuals @ residuals) / ss_total)


def effect_estimates(
    df: pd.DataFrame,
    interventions: list[str],
    target_column: str,
    adjustment_columns: list[str],
) -> list[dict[str, float | str | int | None]]:
    """Estimate intervention coefficients under a linear adjustment model."""
    estimates: list[dict[str, float | str | int | None]] = []
    for intervention in interventions:
        columns = [intervention, *[name for name in adjustment_columns if name != intervention]]
        try:
            fit = fit_linear_model(df, columns, target_column)
        except ValueError:
            continue
        if intervention not in fit.feature_names:
            estimates.append(
                {
                    "intervention": intervention,
                    "target": target_column,
                    "coefficient": None,
                    "stderr": None,
                    "t_stat": None,
                    "nobs": fit.nobs,
                    "note": "intervention was categorical or expanded into dummy columns",
                }
            )
            continue
        index = fit.feature_names.index(intervention)
        coef = float(fit.coefficients[index])
        stderr = float(fit.stderr[index])
        t_stat = None if stderr == 0.0 else coef / stderr
        estimates.append(
            {
                "intervention": intervention,
                "target": target_column,
                "coefficient": coef,
                "stderr": stderr,
                "t_stat": None if t_stat is None else float(t_stat),
                "nobs": fit.nobs,
                "model_r2": fit.r2,
            }
        )
    return sorted(
        estimates,
        key=lambda item: abs(float(item["t_stat"] or 0.0)),
        reverse=True,
    )
