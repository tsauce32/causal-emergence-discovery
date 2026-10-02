"""Small modeling utilities used by the discovery MVP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DesignEncoder:
    """Training-only imputation and categorical levels for a linear design."""

    columns: tuple[str, ...]
    numeric_fills: dict[str, float]
    categories: dict[str, tuple[str, ...]]
    feature_names: tuple[str, ...]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        frames = []
        for column in self.columns:
            if column in self.numeric_fills:
                values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
                frames.append(values.fillna(self.numeric_fills[column]).to_numpy(dtype=float).reshape(-1, 1))
            else:
                values = df[column].astype("string").fillna("__missing__")
                frames.append(np.column_stack([(values == level).to_numpy(dtype=float) for level in self.categories[column]]))
        return np.hstack(frames) if frames else np.empty((len(df), 0))


def fit_design_encoder(df: pd.DataFrame, columns: list[str]) -> DesignEncoder:
    """Fit medians and levels without consulting validation rows."""
    numeric_fills: dict[str, float] = {}
    categories: dict[str, tuple[str, ...]] = {}
    names: list[str] = []
    for column in dict.fromkeys(columns):
        values = df[column]
        if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
            numeric_fills[column] = float(numeric.median()) if numeric.notna().any() else 0.0
            names.append(column)
        else:
            levels = tuple(sorted(values.astype("string").fillna("__missing__").unique().tolist()))
            categories[column] = levels
            names.extend(f"{column}={level}" for level in levels)
    return DesignEncoder(tuple(dict.fromkeys(columns)), numeric_fills, categories, tuple(names))


@dataclass(frozen=True)
class LinearFit:
    feature_names: tuple[str, ...]
    coefficients: np.ndarray
    stderr: np.ndarray
    r2: float
    nobs: int
    encoder: DesignEncoder


def design_matrix(df: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, tuple[str, ...]]:
    """Fit and encode a standalone table; predictions use the fit's encoder."""
    encoder = fit_design_encoder(df, columns)
    return encoder.transform(df), encoder.feature_names


def fit_linear_model(df: pd.DataFrame, feature_columns: list[str], target_column: str) -> LinearFit:
    """Fit ordinary least squares with a pseudoinverse for singular designs."""
    clean = df.dropna(subset=[target_column]).reset_index(drop=True)
    y = pd.to_numeric(clean[target_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    clean = clean.loc[y.notna()].reset_index(drop=True)
    y_values = y.loc[y.notna()].to_numpy(dtype=float)
    if len(clean) < 2:
        raise ValueError("At least two non-missing target rows are required.")

    encoder = fit_design_encoder(clean, feature_columns)
    x, feature_names = encoder.transform(clean), encoder.feature_names
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
        encoder=encoder,
    )


def cross_validated_r2(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    *,
    folds: int = 5,
    splits=None,
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
        fit = fit_linear_model(train, feature_columns, target_column)
        score = predict_r2(fit, test, feature_columns, target_column)
        scores.append(score)
    if not scores:
        raise ValueError("No validation splits were supplied.")
    return float(np.mean(scores)), [float(score) for score in scores]


def predict_linear_model(fit: LinearFit, df: pd.DataFrame) -> np.ndarray:
    """Predict with frozen training preprocessing; unknown categories are zero."""
    x = np.column_stack([np.ones(len(df)), fit.encoder.transform(df)])
    return x @ fit.coefficients


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
    centered = y_values - y_values.mean()
    ss_total = float(centered @ centered)
    if ss_total == 0.0:
        raise ValueError("Validation R2 is undefined for a constant target.")
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
