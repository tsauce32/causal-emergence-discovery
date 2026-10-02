"""Independent NumPy baselines for the selection-aware Discovery benchmark.

All encoders in this module are fitted on the training rows supplied to
``fit_predict``. No Discovery regression/scoring implementation is imported.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass
class FittedOLS:
    columns: tuple[str, ...]
    numeric: dict[str, float]
    categories: dict[str, tuple[str, ...]]
    category_keys: dict[str, tuple[tuple[bool, str], ...]]
    mean: np.ndarray
    scale: np.ndarray
    numeric_indices: tuple[int, ...]
    terms: tuple[tuple[int, ...], ...]
    coefficients: np.ndarray

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        base = _base_matrix(frame, self.columns, self.numeric, self.categories, self.category_keys)
        if base.shape[1]:
            base = (base - self.mean) / self.scale
        blocks = [np.ones((len(frame), 1)), base]
        if self.terms:
            blocks.extend(np.prod(base[:, term], axis=1, keepdims=True) for term in self.terms)
        return np.column_stack(blocks)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.transform(frame) @ self.coefficients


def fit_ols(train: pd.DataFrame, features: Iterable[str], target: str, *,
            polynomial_degree: int = 1) -> FittedOLS:
    """Fit intercept OLS with train-only encoders and bounded numeric terms."""
    columns = tuple(dict.fromkeys(features))
    clean, y = _finite_targets(train, target)
    if len(y) < 2:
        raise ValueError("baseline needs at least two finite training targets")
    numeric: dict[str, float] = {}
    categories: dict[str, tuple[str, ...]] = {}
    category_keys: dict[str, tuple[tuple[bool, str], ...]] = {}
    for col in columns:
        vals = clean[col]
        if pd.api.types.is_numeric_dtype(vals) or pd.api.types.is_bool_dtype(vals):
            a = pd.to_numeric(vals, errors="coerce").replace([np.inf, -np.inf], np.nan)
            numeric[col] = float(a.median()) if a.notna().any() else 0.0
        else:
            string_values = vals.astype("string")
            distinct = {(False, value) for value in string_values.dropna().astype(str).unique().tolist()}
            if string_values.isna().any():
                distinct.add((True, ""))
            keys = tuple(sorted(distinct, key=lambda item: (item[0], item[1])))
            category_keys[col] = keys
            categories[col] = tuple(value for is_missing, value in keys if not is_missing)
    raw = _base_matrix(clean, columns, numeric, categories, category_keys)
    numeric_indices = []
    offset = 0
    for col in columns:
        if col in numeric:
            numeric_indices.append(offset)
            offset += 1
        else:
            offset += max(0, len(category_keys.get(col, ())) - 1)
    mean = raw.mean(axis=0) if raw.shape[1] and polynomial_degree > 1 else np.zeros(raw.shape[1])
    scale = raw.std(axis=0) if raw.shape[1] and polynomial_degree > 1 else np.ones(raw.shape[1])
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    base = (raw - mean) / scale if raw.shape[1] else raw
    # Categorical indicators remain additive; quadratic terms use at most three
    # numeric predictors, keeping the flexible baseline prespecified and bounded.
    poly_indices = numeric_indices[:3]
    terms = tuple(combinations_with_replacement(poly_indices, degree)
                  for degree in range(2, polynomial_degree + 1))
    terms = tuple(term for degree_terms in terms for term in degree_terms)
    blocks = [np.ones((len(base), 1)), base]
    if terms:
        blocks.extend(np.prod(base[:, term], axis=1, keepdims=True) for term in terms)
    design = np.column_stack(blocks)
    # Match the package's documented pseudoinverse OLS contract while retaining
    # this independent encoder/design implementation. This matters for repeated
    # entity-category indicators and other rank-deficient designs.
    beta = _scaled_svd_ols(design, y)
    return FittedOLS(columns, numeric, categories, category_keys, mean, scale,
                     tuple(numeric_indices), terms, beta)


def evaluate_r2(model: FittedOLS, frame: pd.DataFrame, target: str) -> tuple[float, np.ndarray, np.ndarray]:
    clean, y = _finite_targets(frame, target)
    if len(y) < 2:
        raise ValueError("baseline needs at least two finite test targets")
    pred = model.predict(clean)
    denom = float(np.sum((y - y.mean()) ** 2))
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("test target variance is zero or nonfinite")
    residual = float(np.sum((y - pred) ** 2))
    score = 1.0 - residual / denom
    if not np.isfinite(score) or not np.isfinite(pred).all():
        raise ValueError("baseline produced a nonfinite score or prediction")
    return float(score), pred, y


def fit_predict_r2(train: pd.DataFrame, test: pd.DataFrame, features: Iterable[str], target: str,
                   *, polynomial_degree: int = 1):
    model = fit_ols(train, features, target, polynomial_degree=polynomial_degree)
    return evaluate_r2(model, test, target)


def fit_state_model(train: pd.DataFrame, test: pd.DataFrame, train_labels: Iterable[Any],
                    test_labels: Iterable[Any], target: str, *, extra_features: Iterable[str] = ()):
    """Independent reference-coded state-regression comparator."""
    train = train.copy()
    test = test.copy()
    train["__benchmark_state"] = pd.Series(list(train_labels), index=train.index).astype("string")
    test["__benchmark_state"] = pd.Series(list(test_labels), index=test.index).astype("string")
    return fit_predict_r2(train, test, [*extra_features, "__benchmark_state"], target)


def fit_matched_micro(train: pd.DataFrame, test: pd.DataFrame, train_labels: Iterable[Any],
                      test_labels: Iterable[Any], target: str, *, extra_features: Iterable[str] = ()):
    """Independent matched-basis check using explicit frozen state-indicator columns.

    The feature basis is the selected fitted recipe's output. Equality with the
    macro is expected because both encode the same information; this verifies
    that the apparent gain against a restricted raw-micro class is a model-class
    difference, not a gain from extra information.
    """
    train = train.reset_index(drop=True).copy()
    test = test.reset_index(drop=True).copy()
    a, b = pd.Series(list(train_labels)).astype("string"), pd.Series(list(test_labels)).astype("string")
    states = sorted(a.unique().tolist())
    # Match training-reference contrasts: reference is the first sorted state.
    for i, state in enumerate(states[1:]):
        name = f"__matched_recipe_state_{i}"
        train[name] = (a == state).to_numpy(dtype=float)
        test[name] = (b == state).to_numpy(dtype=float)
    return fit_predict_r2(train, test, [*extra_features,
        *[f"__matched_recipe_state_{i}" for i in range(max(0, len(states) - 1))]], target)


def confounding_ols_diagnostics(frame: pd.DataFrame, *, treatment: str, outcome: str,
                                confounder: str, macro: str | None = None) -> dict[str, Any]:
    """Descriptive independent OLS coefficients: marginal/raw-U/coarse-macro."""
    rows = {}
    for name, columns in (("marginal", [treatment]),
                          ("adjusted_raw_confounder", [treatment, confounder]),
                          ("adjusted_coarse_macro", [treatment, macro] if macro else [treatment])):
        clean, y = _finite_targets(frame, outcome)
        design = _base_matrix(clean, columns, *_fit_encoders(clean, columns))
        design = np.column_stack([np.ones(len(y)), design])
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        treatment_index = 1
        rows[name] = {"coefficient": _finite_or_none(beta[treatment_index]), "nobs": int(len(y))}
    return rows


def _fit_encoders(frame, columns):
    numeric, categories = {}, {}
    for col in columns:
        if col is None:
            continue
        vals = frame[col]
        if pd.api.types.is_numeric_dtype(vals) or pd.api.types.is_bool_dtype(vals):
            a = pd.to_numeric(vals, errors="coerce").replace([np.inf, -np.inf], np.nan)
            numeric[col] = float(a.median()) if a.notna().any() else 0.0
        else:
            categories[col] = tuple(sorted(vals.astype("string").fillna("__missing__").unique().tolist()))
    return numeric, categories


def _base_matrix(frame, columns, numeric, categories, category_keys=None):
    pieces = []
    for col in columns:
        if col is None:
            continue
        if col in numeric:
            vals = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            pieces.append(vals.fillna(numeric[col]).to_numpy(dtype=float)[:, None])
        else:
            raw = frame[col]
            string_values = raw.astype("string")
            if category_keys is not None and col in category_keys:
                keys = [(True, "") if missing else (False, value)
                        for missing, value in zip(raw.isna().to_numpy(), string_values.fillna("").astype(str).to_numpy())]
                levels = category_keys[col]
                pieces.extend(np.fromiter((key == level for key in keys), dtype=float, count=len(keys))[:, None]
                              for level in levels[1:])
            else:
                values = string_values.fillna("__missing__")
                levels = categories.get(col, ())
                pieces.extend((values == level).to_numpy(dtype=float)[:, None] for level in levels[1:])
    return np.column_stack(pieces) if pieces else np.empty((len(frame), 0))


def _scaled_svd_ols(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Independent stable NumPy OLS matching the package's scaled-SVD contract."""
    maxima = np.max(np.abs(design), axis=0)
    scales = np.asarray([
        0.0 if maximum == 0.0 else maximum * float(np.sqrt(np.sum((design[:, j] / maximum) ** 2)))
        for j, maximum in enumerate(maxima)
    ])
    scales[scales == 0.0] = 1.0
    z = design / scales
    u, singular, vt = np.linalg.svd(z, full_matrices=False)
    tolerance = singular[0] * max(z.shape) * np.finfo(float).eps if singular.size else 0.0
    rank = int(np.sum(singular > tolerance))
    gamma = vt[:rank].T @ ((u[:, :rank].T @ y) / singular[:rank]) if rank else np.zeros(z.shape[1])
    beta = gamma / scales
    if not np.isfinite(beta).all():
        raise ValueError("baseline OLS produced nonfinite coefficients")
    return beta


def _finite_targets(frame, target):
    y = pd.to_numeric(frame[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = y.notna().to_numpy()
    return frame.loc[mask].reset_index(drop=True), y.loc[mask].to_numpy(dtype=float)


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None
