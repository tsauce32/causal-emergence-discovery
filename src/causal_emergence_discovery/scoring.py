"""Scoring functions for candidate emergent macro variables."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from causal_emergence_discovery.macro import MacroAssignment, apply_macro, refit_macro
from causal_emergence_discovery.models import fit_linear_model, predict_r2


MACRO_COLUMN = "__ced_macro_state"


@dataclass(frozen=True)
class MacroScore:
    macro_name: str
    outcome: str
    target_column: str
    score: float
    emergence_delta: float
    macro_r2: float
    micro_r2: float
    specificity: float
    stability: float
    compression: float
    state_count: int
    row_count: int
    fold_scores: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "macro_name": self.macro_name,
            "outcome": self.outcome,
            "target_column": self.target_column,
            "score": self.score,
            "emergence_delta": self.emergence_delta,
            "macro_r2": self.macro_r2,
            "micro_r2": self.micro_r2,
            "specificity": self.specificity,
            "stability": self.stability,
            "compression": self.compression,
            "state_count": self.state_count,
            "row_count": self.row_count,
            "fold_scores": list(self.fold_scores),
        }


def score_macro(
    df: pd.DataFrame,
    macro: MacroAssignment,
    *,
    outcome: str,
    target_column: str,
    micro_feature_columns: list[str],
    intervention_columns: list[str],
    folds: int = 5,
    validation_splits=None,
) -> MacroScore:
    """Rank on development folds with training-only encoders and specificity.

    With no split plan this standalone helper uses legacy row interpolation.
    Only run_discovery supplies a reserved outer holdout; this score is for
    selection and must not be reported as selected-model generalization.
    """
    if validation_splits is None:
        if folds < 2:
            raise ValueError("At least two folds are required.")
        indices = np.arange(len(df))
        validation_splits = [(indices[indices % folds != fold], indices[indices % folds == fold]) for fold in range(folds)]
    macro_features = [*intervention_columns, MACRO_COLUMN]
    micro_features = _ordered_unique([*intervention_columns, *micro_feature_columns])
    fold_scores, micro_scores, specificity_scores = [], [], []
    for split in validation_splits:
        train_indices, test_indices = (split.train_indices, split.test_indices) if hasattr(split, "train_indices") else split
        train = df.iloc[list(train_indices)].reset_index(drop=True).copy()
        test = df.iloc[list(test_indices)].reset_index(drop=True).copy()
        fitted_macro = refit_macro(macro, train)
        train[MACRO_COLUMN] = fitted_macro.labels.astype(str).to_numpy()
        test[MACRO_COLUMN] = apply_macro(fitted_macro, test).labels.astype(str).to_numpy()
        macro_fit = fit_linear_model(train, macro_features, target_column)
        micro_fit = fit_linear_model(train, micro_features, target_column)
        specificity_fit = fit_linear_model(train, [MACRO_COLUMN], target_column)
        fold_scores.append(predict_r2(macro_fit, test, macro_features, target_column))
        micro_scores.append(predict_r2(micro_fit, test, micro_features, target_column))
        specificity_scores.append(max(0.0, predict_r2(specificity_fit, test, [MACRO_COLUMN], target_column)))
    if not fold_scores:
        raise ValueError("No development validation folds were supplied.")
    macro_r2 = float(np.mean(fold_scores))
    micro_r2 = float(np.mean(micro_scores))
    specificity = float(np.mean(specificity_scores))
    stability = fold_stability(fold_scores)
    compression = compression_score(macro.state_count, len(df))

    positive_macro_r2 = max(macro_r2, 0.0)
    positive_micro_r2 = max(micro_r2, 0.0)
    score = (
        0.45 * positive_macro_r2
        + 0.25 * specificity
        + 0.20 * stability
        + 0.10 * compression
    )
    raw_reference_score = 0.45 * positive_micro_r2
    emergence_delta = score - raw_reference_score
    return MacroScore(
        macro_name=macro.name,
        outcome=outcome,
        target_column=target_column,
        score=float(score),
        emergence_delta=float(emergence_delta),
        macro_r2=float(macro_r2),
        micro_r2=float(micro_r2),
        specificity=float(specificity),
        stability=float(stability),
        compression=float(compression),
        state_count=macro.state_count,
        row_count=len(df),
        fold_scores=tuple(float(value) for value in fold_scores),
    )


def outcome_specificity(df: pd.DataFrame, macro_column: str, target_column: str) -> float:
    """Return weighted between-macro outcome variance divided by total variance."""
    clean = df[[macro_column, target_column]].dropna()
    if len(clean) < 2:
        return 0.0
    target = pd.to_numeric(clean[target_column], errors="coerce")
    clean = clean.loc[target.notna()].copy()
    clean[target_column] = target.loc[target.notna()]
    total_var = float(clean[target_column].var(ddof=0))
    if total_var == 0.0:
        return 0.0
    overall = float(clean[target_column].mean())
    grouped = clean.groupby(macro_column)[target_column]
    between = 0.0
    for _, values in grouped:
        weight = len(values) / len(clean)
        between += weight * (float(values.mean()) - overall) ** 2
    return float(max(0.0, min(1.0, between / total_var)))


def fold_stability(scores: list[float]) -> float:
    if not scores:
        return 0.0
    clipped = np.asarray([max(score, 0.0) for score in scores], dtype=float)
    return float(1.0 / (1.0 + float(clipped.std(ddof=0))))


def compression_score(state_count: int, row_count: int) -> float:
    if state_count <= 1:
        return 0.0
    denominator = math.log2(max(row_count, 2))
    value = 1.0 - math.log2(state_count) / denominator
    return float(max(0.0, min(1.0, value)))


def _ordered_unique(columns: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        ordered.append(column)
    return ordered
