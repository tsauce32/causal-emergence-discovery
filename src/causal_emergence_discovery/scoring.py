"""Descriptive ranking preferences and paired predictive comparisons.

Neither quantity is a causal-emergence metric or a detection rule.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from causal_emergence_discovery.macro import MacroAssignment, apply_macro, refit_macro
from causal_emergence_discovery.models import fit_linear_model, predict_r2

MACRO_COLUMN = "__ced_macro_state"
SCORE_SCHEMA_VERSION = 2
DEFAULT_PREDICTIVE_SCOPE = "row_modulo_refit_cv"
_VALIDATION_SPECIFICITY = "mean_positive_clipped_validation_r2_of_training_fitted_state_means"
_DESCRIPTIVE_SPECIFICITY = "descriptive_between_state_outcome_variance_share"


@dataclass(frozen=True)
class PairedPredictiveScores:
    """Caller-declared paired macro/micro R² values.

    The caller is responsible for identical target rows and for accurately
    describing the evaluation. Scope text and optional IDs are provenance
    labels only; they do not certify split independence, preprocessing, or an
    estimand.
    """
    macro_fold_scores: tuple[float, ...]
    micro_fold_scores: tuple[float, ...]
    evaluation_scope: str
    evaluation_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        try:
            macro = tuple(float(value) for value in self.macro_fold_scores)
            micro = tuple(float(value) for value in self.micro_fold_scores)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Paired predictive scores must be numeric.") from exc
        if not macro or len(macro) != len(micro):
            raise ValueError("Paired predictive scores must be nonempty and have equal lengths.")
        if not all(math.isfinite(value) for value in (*macro, *micro)):
            raise ValueError("Paired predictive scores must be finite.")
        if not isinstance(self.evaluation_scope, str) or not self.evaluation_scope.strip():
            raise ValueError("A nonempty predictive evaluation scope is required.")
        if not math.isfinite(float(np.mean(macro))) or not math.isfinite(float(np.mean(micro))):
            raise ValueError("Paired predictive score means must be finite.")
        if not math.isfinite(float(np.std(macro, ddof=0))) or not math.isfinite(float(np.std(micro, ddof=0))):
            raise ValueError("Paired predictive score dispersion must be finite.")
        if not all(math.isfinite(a - b) for a, b in zip(macro, micro)):
            raise ValueError("Paired fold differences must be finite.")
        if not math.isfinite(float(np.mean(macro) - np.mean(micro))):
            raise ValueError("Mean paired predictive difference must be finite.")
        if self.evaluation_ids is not None:
            ids = tuple(self.evaluation_ids)
            if (len(ids) != len(macro) or any(not isinstance(v, str) or not v.strip() for v in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError("Evaluation IDs must be unique nonempty strings, one per paired evaluation.")
            object.__setattr__(self, "evaluation_ids", ids)
        object.__setattr__(self, "macro_fold_scores", macro)
        object.__setattr__(self, "micro_fold_scores", micro)

    @property
    def macro_r2(self) -> float:
        return float(np.mean(self.macro_fold_scores))

    @property
    def micro_r2(self) -> float:
        return float(np.mean(self.micro_fold_scores))

    @property
    def predictive_r2_difference(self) -> float:
        """Unweighted, unclipped mean R² difference; not a significance test."""
        return self.macro_r2 - self.micro_r2

    @property
    def fold_r2_differences(self) -> tuple[float, ...]:
        return tuple(a - b for a, b in zip(self.macro_fold_scores, self.micro_fold_scores))


@dataclass(frozen=True)
class MacroScore:
    macro_name: str
    outcome: str
    target_column: str
    ranking_score: float
    predictive_r2_difference: float
    macro_r2: float
    micro_r2: float
    predictive_provenance: str
    specificity: float
    validation_specificity: float | None
    outcome_specificity: float
    specificity_definition: str
    stability: float
    compression: float
    state_count: int
    row_count: int
    macro_fold_scores: tuple[float, ...]
    micro_fold_scores: tuple[float, ...]
    fold_r2_differences: tuple[float, ...]
    predictive_evaluation_scope: str
    predictive_evaluation_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        scalars = (self.ranking_score, self.predictive_r2_difference, self.macro_r2,
                   self.micro_r2, self.specificity, self.outcome_specificity,
                   self.stability, self.compression)
        if not all(math.isfinite(float(v)) for v in scalars):
            raise ValueError("MacroScore scalar metrics must be finite.")
        if self.validation_specificity is not None and not math.isfinite(float(self.validation_specificity)):
            raise ValueError("Validation specificity must be finite when present.")
        if not isinstance(self.specificity_definition, str) or not self.specificity_definition.strip():
            raise ValueError("A specificity definition is required.")
        if not isinstance(self.predictive_evaluation_scope, str) or not self.predictive_evaluation_scope.strip():
            raise ValueError("A predictive evaluation scope is required.")
        if self.predictive_provenance not in ("caller_declared", "computed_training_only_refit"):
            raise ValueError("Predictive provenance must identify caller-declared or computed scores.")
        if self.specificity_definition not in (_VALIDATION_SPECIFICITY, _DESCRIPTIVE_SPECIFICITY):
            raise ValueError("Specificity definition must match a supported explicit estimand.")
        if (not isinstance(self.state_count, int) or isinstance(self.state_count, bool)
                or not isinstance(self.row_count, int) or isinstance(self.row_count, bool)
                or self.state_count < 1 or self.row_count < 1 or self.state_count > self.row_count):
            raise ValueError("MacroScore row and state counts must be positive.")
        macro = tuple(float(v) for v in self.macro_fold_scores)
        micro = tuple(float(v) for v in self.micro_fold_scores)
        diffs = tuple(float(v) for v in self.fold_r2_differences)
        if not macro or len(macro) != len(micro) or len(macro) != len(diffs):
            raise ValueError("MacroScore paired fold arrays must be nonempty and have equal lengths.")
        if not all(math.isfinite(v) for v in (*macro, *micro, *diffs)):
            raise ValueError("MacroScore fold metrics must be finite.")
        expected_diffs = tuple(a - b for a, b in zip(macro, micro))
        if any(not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(diffs, expected_diffs)):
            raise ValueError("MacroScore fold differences must match paired macro and micro scores.")
        mean_macro, mean_micro = float(np.mean(macro)), float(np.mean(micro))
        if not math.isclose(self.macro_r2, mean_macro, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("MacroScore macro_r2 must equal the mean macro fold score.")
        if not math.isclose(self.micro_r2, mean_micro, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("MacroScore micro_r2 must equal the mean micro fold score.")
        if not math.isclose(self.predictive_r2_difference, mean_macro - mean_micro, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("MacroScore predictive difference must match the paired means.")
        expected_ranking = sum(_ranking_components(mean_macro, self.specificity, self.stability, self.compression).values())
        if not math.isclose(self.ranking_score, expected_ranking, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("MacroScore ranking_score must match its declared ranking components.")
        if self.validation_specificity is not None and not math.isclose(
            self.specificity, self.validation_specificity, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("Ranking specificity must equal validation_specificity when present.")
        if self.validation_specificity is None and (
            self.specificity_definition != _DESCRIPTIVE_SPECIFICITY
            or not math.isclose(self.specificity, self.outcome_specificity, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ValueError("Descriptive fallback specificity must match outcome_specificity.")
        if self.validation_specificity is not None and self.specificity_definition != _VALIDATION_SPECIFICITY:
            raise ValueError("Validation specificity must use its held-out definition.")
        bounded = (self.specificity, self.outcome_specificity, self.stability, self.compression)
        if any(not 0.0 <= v <= 1.0 for v in bounded):
            raise ValueError("Specificity, stability, and compression values must lie in [0, 1].")
        if self.validation_specificity is not None and not 0.0 <= self.validation_specificity <= 1.0:
            raise ValueError("Validation specificity must lie in [0, 1].")
        if self.predictive_evaluation_ids is not None:
            ids = tuple(self.predictive_evaluation_ids)
            if (len(ids) != len(macro) or any(not isinstance(v, str) or not v.strip() for v in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError("Evaluation IDs must be unique nonempty strings, one per paired evaluation.")
            object.__setattr__(self, "predictive_evaluation_ids", ids)
        object.__setattr__(self, "macro_fold_scores", macro)
        object.__setattr__(self, "micro_fold_scores", micro)
        object.__setattr__(self, "fold_r2_differences", diffs)

    @property
    def macro_fold_r2_std(self) -> float | None:
        return _fold_r2_std(self.macro_fold_scores)

    @property
    def micro_fold_r2_std(self) -> float | None:
        return _fold_r2_std(self.micro_fold_scores)

    @property
    def score(self) -> float:
        """Deprecated compatibility alias for the ranking preference."""
        warnings.warn("MacroScore.score is deprecated; use ranking_score.", DeprecationWarning, stacklevel=2)
        return self.ranking_score

    @property
    def fold_scores(self) -> tuple[float, ...]:
        """Deprecated compatibility alias for raw macro fold R² values."""
        warnings.warn("MacroScore.fold_scores is deprecated; use macro_fold_scores.", DeprecationWarning, stacklevel=2)
        return self.macro_fold_scores

    def to_dict(self) -> dict[str, object]:
        return {
            "score_schema_version": SCORE_SCHEMA_VERSION,
            "macro_name": self.macro_name, "outcome": self.outcome, "target_column": self.target_column,
            "ranking_score": self.ranking_score,
            "ranking_components": _ranking_components(self.macro_r2, self.specificity, self.stability, self.compression),
            "ranking_weights": _ranking_weights(),
            "predictive_r2_difference": self.predictive_r2_difference,
            "predictive_provenance": self.predictive_provenance,
            "predictive_evaluation_scope": self.predictive_evaluation_scope,
            "predictive_evaluation_ids": None if self.predictive_evaluation_ids is None else list(self.predictive_evaluation_ids),
            "emergence_evidence_status": "not_assessed",
            "macro_r2": self.macro_r2, "micro_r2": self.micro_r2,
            "specificity": self.specificity,
            "validation_specificity": self.validation_specificity,
            "outcome_specificity": self.outcome_specificity,
            "specificity_definition": self.specificity_definition,
            "stability": self.stability, "compression": self.compression,
            "state_count": self.state_count, "row_count": self.row_count,
            "macro_fold_scores": list(self.macro_fold_scores),
            "micro_fold_scores": list(self.micro_fold_scores),
            "fold_r2_differences": list(self.fold_r2_differences),
            "macro_fold_r2_std": self.macro_fold_r2_std, "micro_fold_r2_std": self.micro_fold_r2_std,
            "score": self.ranking_score, "fold_scores": list(self.macro_fold_scores),
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
    predictive_scores: PairedPredictiveScores | None = None,
) -> MacroScore:
    """Rank a candidate using paired scores and explicit specificity definitions.

    Discovery passes development-only validation splits. Every fold refits its
    macro encoder and regressions from training rows. A caller-provided paired
    object only declares provenance; scope and IDs do not certify a design.
    """
    if outcome not in df.columns or target_column not in df.columns:
        raise ValueError("Outcome and target columns must be present in the scoring table.")
    working = df.reset_index(drop=True).copy()
    finite_target = pd.to_numeric(working[target_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if int(finite_target.notna().sum()) < 2:
        raise ValueError("At least two finite target rows are required for scoring.")
    if macro.labels.isna().any() or len(macro.labels) != len(working):
        raise ValueError("Macro labels must be present for every scoring row.")
    generated_default_splits = validation_splits is None and predictive_scores is None
    if validation_splits is None and predictive_scores is None:
        if folds < 2:
            raise ValueError("At least two folds are required.")
        idx = np.arange(len(working))
        validation_splits = [(idx[idx % folds != fold], idx[idx % folds == fold]) for fold in range(folds)]

    labeled = working.assign(**{MACRO_COLUMN: macro.labels.to_numpy()})
    descriptive = outcome_specificity(labeled, MACRO_COLUMN, target_column)
    validation_specificity = None
    if predictive_scores is not None:
        paired = predictive_scores
        if validation_splits is None:
            specificity, definition = descriptive, "descriptive_between_state_outcome_variance_share"
        else:
            specificity = _validation_specificity(working, macro, target_column, validation_splits)
            validation_specificity = specificity
            definition = "mean_positive_clipped_validation_r2_of_training_fitted_state_means"
    else:
        splits = list(validation_splits)
        if not splits:
            raise ValueError("No development validation folds were supplied.")
        macro_scores, micro_scores, spec_scores = [], [], []
        macro_features = _ordered_unique([*intervention_columns, MACRO_COLUMN])
        micro_features = _ordered_unique([*intervention_columns, *micro_feature_columns])
        for split in splits:
            train_idx, test_idx = _split_indices(split, len(working))
            train = working.iloc[train_idx].reset_index(drop=True).copy()
            test = working.iloc[test_idx].reset_index(drop=True).copy()
            fitted = refit_macro(macro, train)
            train[MACRO_COLUMN] = fitted.labels.astype(str).to_numpy()
            test[MACRO_COLUMN] = apply_macro(fitted, test).labels.astype(str).to_numpy()
            macro_fit = fit_linear_model(train, macro_features, target_column)
            micro_fit = fit_linear_model(train, micro_features, target_column)
            state_fit = fit_linear_model(train, [MACRO_COLUMN], target_column)
            macro_scores.append(predict_r2(macro_fit, test, macro_features, target_column))
            micro_scores.append(predict_r2(micro_fit, test, micro_features, target_column))
            spec_scores.append(_positive_specificity(predict_r2(state_fit, test, [MACRO_COLUMN], target_column)))
        paired = PairedPredictiveScores(
            tuple(macro_scores), tuple(micro_scores),
            (DEFAULT_PREDICTIVE_SCOPE if generated_default_splits
             else "provided_validation_splits_training_only_refit"),
            tuple(f"fold-{i + 1}" for i in range(len(splits))),
        )
        specificity = float(np.mean(spec_scores))
        validation_specificity = specificity
        definition = "mean_positive_clipped_validation_r2_of_training_fitted_state_means"

    stability = fold_stability(paired.macro_fold_scores)
    compression = compression_score(macro.state_count, len(working))
    components = _ranking_components(paired.macro_r2, specificity, stability, compression)
    return MacroScore(
        macro_name=macro.name, outcome=outcome, target_column=target_column,
        ranking_score=float(sum(components.values())),
        predictive_r2_difference=paired.predictive_r2_difference,
        macro_r2=paired.macro_r2, micro_r2=paired.micro_r2,
        predictive_provenance=("caller_declared" if predictive_scores is not None else "computed_training_only_refit"),
        specificity=float(specificity), validation_specificity=validation_specificity,
        outcome_specificity=descriptive, specificity_definition=definition,
        stability=stability, compression=compression,
        state_count=macro.state_count, row_count=len(working),
        macro_fold_scores=paired.macro_fold_scores, micro_fold_scores=paired.micro_fold_scores,
        fold_r2_differences=paired.fold_r2_differences,
        predictive_evaluation_scope=paired.evaluation_scope,
        predictive_evaluation_ids=paired.evaluation_ids,
    )


def _split_indices(split, row_count: int):
    if hasattr(split, "train_indices") and hasattr(split, "test_indices"):
        train_idx, test_idx = split.train_indices, split.test_indices
    else:
        train_idx, test_idx = split
    train_idx, test_idx = np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)
    if (train_idx.ndim != 1 or test_idx.ndim != 1 or not len(train_idx) or not len(test_idx)
            or np.any(train_idx < 0) or np.any(test_idx < 0)
            or np.any(train_idx >= row_count) or np.any(test_idx >= row_count)
            or len(np.unique(train_idx)) != len(train_idx) or len(np.unique(test_idx)) != len(test_idx)
            or np.intersect1d(train_idx, test_idx).size):
        raise ValueError("Each scoring split must have unique, disjoint, nonempty in-range index arrays.")
    return train_idx.tolist(), test_idx.tolist()


def _validation_specificity(df: pd.DataFrame, macro: MacroAssignment, target: str, splits) -> float:
    scores = []
    for split in splits:
        train_idx, test_idx = _split_indices(split, len(df))
        train, test = df.iloc[train_idx].reset_index(drop=True).copy(), df.iloc[test_idx].reset_index(drop=True).copy()
        fitted = refit_macro(macro, train)
        train[MACRO_COLUMN] = fitted.labels.astype(str).to_numpy()
        test[MACRO_COLUMN] = apply_macro(fitted, test).labels.astype(str).to_numpy()
        fit = fit_linear_model(train, [MACRO_COLUMN], target)
        scores.append(_positive_specificity(predict_r2(fit, test, [MACRO_COLUMN], target)))
    if not scores:
        raise ValueError("No development validation folds were supplied.")
    return float(np.mean(scores))


def _positive_specificity(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Validation specificity must be finite.")
    return max(0.0, value)


def _ranking_weights() -> dict[str, float]:
    return {"positive_macro_r2": 0.45, "specificity": 0.25, "stability": 0.20, "compression": 0.10}


def _ranking_components(macro_r2: float, specificity: float, stability: float, compression: float) -> dict[str, float]:
    weights = _ranking_weights()
    return {
        "positive_macro_r2": weights["positive_macro_r2"] * max(macro_r2, 0.0),
        "specificity": weights["specificity"] * specificity,
        "stability": weights["stability"] * stability,
        "compression": weights["compression"] * compression,
    }


def outcome_specificity(df: pd.DataFrame, macro_column: str, target_column: str) -> float:
    """Descriptive in-sample between-state variance share."""
    clean = df[[macro_column, target_column]].dropna()
    if len(clean) < 2:
        return 0.0
    target = pd.to_numeric(clean[target_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    clean = clean.loc[target.notna()].copy()
    finite = target.loc[target.notna()].astype(float)
    if len(finite) < 2:
        return 0.0
    # The variance share is scale-invariant. Normalize first so finite, very
    # large or small targets do not overflow or underflow into a fabricated 0.
    scale = float(finite.abs().max())
    clean[target_column] = finite / scale if scale else finite
    total_var = float(clean[target_column].var(ddof=0))
    if not math.isfinite(total_var):
        raise ValueError("Outcome specificity variance is not finite.")
    if total_var == 0.0:
        return 0.0
    overall, between = float(clean[target_column].mean()), 0.0
    for _, values in clean.groupby(macro_column)[target_column]:
        between += len(values) / len(clean) * (float(values.mean()) - overall) ** 2
    value = between / total_var
    if not math.isfinite(value):
        raise ValueError("Outcome specificity is not finite.")
    return float(max(0.0, min(1.0, value)))


def fold_stability(scores: list[float] | tuple[float, ...]) -> float:
    """Inverse dispersion of raw macro R², including negative values.

    Consistently poor folds can still be stable. This measures dispersion, not
    accuracy or uncertainty. One fold cannot estimate it.
    """
    spread = _fold_r2_std(scores)
    return 0.0 if spread is None else float(1.0 / (1.0 + spread))


def _fold_r2_std(scores: list[float] | tuple[float, ...]) -> float | None:
    raw = np.asarray(scores, dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("Fold scores must be finite.")
    return None if len(raw) < 2 else float(raw.std(ddof=0))


def compression_score(state_count: int, row_count: int) -> float:
    if state_count <= 1:
        return 0.0
    value = 1.0 - math.log2(state_count) / math.log2(max(row_count, 2))
    return float(max(0.0, min(1.0, value)))


def _ordered_unique(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))

