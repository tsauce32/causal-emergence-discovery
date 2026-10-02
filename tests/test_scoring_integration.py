from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from causal_emergence_discovery.macro import MacroAssignment, quantile_macro
from causal_emergence_discovery.scoring import (
    MacroScore,
    PairedPredictiveScores,
    score_macro,
)


def _frame_and_macro():
    x = np.linspace(-2.0, 2.0, 60)
    y = np.where(x > 0, 1.5, -1.0) + 0.2 * np.sin(np.arange(len(x)))
    frame = pd.DataFrame({"x": x, "y": y})
    return frame, quantile_macro(frame, "x", 3)


def _score():
    frame, macro = _frame_and_macro()
    return score_macro(
        frame,
        macro,
        outcome="y",
        target_column="y",
        micro_feature_columns=["x"],
        intervention_columns=[],
        folds=3,
    )


def test_macro_score_constructor_rejects_inconsistent_paired_and_ranking_fields():
    score = _score()
    with pytest.raises(ValueError, match="predictive difference"):
        replace(score, predictive_r2_difference=score.predictive_r2_difference + 0.1)
    with pytest.raises(ValueError, match="ranking_score"):
        replace(score, ranking_score=score.ranking_score + 0.1)
    with pytest.raises(ValueError, match="fold differences"):
        replace(score, fold_r2_differences=(0.0,) * len(score.fold_r2_differences))


def test_macro_score_constructor_rejects_specificity_and_count_contradictions():
    score = _score()
    changed_specificity = score.specificity / 2
    with pytest.raises(ValueError, match="equal validation_specificity"):
        replace(
            score,
            specificity=changed_specificity,
            ranking_score=score.ranking_score + 0.25 * (changed_specificity - score.specificity),
        )
    with pytest.raises(ValueError, match="Descriptive fallback"):
        replace(score, validation_specificity=None)
    with pytest.raises(ValueError, match="counts"):
        replace(score, state_count=score.row_count + 1)


def test_paired_score_validates_finite_values_lengths_scope_and_ids():
    with pytest.raises(ValueError, match="equal lengths"):
        PairedPredictiveScores((0.1,), (0.2, 0.3), "caller says paired")
    with pytest.raises(ValueError, match="finite"):
        PairedPredictiveScores((float("nan"),), (0.2,), "caller says paired")
    with pytest.raises(ValueError, match="scope"):
        PairedPredictiveScores((0.1,), (0.2,), "  ")
    with pytest.raises(ValueError, match="IDs"):
        PairedPredictiveScores((0.1, 0.2), (0.2, 0.3), "caller says paired", ("fold-1", "fold-1"))


def test_caller_provenance_is_labeled_without_certifying_the_evaluation_design():
    frame, _ = _frame_and_macro()
    labels = pd.Series(np.where(frame["x"] > 0, 1, 0))
    fixed = MacroAssignment("fixed", labels, "caller", (), {})
    paired = PairedPredictiveScores((0.3, -0.1), (0.2, 0.0), "caller says held out", ("a", "b"))
    score = score_macro(
        frame,
        fixed,
        outcome="y",
        target_column="y",
        micro_feature_columns=["x"],
        intervention_columns=[],
        predictive_scores=paired,
    )

    payload = score.to_dict()
    assert payload["predictive_provenance"] == "caller_declared"
    assert payload["predictive_evaluation_scope"] == "caller says held out"
    assert payload["predictive_evaluation_ids"] == ["a", "b"]
    assert payload["emergence_evidence_status"] == "not_assessed"
    assert "emergence_delta" not in payload


def test_validation_specificity_is_computed_from_fold_local_recipe_refits():
    score = _score()
    payload = score.to_dict()

    assert payload["predictive_provenance"] == "computed_training_only_refit"
    assert payload["validation_specificity"] == pytest.approx(payload["specificity"])
    assert payload["specificity_definition"] == (
        "mean_positive_clipped_validation_r2_of_training_fitted_state_means"
    )
    assert "outcome_specificity" in payload
