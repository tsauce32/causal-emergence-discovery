import numpy as np
import pandas as pd
import pytest

import causal_emergence_discovery.scoring as scoring
from causal_emergence_discovery.macro import MacroAssignment, quantile_macro
from causal_emergence_discovery.scoring import (
    PairedPredictiveScores,
    fold_stability,
    score_macro,
)


def _score_fixed_assignment(df, macro, *, folds=5, predictive_scores=None, **kwargs):
    """Preserve reviewed fixed-candidate controls via caller-declared scores.

    Their externally specified state assignment is fixed before row CV. This
    is distinct from Discovery's training-only recipe refits, tested elsewhere.
    An independent NumPy OLS calculation preserves the original raw control
    numbers without restoring the old scorer's fitted-full-data fallback.
    """
    if predictive_scores is None:
        target = df[kwargs["target_column"]].to_numpy(dtype=float)
        macro_design = np.column_stack([np.ones(len(df)), macro.labels.to_numpy(dtype=float)])
        micro_design = np.column_stack([np.ones(len(df)), df[kwargs["micro_feature_columns"]].to_numpy(dtype=float)])
        rows = np.arange(len(df))
        macro_scores, micro_scores = [], []
        for fold in range(folds):
            test = rows % folds == fold
            train = ~test
            centered = target[test] - target[test].mean()
            total = float(centered @ centered)
            for design, scores in ((macro_design, macro_scores), (micro_design, micro_scores)):
                beta = np.linalg.pinv(design[train].T @ design[train]) @ design[train].T @ target[train]
                residual = target[test] - design[test] @ beta
                scores.append(1.0 - float(residual @ residual) / total)
        predictive_scores = PairedPredictiveScores(tuple(macro_scores), tuple(micro_scores),
                                                  "fixed_assignment_row_modulo_cv_caller_declared")
    return score_macro(df, macro, folds=folds, predictive_scores=predictive_scores, **kwargs)


def test_seeded_null_keeps_raw_predictive_comparison_separate_from_ranking():
    rng = np.random.default_rng(20261001)
    df = pd.DataFrame({"x": rng.normal(size=1000), "y": rng.normal(size=1000)})
    result = _score_fixed_assignment(
        df,
        quantile_macro(df, "x", 2),
        outcome="y",
        target_column="y",
        micro_feature_columns=["x"],
        intervention_columns=[],
    )

    assert result.macro_r2 == pytest.approx(-0.0031202397463505173)
    assert result.micro_r2 == pytest.approx(-0.004641226000242682)
    assert result.predictive_r2_difference == pytest.approx(0.0015209862538921647)
    assert result.ranking_score > 0.25
    payload = result.to_dict()
    assert payload["ranking_components"] == pytest.approx(
        {
            "positive_macro_r2": 0.0,
            "specificity": 0.25 * result.specificity,
            "stability": 0.20 * result.stability,
            "compression": 0.10 * result.compression,
        }
    )
    assert payload["ranking_score"] == pytest.approx(sum(payload["ranking_components"].values()))
    assert payload["emergence_evidence_status"] == "not_assessed"
    assert "emergence_delta" not in payload
    assert not hasattr(result, "emergence_delta")


def test_paired_predictive_scores_are_raw_and_swap_sign_with_models():
    macro = PairedPredictiveScores((-0.2, 0.1, -0.1), (-0.1, -0.2, -0.3), "outer-folds")
    swapped = PairedPredictiveScores(
        macro.micro_fold_scores,
        macro.macro_fold_scores,
        "outer-folds",
    )

    assert macro.macro_r2 == pytest.approx(np.mean((-0.2, 0.1, -0.1)))
    assert macro.micro_r2 == pytest.approx(np.mean((-0.1, -0.2, -0.3)))
    assert macro.fold_r2_differences == pytest.approx((-0.1, 0.3, 0.2))
    assert macro.predictive_r2_difference == pytest.approx(-swapped.predictive_r2_difference)
    assert swapped.fold_r2_differences == pytest.approx(tuple(-x for x in macro.fold_r2_differences))


def test_negative_predictive_difference_can_coexist_with_positive_heuristic_preference():
    labels = pd.Series([0] * 50 + [1] * 50)
    df = pd.DataFrame({"y": [0.0] * 50 + [10.0] * 50})
    macro = MacroAssignment("group", labels, "test", (), {})
    paired = PairedPredictiveScores((-0.2, -0.3), (-0.05, -0.1), "provided")

    result = _score_fixed_assignment(
        df,
        macro,
        outcome="y",
        target_column="y",
        micro_feature_columns=[],
        intervention_columns=[],
        predictive_scores=paired,
    )

    assert result.predictive_r2_difference < 0.0
    assert result.ranking_score > 0.5
    assert result.to_dict()["emergence_evidence_status"] == "not_assessed"


def test_tiny_null_and_strong_planted_signal_both_leave_emergence_unassessed():
    rng = np.random.default_rng(20261001)
    null_df = pd.DataFrame({"x": rng.normal(size=1000), "y": rng.normal(size=1000)})
    null_result = _score_fixed_assignment(
        null_df,
        quantile_macro(null_df, "x", 2),
        outcome="y",
        target_column="y",
        micro_feature_columns=["x"],
        intervention_columns=[],
    )

    rng = np.random.default_rng(20261001)
    x1 = rng.normal(size=1000)
    x2 = rng.normal(size=1000)
    labels = pd.Series((x1 * x2 > 0).astype(int))
    plant_df = pd.DataFrame(
        {"x1": x1, "x2": x2, "y": 2.0 * labels.to_numpy() + rng.normal(scale=0.7, size=1000)}
    )
    plant_macro = MacroAssignment("sign_product", labels, "synthetic", ("x1", "x2"), {})
    plant_result = _score_fixed_assignment(
        plant_df,
        plant_macro,
        outcome="y",
        target_column="y",
        micro_feature_columns=["x1", "x2"],
        intervention_columns=[],
    )

    assert 0.0 < null_result.predictive_r2_difference < 0.01
    assert null_result.to_dict()["emergence_evidence_status"] == "not_assessed"
    assert plant_result.predictive_r2_difference > 0.6
    assert plant_result.to_dict()["emergence_evidence_status"] == "not_assessed"


def test_equivalent_micro_interaction_feature_removes_planted_macro_predictive_gap():
    rng = np.random.default_rng(20261001)
    x1 = rng.normal(size=1000)
    x2 = rng.normal(size=1000)
    signal_indicator = (x1 * x2 > 0).astype(int)
    labels = pd.Series(signal_indicator)
    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "signal_indicator": signal_indicator,
            "y": 2.0 * signal_indicator + rng.normal(scale=0.7, size=1000),
        }
    )
    macro = MacroAssignment("sign_product", labels, "synthetic", ("x1", "x2"), {})

    result = _score_fixed_assignment(
        df,
        macro,
        outcome="y",
        target_column="y",
        micro_feature_columns=["signal_indicator"],
        intervention_columns=[],
    )

    assert result.macro_r2 > 0.6
    assert result.micro_r2 == pytest.approx(result.macro_r2, abs=1e-12)
    assert result.predictive_r2_difference == pytest.approx(0.0, abs=1e-12)
    assert result.to_dict()["emergence_evidence_status"] == "not_assessed"


def test_raw_negative_fold_dispersion_affects_stability_but_uniformly_poor_folds_can_be_stable():
    near = fold_stability((-1.0, -1.1, -0.9))
    dispersed = fold_stability((-3.0, -1.0, -2.0))
    uniformly_poor = fold_stability((-4.0, -4.0, -4.0))

    assert near == pytest.approx(1.0 / (1.0 + np.std((-1.0, -1.1, -0.9))))
    assert dispersed == pytest.approx(1.0 / (1.0 + np.std((-3.0, -1.0, -2.0))))
    assert dispersed < near
    assert uniformly_poor == pytest.approx(1.0)


def test_single_fold_has_unassessed_dispersion_and_zero_stability_preference():
    assert fold_stability((0.8,)) == 0.0
    paired = PairedPredictiveScores((0.8,), (0.1,), "single-holdout")
    df = pd.DataFrame({"y": [0.0, 1.0, 2.0, 3.0]})
    macro = MacroAssignment("two-state", pd.Series([0, 0, 1, 1]), "test", (), {})

    result = _score_fixed_assignment(
        df,
        macro,
        outcome="y",
        target_column="y",
        micro_feature_columns=[],
        intervention_columns=[],
        predictive_scores=paired,
    )

    assert result.stability == 0.0
    assert result.to_dict()["ranking_components"]["stability"] == 0.0
    assert result.to_dict()["macro_fold_r2_std"] is None
    assert result.to_dict()["micro_fold_r2_std"] is None


def test_caller_supplied_paired_scores_bypass_model_fitting(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("model fitting should be bypassed for supplied scores")

    monkeypatch.setattr(scoring, "fit_linear_model", fail_if_called)
    paired = PairedPredictiveScores(
        (0.6, 0.7), (0.1, 0.2), "outer-folds", evaluation_ids=("fold-a", "fold-b")
    )
    df = pd.DataFrame({"y": [0.0, 0.1, 0.9, 1.0]})
    macro = MacroAssignment("two-state", pd.Series([0, 0, 1, 1]), "test", (), {})

    result = _score_fixed_assignment(
        df,
        macro,
        outcome="y",
        target_column="y",
        micro_feature_columns=[],
        intervention_columns=[],
        predictive_scores=paired,
    )

    assert result.macro_fold_scores == (0.6, 0.7)
    assert result.micro_fold_scores == (0.1, 0.2)
    assert result.predictive_evaluation_scope == "outer-folds"
    assert result.predictive_evaluation_ids == ("fold-a", "fold-b")
    assert result.to_dict()["predictive_evaluation_ids"] == ["fold-a", "fold-b"]
    assert result.macro_fold_r2_std == pytest.approx(0.05)
    assert result.micro_fold_r2_std == pytest.approx(0.05)


@pytest.mark.parametrize(
    "macro_scores,micro_scores,scope",
    [
        ((), (), "scope"),
        ((0.1,), (), "scope"),
        ((0.1, float("nan")), (0.0, 0.0), "scope"),
        ((0.1,), (float("inf"),), "scope"),
        ((0.1,), (0.0,), "  "),
    ],
)
def test_invalid_paired_predictive_inputs_are_rejected(macro_scores, micro_scores, scope):
    with pytest.raises(ValueError):
        PairedPredictiveScores(macro_scores, micro_scores, scope)


@pytest.mark.parametrize(
    "evaluation_ids",
    [
        (),
        ("fold-a",),
        ("fold-a", "fold-a"),
        ("fold-a", " "),
        ("fold-a", 2),
    ],
)
def test_invalid_evaluation_ids_are_rejected(evaluation_ids):
    with pytest.raises(ValueError):
        PairedPredictiveScores((0.1, 0.2), (0.0, 0.0), "scope", evaluation_ids)


def test_deprecated_aliases_warn_and_serialization_has_only_honest_compatibility_aliases():
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 0.0, 1.0, 1.0]})
    result = _score_fixed_assignment(
        df,
        quantile_macro(df, "x", 2),
        outcome="y",
        target_column="y",
        micro_feature_columns=["x"],
        intervention_columns=[],
        folds=2,
    )

    with pytest.warns(DeprecationWarning, match="ranking_score"):
        assert result.score == result.ranking_score
    with pytest.warns(DeprecationWarning, match="macro_fold_scores"):
        assert result.fold_scores == result.macro_fold_scores

    payload = result.to_dict()
    assert payload["score"] == payload["ranking_score"]
    assert payload["fold_scores"] == payload["macro_fold_scores"]
    assert "emergence_delta" not in payload
