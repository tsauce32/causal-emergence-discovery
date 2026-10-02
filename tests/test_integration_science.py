"""Scientific regressions for the integrated discovery and scoring pipeline.

These tests check cross-seam meaning: search preferences remain distinct from
paired predictive estimates, development validation remains distinct from the
selected outer evaluation, and prediction gaps depend on the declared model
classes. They make no emergence or causal-identification claim.
"""

import numpy as np
import pandas as pd
import pytest

from causal_emergence_discovery.discovery import DiscoveryConfig, run_discovery
from causal_emergence_discovery.macro import MacroAssignment, quantile_macro
from causal_emergence_discovery.scoring import PairedPredictiveScores, score_macro
from causal_emergence_discovery.spec import StudySpec
from causal_emergence_discovery.validation import build_validation_plan


def test_integrated_result_separates_ranking_prediction_and_outer_evaluation():
    frame = _panel(seed=75, entities=30, periods=14)
    study = _spec()
    result = run_discovery(
        frame,
        study,
        DiscoveryConfig(
            outcome="y", max_states=3, paths=2, branching_factor=2,
            folds=3, top_k=4, seed=7,
        ),
    )

    assert result["schema_version"] == 2
    assert result["status"] == "experimental_hypothesis_generation"
    assert result["top_macros"]
    for record in result["top_macros"]:
        assert record["evaluation_scope"] == "development_selection_only"
        assert record["specificity_definition"] == (
            "mean_positive_clipped_validation_r2_of_training_fitted_state_means"
        )
        assert record["validation_specificity"] == pytest.approx(record["specificity"])
        assert record["ranking_score"] == pytest.approx(
            sum(record["ranking_components"].values())
        )
        assert record["predictive_r2_difference"] == pytest.approx(
            record["macro_r2"] - record["micro_r2"]
        )
        assert record["macro_r2"] == pytest.approx(np.mean(record["macro_fold_scores"]))
        assert record["micro_r2"] == pytest.approx(np.mean(record["micro_fold_scores"]))
        assert record["fold_r2_differences"] == pytest.approx(
            np.subtract(record["macro_fold_scores"], record["micro_fold_scores"])
        )
        assert record["emergence_evidence_status"] == "not_assessed"
        assert "emergence_delta" not in record

    rank_order = [
        (row["ranking_score"], row["specificity"], row["state_count"], row["macro_name"])
        for row in result["top_macros"]
    ]
    assert rank_order == sorted(rank_order, reverse=True)

    selected = result["top_macros"][0]
    outer = result["outer_evaluation"]
    assert outer["scope"] == "untouched_outer_holdout_selected_macro"
    assert outer["macro_name"] == selected["macro_name"]
    assert "ranking_score" not in outer
    assert "predictive_r2_difference" in outer
    assert result["validation"]["audit"]["outcomes_inspected_for_selection"] is False


def test_integrated_fold_dispersion_retains_negative_validation_r2():
    rng = np.random.default_rng(20261001)
    frame = pd.DataFrame({"x": rng.normal(size=1000), "y": rng.normal(size=1000)})
    splits = [
        (np.flatnonzero(np.arange(len(frame)) % 5 != fold),
         np.flatnonzero(np.arange(len(frame)) % 5 == fold))
        for fold in range(5)
    ]
    result = score_macro(
        frame,
        quantile_macro(frame, "x", 2),
        outcome="y",
        target_column="y",
        micro_feature_columns=["x"],
        intervention_columns=[],
        validation_splits=splits,
    ).to_dict()

    raw_folds = np.asarray(result["macro_fold_scores"], dtype=float)
    assert np.any(raw_folds < 0.0)
    assert result["macro_fold_r2_std"] == pytest.approx(raw_folds.std(ddof=0))
    assert result["stability"] == pytest.approx(1.0 / (1.0 + raw_folds.std(ddof=0)))
    assert result["macro_r2"] == pytest.approx(raw_folds.mean())
    assert result["predictive_r2_difference"] == pytest.approx(
        result["macro_r2"] - result["micro_r2"]
    )
    assert result["emergence_evidence_status"] == "not_assessed"


def test_external_pair_scope_is_serialized_as_caller_declared_provenance():
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 0.2, 0.8, 1.0]})
    macro = quantile_macro(frame, "x", 2)
    supplied = PairedPredictiveScores(
        (0.31, 0.29), (0.12, 0.10), "caller_claimed_outer_holdout", ("fold-a", "fold-b")
    )
    payload = score_macro(
        frame, macro, outcome="y", target_column="y", micro_feature_columns=["x"],
        intervention_columns=[], predictive_scores=supplied,
    ).to_dict()

    assert payload["predictive_provenance"] == "caller_declared"
    assert payload["predictive_evaluation_scope"] == "caller_claimed_outer_holdout"
    assert payload["predictive_evaluation_ids"] == ["fold-a", "fold-b"]
    assert payload["emergence_evidence_status"] == "not_assessed"


def test_descriptive_specificity_is_scale_invariant_for_large_finite_targets():
    labels = pd.Series([0, 0, 1, 1])
    macro = MacroAssignment("two-state", labels, "manual-control", (), {})
    paired = PairedPredictiveScores((0.2, 0.3), (0.1, 0.2), "caller_declared_test_pair")
    ordinary = pd.DataFrame({"y": [1.0, 1.0, -1.0, -1.0]})
    extreme = ordinary.assign(y=ordinary["y"] * 1e308)

    def outcome_specificity(frame):
        return score_macro(
            frame, macro, outcome="y", target_column="y", micro_feature_columns=[],
            intervention_columns=[], predictive_scores=paired,
        ).outcome_specificity

    assert outcome_specificity(ordinary) == pytest.approx(1.0)
    assert outcome_specificity(extreme) == pytest.approx(outcome_specificity(ordinary))


def test_caller_supplied_pair_does_not_make_all_missing_targets_scoreable():
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [np.nan] * 4})
    macro = quantile_macro(frame.assign(y=0.0), "x", 2)
    paired = PairedPredictiveScores((0.4, 0.5), (0.1, 0.2), "caller_claimed_outer_holdout")

    with pytest.raises(ValueError, match="finite target"):
        score_macro(
            frame, macro, outcome="y", target_column="y", micro_feature_columns=["x"],
            intervention_columns=[], predictive_scores=paired,
        )


def test_sign_product_gain_disappears_with_information_matched_micro_feature():
    rng = np.random.default_rng(20261001)
    row_count = 1000
    # Balance all four sign quadrants so every validation training fold retains
    # exactly half positive and half negative products. This keeps a fitted
    # median split on the binary planted state identifiable after each refit.
    quadrant_signs = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
    x1 = np.concatenate([a * np.abs(rng.normal(size=row_count // 4)) for a, _ in quadrant_signs])
    x2 = np.concatenate([b * np.abs(rng.normal(size=row_count // 4)) for _, b in quadrant_signs])
    permutation = rng.permutation(row_count)
    x1, x2 = x1[permutation], x2[permutation]
    indicator = (x1 * x2 > 0.0).astype(float)
    y = 2.0 * indicator + rng.normal(scale=0.7, size=row_count)
    frame = pd.DataFrame({"x1": x1, "x2": x2, "interaction_indicator": indicator, "y": y})
    # The representation is planted from x1*x2 so this is a model-class
    # control, not evidence that candidate discovery found a new causal state.
    macro = quantile_macro(frame, "interaction_indicator", 2)
    folds = 5
    classes = [np.flatnonzero(indicator == value) for value in (0.0, 1.0)]
    class_folds = [np.array_split(indices, folds) for indices in classes]
    splits = [
        (
            np.setdiff1d(np.arange(row_count), np.concatenate([class_folds[0][fold], class_folds[1][fold]])),
            np.concatenate([class_folds[0][fold], class_folds[1][fold]]),
        )
        for fold in range(folds)
    ]
    additive = score_macro(
        frame, macro, outcome="y", target_column="y",
        micro_feature_columns=["x1", "x2"], intervention_columns=[],
        validation_splits=splits,
    )
    matched = score_macro(
        frame, macro, outcome="y", target_column="y",
        # Supply the exact interaction information to the micro reference.
        micro_feature_columns=["interaction_indicator"],
        intervention_columns=[], validation_splits=splits,
    )

    assert additive.macro_r2 > 0.6
    assert additive.predictive_r2_difference > 0.6
    assert matched.macro_r2 == pytest.approx(additive.macro_r2, abs=1e-12)
    assert matched.micro_r2 == pytest.approx(matched.macro_r2, abs=1e-12)
    assert matched.predictive_r2_difference == pytest.approx(0.0, abs=1e-12)
    assert additive.to_dict()["emergence_evidence_status"] == "not_assessed"
    assert matched.to_dict()["emergence_evidence_status"] == "not_assessed"


def test_mixed_integer_string_entity_holdout_is_permutation_invariant():
    entities = [value for n in range(1, 7) for value in (str(n), n)]
    rows = [
        {"entity": entity, "time": time, "x": float(time), "y": float(time + i)}
        for i, entity in enumerate(entities)
        for time in range(5)
    ]
    frame = pd.DataFrame(rows)

    def membership(data):
        plan = build_validation_plan(
            data, id_column="entity", time_column="time", target_time_column="target_time",
            mode="entity_holdout", folds=3, holdout_fraction=0.25, seed=4,
        )
        outer = data.iloc[list(plan.outer.test_indices)]["entity"]
        inner_frame = data.iloc[list(plan.outer.train_indices)].reset_index(drop=True)
        inner = [
            frozenset(_typed(value) for value in inner_frame.iloc[list(split.test_indices)]["entity"])
            for split in plan.inner
        ]
        return frozenset(_typed(value) for value in outer), inner

    frame["target_time"] = frame["time"] + 1
    forward = membership(frame.reset_index(drop=True))
    reversed_rows = membership(frame.iloc[::-1].reset_index(drop=True))
    assert forward == reversed_rows


def _typed(value):
    return (type(value).__name__, value)


def _panel(*, seed: int, entities: int, periods: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for entity in range(entities):
        previous_y = rng.normal()
        entity_offset = rng.normal(scale=1.2)
        for time in range(periods):
            x = rng.normal()
            rows.append({"entity": entity, "time": time, "x": x, "y": previous_y})
            previous_y = 1.7 * x + entity_offset + rng.normal(scale=0.35)
    return pd.DataFrame(rows)


def _spec() -> StudySpec:
    return StudySpec.from_dict({
        "dataset": {"id_column": "entity", "time_column": "time"},
        "outcomes": ["y"],
        "columns": {
            "x": {"role": "context", "type": "numeric"},
            "y": {
                "role": "outcome", "type": "numeric",
                "allowed_as_cause": False, "allowed_as_adjustment": False,
            },
        },
    })
