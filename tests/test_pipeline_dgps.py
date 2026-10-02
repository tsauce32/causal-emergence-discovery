"""Contract tests for known-mechanism selection benchmark panels."""

import numpy as np
import pandas as pd
import pytest

from benchmarks.pipeline.dgps import CASES, generate_case
from causal_emergence_discovery.panel import build_lagged_table
from causal_emergence_discovery.spec import StudySpec
from causal_emergence_discovery.validation import build_validation_plan


def _lagged(case: str, seed: int = 31, **kwargs):
    generated = generate_case(case, seed=seed, **kwargs)
    spec = StudySpec.from_dict(generated.spec_dict)
    lagged = build_lagged_table(generated.frame, spec, lag=1)
    return generated, spec, lagged


@pytest.mark.parametrize("case", CASES)
def test_generator_is_deterministic_and_labels_align_with_raw_rows(case):
    left = generate_case(case, seed=9, n_entities=12, n_periods=8)
    right = generate_case(case, seed=9, n_entities=12, n_periods=8)
    pd.testing.assert_frame_equal(left.frame, right.frame)
    np.testing.assert_array_equal(left.prespecified_labels, right.prespecified_labels)
    assert left.case == case
    assert left.prespecified_labels.ndim == 1
    assert len(left.prespecified_labels) == len(left.frame)
    assert left.frame[["entity_id", "time"]].reset_index(drop=True).equals(
        pd.DataFrame(
            {
                "entity_id": np.repeat([f"e{i:04d}" for i in range(12)], 8),
                "time": np.tile(np.arange(8), 12),
            }
        )
    )


@pytest.mark.parametrize("case", CASES)
def test_lag_one_target_is_next_observed_outcome(case):
    generated, _, lagged = _lagged(case, seed=14, n_entities=7, n_periods=6)
    expected = generated.frame.copy()
    expected["y__expected"] = expected.groupby("entity_id", sort=False)["y"].shift(-1)
    expected["__lead_time_expected"] = expected.groupby("entity_id", sort=False)["time"].shift(-1)
    expected = expected.dropna(subset=["y__expected"]).reset_index(drop=True)
    np.testing.assert_array_equal(lagged.data["y__lead1"].to_numpy(), expected["y__expected"].to_numpy())
    np.testing.assert_array_equal(lagged.data["__lead_time"].to_numpy(), expected["__lead_time_expected"].to_numpy())
    # A raw initial y value is never part of the cause or adjustment contract.
    spec = StudySpec.from_dict(generated.spec_dict)
    assert spec.column_spec("y").can_be_effect()
    assert not spec.column_spec("y").can_cause()
    assert not spec.column_spec("y").can_adjust()
    assert "y" not in spec.feature_columns(list(generated.frame.columns))


def test_null_target_is_seeded_independently_of_available_predictors():
    generated, _, lagged = _lagged("iid_null", seed=202, n_entities=180, n_periods=40)
    data = lagged.data
    target = data["y__lead1"].to_numpy()
    for feature in ("x1", "x2", "x3"):
        assert abs(np.corrcoef(data[feature].to_numpy(), target)[0, 1]) < 0.05
    assert generated.truth["effect"] == 0.0


def test_additive_coefficients_and_threshold_oracle_are_grounded_in_lagged_rows():
    additive, _, add_lagged = _lagged("additive_signal", seed=3, n_entities=120, n_periods=30)
    add = add_lagged.data
    design = np.column_stack([np.ones(len(add)), add["x1"], add["x2"]])
    estimates = np.linalg.lstsq(design, add["y__lead1"], rcond=None)[0]
    np.testing.assert_allclose(estimates[1:], [1.5, -0.8], atol=0.06)
    assert additive.oracle_features == []

    threshold, _, threshold_lagged = _lagged("nonlinear_threshold", seed=4, n_entities=100, n_periods=24)
    tdata = threshold_lagged.data
    np.testing.assert_array_equal(threshold.frame["oracle_threshold"].to_numpy(), (threshold.frame.x1 > 0).astype(int))
    assert threshold.oracle_features == ["oracle_threshold"]
    assert "oracle_threshold" in threshold.spec_dict["exclude"]
    target_means = tdata.groupby((tdata.x1 > 0).astype(int))["y__lead1"].mean()
    assert target_means[1] - target_means[0] == pytest.approx(2.0, abs=0.12)


def test_sign_interaction_oracle_matches_formula_and_reference_contrast():
    generated, _, lagged = _lagged("sign_interaction", seed=15, n_entities=100, n_periods=25)
    frame = generated.frame
    oracle = np.sign(frame["x1"]) * np.sign(frame["x2"])
    np.testing.assert_array_equal(frame["oracle_sign_product"].to_numpy(), oracle.to_numpy())
    assert generated.oracle_features == ["oracle_sign_product"]
    assert generated.spec_dict["columns"]["oracle_sign_product"]["role"] == "exclude"
    data = lagged.data
    groups = (np.sign(data.x1) * np.sign(data.x2) > 0).astype(int)
    means = data.groupby(groups)["y__lead1"].mean()
    assert means[1] - means[0] == pytest.approx(3.0, abs=0.16)
    assert "oracle" in generated.truth["matched_reference"]


def test_fingerprint_is_entity_specific_and_has_no_numeric_shared_signal():
    generated, _, lagged = _lagged("entity_fingerprint", seed=7, n_entities=96, n_periods=30)
    data = lagged.data
    entity_means = data.groupby("entity_id")["y__lead1"].mean()
    assert entity_means.std() > 1.5
    for feature in ("x1", "x2", "x3"):
        assert abs(np.corrcoef(data[feature], data["y__lead1"])[0, 1]) < 0.08
    assert "entity_marker" in generated.frame
    assert generated.spec_dict["columns"]["entity_marker"]["type"] == "categorical"
    assert generated.spec_dict["columns"]["entity_marker"]["role"] == "covariate"
    effects = generated.truth["entity_effects"]
    assert len(set(effects.values())) == len(effects)


def test_drift_and_population_shift_use_forward_lag_eligible_boundary():
    fraction = 0.25
    n_periods = 16
    expected_boundary = (n_periods - 1) - int(np.ceil((n_periods - 1) * fraction))
    drift, _, drift_lagged = _lagged("temporal_drift", seed=8, n_entities=32, n_periods=n_periods, holdout_fraction=fraction)
    shift, _, _ = _lagged("population_shift", seed=8, n_entities=32, n_periods=n_periods, holdout_fraction=fraction)
    assert drift.truth["boundary_predictor_time"] == expected_boundary
    assert shift.truth["boundary_predictor_time"] == expected_boundary
    assert drift_lagged.data["__lead_time"].max() == n_periods - 1
    y = drift_lagged.data["y__lead1"]
    x = drift_lagged.data["x1"]
    times = drift_lagged.data["time"]
    pre_slope = np.polyfit(x[times < expected_boundary], y[times < expected_boundary], 1)[0]
    post_slope = np.polyfit(x[times >= expected_boundary], y[times >= expected_boundary], 1)[0]
    assert pre_slope > 1.0
    assert post_slope < -1.0
    raw_shift = shift.frame.groupby("time")["x1"].mean()
    assert raw_shift.iloc[expected_boundary:].mean() - raw_shift.iloc[:expected_boundary].mean() == pytest.approx(1.5, abs=0.08)


def test_rare_state_has_valid_heldout_transition_and_unit_support_metadata():
    split_seed = 19
    fraction = 0.25
    generated, spec, lagged = _lagged(
        "rare_absent_states", seed=5, n_entities=48, n_periods=16,
        split_seed=split_seed, holdout_fraction=fraction,
    )
    tail = generated.frame["rare_tail_state"]
    tail_rows = generated.frame.loc[tail == 1]
    assert len(tail_rows) == 1
    predictor_time = generated.truth["rare_tail_predictor_time"]
    assert int(tail_rows.iloc[0]["time"]) == predictor_time
    assert predictor_time in set(lagged.data["time"])
    observed = lagged.data.loc[(lagged.data.time == predictor_time) & (lagged.data.rare_tail_state == 1)]
    assert len(observed) == 1
    assert observed["y__lead1"].notna().all()
    support = generated.truth["prespecified_state_support"]
    assert support
    assert all({"raw_rows", "raw_entities", "lag_eligible_rows", "lag_eligible_entities"} <= set(v) for v in support.values())
    assert generated.truth["rare_tail_row_support"] == generated.truth["rare_tail_unit_support"] == 1
    forward = build_validation_plan(
        lagged.data, id_column=spec.id_column, time_column=spec.time_column,
        target_time_column=lagged.target_time_column, mode="forward_time", folds=2,
        holdout_fraction=fraction, seed=split_seed,
    )
    tail_frame_index = int(observed.index[0])
    assert tail_frame_index in forward.outer.test_indices
    entity_plan = build_validation_plan(
        lagged.data, id_column=spec.id_column, time_column=spec.time_column,
        target_time_column=lagged.target_time_column, mode="entity_holdout", folds=2,
        holdout_fraction=fraction, seed=split_seed,
    )
    assert tail_frame_index in entity_plan.outer.test_indices


def test_confounding_and_correlated_treatment_joint_truth_controls():
    confounded, spec, lagged = _lagged("known_confounding", seed=61, n_entities=100, n_periods=24)
    assert spec.adjustment is not None and list(spec.adjustment.columns) == ["u"]
    cdata = lagged.data
    cdesign = np.column_stack([np.ones(len(cdata)), cdata["treatment"], cdata["u"]])
    ccoef = np.linalg.lstsq(cdesign, cdata["y__lead1"], rcond=None)[0]
    np.testing.assert_allclose(ccoef[1:], [1.5, 4.0], atol=0.08)
    assert confounded.truth["treatment_effect"] == 1.5

    correlated, spec, lagged = _lagged("correlated_treatments", seed=62, n_entities=100, n_periods=24)
    assert tuple(spec.interventions) == ("treatment_a", "treatment_b")
    assert spec.adjustment is not None and tuple(spec.adjustment.columns) == ("u",)
    data = lagged.data
    design = np.column_stack([np.ones(len(data)), data["treatment_a"], data["treatment_b"], data["u"]])
    coef = np.linalg.lstsq(design, data["y__lead1"], rcond=None)[0]
    np.testing.assert_allclose(coef[1:], [1.7, -0.9, 1.2], atol=0.08)
    assert correlated.truth["joint_conditional_effects"] == {"treatment_a": 1.7, "treatment_b": -0.9}
    assert "W is independent" in correlated.truth["latent_shared_shock"]
    # The shared latent shock keeps the treatments correlated after U is
    # removed, so fitting either treatment alone alongside U changes its
    # coefficient relative to the joint conditional estimand.
    u_design = np.column_stack([np.ones(len(data)), data["u"]])
    residual_a = data["treatment_a"].to_numpy() - u_design @ np.linalg.lstsq(u_design, data["treatment_a"], rcond=None)[0]
    residual_b = data["treatment_b"].to_numpy() - u_design @ np.linalg.lstsq(u_design, data["treatment_b"], rcond=None)[0]
    assert np.corrcoef(residual_a, residual_b)[0, 1] == pytest.approx(0.69, abs=0.04)
    only_a = np.column_stack([np.ones(len(data)), data["treatment_a"], data["u"]])
    only_b = np.column_stack([np.ones(len(data)), data["treatment_b"], data["u"]])
    coef_a = np.linalg.lstsq(only_a, data["y__lead1"], rcond=None)[0][1]
    coef_b = np.linalg.lstsq(only_b, data["y__lead1"], rcond=None)[0][1]
    assert coef_a == pytest.approx(1.08, abs=0.12)
    assert coef_b == pytest.approx(0.28, abs=0.12)
    assert abs(coef_a - 1.7) > 0.4
    assert abs(coef_b - -0.9) > 0.8
