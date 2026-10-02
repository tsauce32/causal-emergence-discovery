import numpy as np
import pandas as pd
import pytest

from benchmarks.pipeline.baselines import (
    evaluate_r2,
    fit_matched_micro,
    fit_ols,
    fit_predict_r2,
    fit_state_model,
)


def test_numeric_ols_recovers_fixed_relation_and_reports_finite_holdout_r2():
    x = np.arange(40, dtype=float)
    frame = pd.DataFrame({"x": x, "y": 2.0 + 3.0 * x})
    score, prediction, target = fit_predict_r2(frame.iloc[:30], frame.iloc[30:], ["x"], "y")
    assert score == pytest.approx(1.0)
    np.testing.assert_allclose(prediction, target)


def test_encoder_is_train_only_for_missing_values_and_unseen_categories():
    train = pd.DataFrame({"x": [1.0, np.nan, 3.0], "group": ["a", "b", "a"], "y": [1, 2, 3]})
    test = pd.DataFrame({"x": [10_000.0], "group": ["new"], "y": [4]})
    fit = fit_ols(train, ["x", "group"], "y")
    assert fit.numeric["x"] == 2.0
    assert fit.categories["group"] == ("a", "b")
    assert np.isfinite(fit.predict(test)).all()


def test_flexible_basis_does_not_quadratic_expand_category_dummies():
    frame = pd.DataFrame({"x": np.arange(8.), "marker": [f"e{i}" for i in range(8)],
                          "y": np.arange(8.) ** 2})
    fit = fit_ols(frame, ["x", "marker"], "y", polynomial_degree=2)
    # Intercept + numeric + 7 reference-coded levels + one numeric square.
    assert len(fit.coefficients) == 10
    assert fit.terms == ((0, 0),)


def test_unseen_category_uses_training_reference_contrast():
    train = pd.DataFrame({"entity_marker": ["e0", "e0", "e1", "e1"],
                          "y": [0.0, 0.0, 10.0, 10.0]})
    test = pd.DataFrame({"entity_marker": ["unseen", "unseen"], "y": [4.0, 6.0]})
    fit = fit_ols(train, ["entity_marker"], "y")
    np.testing.assert_allclose(fit.predict(test), [0.0, 0.0], atol=1e-12)
    # Matches the package contract: levels[0] is the reference and unknown
    # categories are all-zero contrasts, hence they receive the reference mean.
    assert fit.categories["entity_marker"] == ("e0", "e1")
    assert fit.category_keys["entity_marker"] == ((False, "e0"), (False, "e1"))


def test_matched_recipe_unknown_state_uses_reference_contrast():
    train = pd.DataFrame({"y": [0.0, 10.0, 0.0, 10.0]})
    test = pd.DataFrame({"y": [4.0, 6.0]})
    matched = fit_matched_micro(train, test, ["low", "high", "low", "high"],
                                ["not-seen", "not-seen"], "y")[0]
    # Unknown state indicators are zero, matching a reference-level macro prediction.
    assert matched == pytest.approx(-25.0)


def test_matched_recipe_basis_reproduces_state_model_independently():
    train = pd.DataFrame({"t": [0., 1., 0., 1., 0., 1.], "y": [0., 2., .1, 2.1, -.1, 1.9]})
    test = pd.DataFrame({"t": [0., 1., 0., 1.], "y": [0., 2., .2, 1.8]})
    train_state = ["low", "high", "low", "high", "low", "high"]
    test_state = ["low", "high", "low", "high"]
    macro = fit_state_model(train, test, train_state, test_state, "y", extra_features=["t"])[0]
    matched = fit_matched_micro(train, test, train_state, test_state, "y", extra_features=["t"])[0]
    assert matched == pytest.approx(macro, abs=1e-12)


def test_nonconstant_test_target_is_required():
    train = pd.DataFrame({"x": [0., 1., 2.], "y": [0., 1., 2.]})
    fit = fit_ols(train, ["x"], "y")
    with pytest.raises(ValueError, match="variance"):
        evaluate_r2(fit, pd.DataFrame({"x": [0., 1.], "y": [2., 2.]}), "y")
