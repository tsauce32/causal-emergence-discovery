import numpy as np
import pandas as pd

from causal_emergence_discovery.models import (
    fit_linear_model,
    predict_linear_model,
    treatment_effect_records,
)


def test_nonestimable_treatment_is_suppressed_but_predictions_remain_finite():
    rng = np.random.default_rng(991)
    treatment = rng.normal(size=300)
    frame = pd.DataFrame({"t": treatment, "u": treatment.copy(), "y": 2 * treatment + rng.normal(size=300)})

    fit = fit_linear_model(frame, ["t", "u"], "y")
    record = treatment_effect_records(frame, ["t"], "y", ["u"])[0]

    assert fit.rank == 2
    assert fit.design_columns == 3
    assert np.isnan(fit.coefficients[fit.feature_names.index("t")])
    assert np.isfinite(predict_linear_model(fit, frame)).all()
    assert record["status"] == "non_estimable"
    assert record["coefficient"] is None
    assert record["stderr_iid"] is None
    assert record["uncertainty_status"] == "aliased_term"


def test_numeric_slope_stays_estimable_with_constant_nuisance_and_reference_categories():
    x = np.arange(20, dtype=float)
    frame = pd.DataFrame({"x": x, "constant": 1.0, "category": ["a", "b"] * 10, "y": 3 * x + np.tile([0.0, 1.0], 10)})

    fit = fit_linear_model(frame, ["x", "constant", "category"], "y")
    record = treatment_effect_records(frame, ["x"], "y", ["constant", "category"])[0]

    assert fit.rank < fit.design_columns
    np.testing.assert_allclose(fit.coefficients[fit.feature_names.index("x")], 3.0, atol=1e-10)
    assert record["estimable"] is True
    assert record["status"] == "estimable"


def test_literal_missing_label_and_actual_null_are_distinct_categorical_levels():
    frame = pd.DataFrame({
        "arm": pd.Series(["__missing__", None, "control", "__missing__", None, "control"], dtype="object"),
        "y": [1.0, 2.0, 0.0, 1.2, 2.1, -0.1],
    })

    record = treatment_effect_records(frame, ["arm"], "y", [])[0]
    from causal_emergence_discovery.models import fit_design_encoder

    levels = fit_design_encoder(frame, ["arm"]).category_keys["arm"]
    assert len(record["terms"]) == 2
    term_levels = [term["level"] for term in record["terms"]]
    assert {tuple(sorted(level.items())) for level in term_levels} == {
        (("is_missing", True), ("value", None)),
        (("is_missing", False), ("value", "control")),
    }
    assert (False, "__missing__") in levels
    assert (True, "") in levels


def test_saturated_model_has_no_fabricated_residual_uncertainty():
    frame = pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]})
    record = treatment_effect_records(frame, ["x"], "y", [])[0]

    assert record["residual_df"] == 0
    assert record["coefficient"] is not None
    assert record["stderr_iid"] is None
    assert record["uncertainty_status"] == "residual_df_zero"
