import numpy as np
import pandas as pd

from causal_emergence_discovery.macro import apply_macro, generate_candidate_macros, quantile_macro
from causal_emergence_discovery.models import fit_linear_model, predict_linear_model_with_report
from causal_emergence_discovery.schema import resolve_feature_schema


def test_object_dtype_from_heldout_string_does_not_remove_numeric_macro_candidates():
    whole = pd.DataFrame({"x": np.arange(20, dtype=float)})
    whole["x"] = whole["x"].astype(object)
    whole.loc[18, "x"] = "bad heldout value"
    development = whole.iloc[:16]

    candidates = generate_candidate_macros(development, ["x"], max_states=3)

    assert any(candidate.name.startswith("q2:x") for candidate in candidates)
    assert resolve_feature_schema(development, ["x"]).kind("x") == "numeric"


def test_regression_freezes_numeric_training_values_and_reports_malformed_holdout():
    train = pd.DataFrame({"x": np.arange(8, dtype=float).astype(object), "y": 2 * np.arange(8, dtype=float)})
    fit = fit_linear_model(train, ["x"], "y")
    heldout = pd.DataFrame({"x": ["malformed"]})

    predictions, diagnostics = predict_linear_model_with_report(fit, heldout)

    assert fit.encoder.schema.kind("x") == "numeric"
    assert np.isfinite(predictions).all()
    assert diagnostics["x"] == {"missing": 0, "invalid_numeric": 1, "unknown_category": 0}


def test_declared_type_overrides_training_inference_and_macro_application_reports_invalid_values():
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    macro = quantile_macro(train, "x", 2, column_specs={"x": {"type": "numeric"}})
    heldout = pd.DataFrame({"x": ["not numeric", 100.0]})

    applied = apply_macro(macro, heldout)

    assert applied.metadata["application_diagnostics"]["x"]["invalid_numeric"] == 1
    assert applied.labels.nunique() == 2

    numeric = pd.DataFrame({"v": [1, 2, 3]})
    forced_category = resolve_feature_schema(numeric, ["v"], {"v": {"type": "categorical"}})
    assert forced_category.kind("v") == "categorical"


def test_unknown_category_is_zero_encoded_and_reported_separately_from_missing():
    train = pd.DataFrame({"category": ["a", None, "b"], "y": [0.0, 1.0, 2.0]})
    fit = fit_linear_model(train, ["category"], "y")
    heldout = pd.DataFrame({"category": ["new", None]})

    _, diagnostics = predict_linear_model_with_report(fit, heldout)

    assert diagnostics["category"] == {"missing": 1, "invalid_numeric": 0, "unknown_category": 1}
    assert fit.encoder.categories["category"] == ("a", "b")


def test_heldout_string_does_not_change_training_boolean_fallback_kind():
    whole = pd.DataFrame({"flag": [True, False, True, False, True]})
    original_kind = resolve_feature_schema(whole.iloc[:4], ["flag"]).kind("flag")
    whole["flag"] = whole["flag"].astype(object)
    whole.loc[4, "flag"] = "heldout"

    poisoned_kind = resolve_feature_schema(whole.iloc[:4], ["flag"]).kind("flag")

    assert original_kind == poisoned_kind == "categorical"


def test_type_schema_uses_full_training_partition_before_target_filtering():
    train = pd.DataFrame({
        "x": pd.Series([0, 1, 2, 3, "malformed"], dtype=object),
        "y": [0.0, 1.0, 2.0, 3.0, np.nan],
    })
    schema = resolve_feature_schema(train, ["x"])
    candidates = generate_candidate_macros(train, ["x"], max_states=3)
    fit = fit_linear_model(train, ["x"], "y")

    assert schema.kind("x") == fit.encoder.schema.kind("x") == "categorical"
    assert candidates == []
    assert fit.encoder.categories["x"] == ("0", "1", "2", "3")
    _, diagnostics = fit.encoder.transform_with_report(pd.DataFrame({"x": ["malformed"]}))
    assert diagnostics["x"]["unknown_category"] == 1
