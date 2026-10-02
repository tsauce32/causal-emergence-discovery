"""Regression tests for untouched selection and frozen regression preprocessing."""

import numpy as np
import pandas as pd
import pytest

from causal_emergence_discovery.discovery import DiscoveryConfig, run_discovery
from causal_emergence_discovery.discovery import _find_macro_for_record
from causal_emergence_discovery.macro import composite_quantile_macro, quantile_macro
from causal_emergence_discovery.models import fit_linear_model, predict_linear_model, predict_r2
from causal_emergence_discovery.panel import build_lagged_table
from causal_emergence_discovery.spec import StudySpec
from causal_emergence_discovery.validation import build_validation_plan


def panel():
    rng = np.random.default_rng(75)
    rows = []
    for entity in range(24):
        previous_y = rng.normal()
        for time in range(18):
            x = rng.normal()
            rows.append(dict(entity=entity, time=time, x=x, y=previous_y))
            previous_y = 2 * x + rng.normal(scale=0.3)
    return pd.DataFrame(rows)


def spec():
    return StudySpec.from_dict({
        "dataset": {"id_column": "entity", "time_column": "time"}, "outcomes": ["y"],
        "columns": {"x": "context", "y": {"role": "outcome", "allowed_as_cause": False, "allowed_as_adjustment": False}},
    })


@pytest.mark.parametrize("mode", ["entity_holdout", "forward_time", "within_entity_interpolation"])
def test_outer_outcome_changes_cannot_change_selection(mode):
    df = panel()
    study = spec()
    config = DiscoveryConfig(outcome="y", validation_mode=mode, folds=2, max_states=3, paths=2, seed=1)
    lagged = build_lagged_table(df, study)
    plan = build_validation_plan(lagged.data, id_column="entity", time_column="time", target_time_column=lagged.target_time_column,
                                 mode=mode, folds=2, seed=1)
    test = lagged.data.iloc[list(plan.outer.test_indices)]
    # Change just the source observations of outer test lead outcomes.
    keys = set(zip(test.entity, test[lagged.target_time_column]))
    changed = df.copy()
    for row_index, row in changed.iterrows():
        if (row.entity, row.time) in keys:
            changed.loc[row_index, "y"] = row.y * -20 + row.time * 13
    original_result = run_discovery(df, study, config)
    changed_result = run_discovery(changed, study, config)
    assert original_result["top_macros"] == changed_result["top_macros"]
    assert original_result["searches"] == changed_result["searches"]
    assert original_result["candidate_pathways"] == changed_result["candidate_pathways"]
    assert original_result["outer_evaluation"]["macro_r2"] != changed_result["outer_evaluation"]["macro_r2"]
    assert all(m["evaluation_scope"] == "development_selection_only" for m in original_result["top_macros"])


def test_prediction_uses_training_median_and_categories():
    train = pd.DataFrame({"x": [0., 1., 2., 3.], "category": ["a", "a", "b", "b"], "y": [0., 2., 4., 6.]})
    fit = fit_linear_model(train, ["x", "category"], "y")
    test = pd.DataFrame({"x": [np.nan, 1000., -1000.], "category": ["new", "a", "b"]})
    single = predict_linear_model(fit, test.iloc[:1])
    together = predict_linear_model(fit, test)
    assert single[0] == pytest.approx(together[0])
    assert fit.encoder.numeric_fills["x"] == 1.5
    assert fit.encoder.categories["category"] == ("a", "b")
    assert np.isfinite(together).all()


def test_invalid_holdout_target_has_no_fabricated_score():
    train = pd.DataFrame({"x": [0., 1., 2.], "y": [0., 2., 4.]})
    fit = fit_linear_model(train, ["x"], "y")
    with pytest.raises(ValueError, match="constant target"):
        predict_r2(fit, pd.DataFrame({"x": [3., 4.], "y": [1., 1.]}), ["x"], "y")
    with pytest.raises(ValueError, match="two finite"):
        predict_r2(fit, pd.DataFrame({"x": [3., 4.], "y": [1., np.inf]}), ["x"], "y")


def test_discovery_explicitly_rejects_insufficient_panel():
    with pytest.raises(ValueError):
        run_discovery(panel().query("entity < 2"), spec(), DiscoveryConfig(folds=5))


@pytest.mark.parametrize("reserved", ["y__lead1", "__lead_time", "__ced_macro_state"])
def test_input_cannot_reintroduce_generated_target_as_a_predictor(reserved):
    contaminated = panel().assign(**{reserved: 123.0})
    with pytest.raises(ValueError, match="collide with generated"):
        run_discovery(contaminated, spec(), DiscoveryConfig(folds=2))


def test_locked_recipe_is_recovered_from_its_search_when_names_collide():
    frame = pd.DataFrame({"composite": [0., 1., 2., 3., 4., 5.], "z": [9., 0., 9., 0., 9., 0.]})
    single = quantile_macro(frame, "composite", 2)
    composite = composite_quantile_macro(frame, ["composite", "z"], 2)
    assert single.name == composite.name
    first = {"macro_name": single.name, "method": single.method}
    selected = {"macro_name": composite.name, "method": composite.method}
    assert _find_macro_for_record([single, composite], [{"best": first}, {"best": selected}], selected) is composite


def test_lagged_numeric_string_times_are_sorted_numerically():
    frame = pd.DataFrame({"entity": [1] * 4, "time": ["10", "1", "20", "2"], "x": [0.] * 4, "y": [10., 1., 20., 2.]})
    lagged = build_lagged_table(frame, spec())
    assert lagged.data.time.tolist() == ["1", "2", "10"]
    assert lagged.data["y__lead1"].tolist() == [2., 10., 20.]


def test_lagged_integer_timestamp_metadata_preserves_precision():
    frame = panel()
    frame["time"] = frame["time"] + 10 ** 18
    lagged = build_lagged_table(frame, spec())
    assert int(lagged.data[lagged.target_time_column].iloc[0]) == 10 ** 18 + 1
    plan = build_validation_plan(lagged.data, id_column="entity", time_column="time", target_time_column=lagged.target_time_column,
                                 mode="forward_time", folds=2)
    assert plan.outer.audit["train_target_time_max"] < plan.outer.audit["test_predictor_time_min"]
