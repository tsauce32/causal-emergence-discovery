import json

import pytest

from benchmarks.pipeline.protocol import (
    CELLS,
    generate_seeds,
    get_profile,
    summarize_trials,
)


def test_profiles_are_frozen_serializable_and_have_expected_budget():
    full = get_profile("full")
    quick = get_profile("quick")

    assert len(full["cells"]) == len(CELLS) == 12
    assert full["replicates_per_cell"] == 32
    assert full["planned_runs"] == 384
    assert quick["replicates_per_cell"] == 2
    assert quick["planned_runs"] == 24
    assert [cell for cell in full["cells"] if cell["split"] == "interpolation"] == [
        {"case": "entity_fingerprint", "split": "interpolation"},
        {"case": "temporal_drift", "split": "interpolation"},
    ]
    assert full["max_states"] == 3
    assert full["inner_folds"] == 3
    assert full["lag"] == 1
    assert full["holdout_fraction"] == 0.25
    assert full["emergence_evidence_status"] == "not_assessed"
    assert full["bootstrap_replicates"] == 1999
    assert full["bootstrap_seed"] == 20261003
    json.dumps(full)


def test_seed_families_are_deterministic_and_quick_is_full_prefix():
    full_seeds = generate_seeds(get_profile("full"))
    assert full_seeds == generate_seeds(get_profile("full"))
    assert len(full_seeds) == len(set(full_seeds)) == 32
    assert generate_seeds(get_profile("quick")) == full_seeds[:2]


def _record(seed, status="success", delta=None):
    return {
        "case": "iid_null",
        "split": "entity_holdout",
        "seed": seed,
        "status": status,
        "paired_r2": {
            "selected_macro": 0.1 if delta is not None else None,
            "restricted_micro": 0.1 - delta if delta is not None else None,
        },
        "selected_family": "quantile" if status == "success" else None,
        "selected_state_count": 3 if status == "success" else None,
        "compression": 0.4 if status == "success" else None,
        "support": {
            "target_train_rows": 100,
            "target_train_entities": 8,
            "target_test_rows": 25,
            "target_test_entities": 2,
            "test_coverage": 0.95,
            "states": [{"state": 0, "train_rows": 30, "train_entities": 6}]
        } if status == "success" else None,
        "failure_reason": "refit unsupported" if status == "unsupported" else None,
    }


def test_summary_uses_replicates_for_mc_uncertainty_and_not_fold_sd():
    profile = get_profile("quick")
    seeds = generate_seeds(profile)
    records = [_record(seeds[0], delta=1.0), _record(seeds[1], delta=3.0)]
    summary = summarize_trials(records, profile)
    cell = summary["cells"]["iid_null::entity_holdout"]
    effect = cell["paired_r2_differences"]["selected_macro_minus_restricted_micro"]

    assert effect["n"] == 2
    assert effect["mean"] == 2.0
    assert effect["mc_se"] == pytest.approx(1.0)
    assert effect["ci95_t"] == pytest.approx([2 - 12.706205, 2 + 12.706205])
    assert effect["bootstrap_ci95"] == summarize_trials(records, profile)["cells"][
        "iid_null::entity_holdout"
    ]["paired_r2_differences"]["selected_macro_minus_restricted_micro"]["bootstrap_ci95"]
    assert effect["bootstrap_ci95"][0] >= 1.0
    assert effect["bootstrap_ci95"][1] <= 3.0
    raw_macro = cell["heldout_r2"]["selected_macro"]
    raw_restricted = cell["heldout_r2"]["restricted_micro"]
    assert raw_macro["mean"] == pytest.approx(0.1)
    assert raw_restricted["mean"] == pytest.approx(-1.9)
    assert raw_macro["n"] == raw_restricted["n"] == 2
    assert cell["oracle_micro"]["n"] == 0
    assert cell["oracle_micro"]["r2"]["mean"] is None
    assert effect["positive_rate"]["successes"] == 2
    assert effect["positive_rate"]["n"] == 2
    assert "fold_sd" not in effect
    assert summary["uncertainty_unit"] == "independent_complete_pipeline_replicate"
    assert summary["emergence_evidence_status"] == "not_assessed"


def test_failures_and_missing_trials_remain_in_planned_denominators():
    profile = get_profile("quick")
    seeds = generate_seeds(profile)
    summary = summarize_trials(
        [_record(seeds[0], status="success", delta=0.5), _record(seeds[1], status="unsupported")],
        profile,
    )
    cell = summary["cells"]["iid_null::entity_holdout"]
    effect = cell["paired_r2_differences"]["selected_macro_minus_restricted_micro"]

    assert cell["n_planned"] == 2
    assert cell["n_records"] == 2
    assert cell["n_success"] == 1
    assert cell["n_failed_or_unsupported"] == 1
    assert cell["failure_rate_planned"] == 0.5
    assert cell["failure_reasons"] == {"refit unsupported": 1}
    assert effect["n"] == 1
    assert effect["mc_se"] is None
    assert effect["ci95_t"] is None

    missing = summarize_trials([_record(seeds[0], delta=0.5)], profile)
    missing_cell = missing["cells"]["iid_null::entity_holdout"]
    assert missing_cell["n_missing_records"] == 1
    assert missing_cell["failure_rate_planned"] == 0


def test_summary_rejects_duplicate_and_unfrozen_seed_records():
    profile = get_profile("quick")
    seed = generate_seeds(profile)[0]
    with pytest.raises(ValueError, match="duplicate"):
        summarize_trials([_record(seed, delta=0.0), _record(seed, delta=0.0)], profile)
    with pytest.raises(ValueError, match="outside frozen"):
        summarize_trials([_record(123, delta=0.0)], profile)


def test_summary_reports_support_threshold_as_diagnostic():
    profile = get_profile("quick")
    seed = generate_seeds(profile)[0]
    record = _record(seed, delta=0.0)
    record["support"]["states"] = [{"state": 1, "train_rows": 19, "train_entities": 5}]
    cell = summarize_trials([record], profile)["cells"]["iid_null::entity_holdout"]
    support = cell["support_threshold"]

    assert support["n_below_threshold"] == 1
    assert support["minimum_training_rows"] == 20
    assert support["minimum_training_entities"] == 5
    assert support["role"] == "stress_diagnostic_not_universal_validity_rule"
    assert cell["target_support"]["target_train_rows"]["mean"] == 100
    assert cell["target_support"]["test_coverage"]["mean"] == 0.95
    assert cell["state_support_by_replicate"][0]["seed"] == seed


def test_cluster_bootstrap_summary_keeps_fixed_recipe_scope_explicit():
    profile = get_profile("quick")
    seed = generate_seeds(profile)[0]
    record = _record(seed, delta=0.0)
    record["truth_recovered"] = True
    record["adjustment_cluster_bootstrap"] = {
        "estimate": 1.4,
        "ci95": [1.1, 1.7],
        "true_effect": 1.5,
        "interval_scope": "prespecified_raw_U_adjustment_model_entity_bootstrap",
    }
    record["oracle_micro"] = {
        "features": ["generator-known interaction"],
        "label": "generator-known oracle basis; excluded from Discovery selection",
        "r2": 0.72,
    }
    cell = summarize_trials([record], profile)["cells"]["iid_null::entity_holdout"]

    assert cell["truth_recovery_rate"]["estimate"] == 1.0
    bootstrap = cell["adjustment_cluster_bootstrap"]
    assert bootstrap["estimate_across_independent_replicates"]["mean"] == 1.4
    assert bootstrap["constructed_scm_coefficient_interval_coverage"]["successes"] == 1
    assert bootstrap["interval_scope"] == "prespecified_raw_U_adjustment_model_entity_bootstrap"
    assert "not_nominal_coverage_evidence" in bootstrap["coverage_interpretation"]
    oracle = cell["oracle_micro"]
    assert oracle["n"] == 1
    assert oracle["r2"]["mean"] == 0.72
    assert oracle["label"] == "generator-known oracle basis; excluded from Discovery selection"

