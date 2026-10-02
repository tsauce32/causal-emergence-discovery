import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.pipeline import runner


def test_native_adjustment_must_match_joint_estimand_and_coefficients():
    paths = [{"intervention": "a", "coefficient": 1.7}, {"intervention": "b", "coefficient": -.9}]
    joint = {"a": 1.7, "b": -.9}
    assert runner._validate_package_adjustment({"estimand": "joint_conditional", "pathways": paths}, joint, ["a", "b"])["validated"]
    assert not runner._validate_package_adjustment({"estimand": "marginal", "pathways": paths}, joint, ["a", "b"])["validated"]
    paths[0]["coefficient"] = 1.08
    assert not runner._validate_package_adjustment({"estimand": "joint_conditional", "pathways": paths}, joint, ["a", "b"])["validated"]


def test_protocol_hash_uses_frozen_json_content():
    a = {"cells": [{"case": "null", "split": "entity_holdout"}], "seed": [1, 2]}
    b = {"seed": [1, 2], "cells": [{"split": "entity_holdout", "case": "null"}]}
    assert runner.protocol_hash(a) == runner.protocol_hash(b)
    assert runner.protocol_hash(a) != runner.protocol_hash({**a, "seed": [1, 3]})


def test_source_pin_and_relevant_cleanliness_guard(monkeypatch, tmp_path):
    commit = "a" * 40
    def clean_git(_source, *args):
        if args == ("rev-parse", "HEAD"):
            return commit
        return ""
    monkeypatch.setattr(runner, "_git", clean_git)
    result = runner.verify_source(tmp_path, commit)
    assert result["head"] == commit and result["relevant_source_clean"]

    def dirty_git(_source, *args):
        return commit if args == ("rev-parse", "HEAD") else " M src/causal_emergence_discovery/macro.py"
    monkeypatch.setattr(runner, "_git", dirty_git)
    with pytest.raises(RuntimeError, match="dirty"):
        runner.verify_source(tmp_path, commit)
    with pytest.raises(RuntimeError, match="does not match"):
        runner.verify_source(tmp_path, "b" * 40)


def test_strict_json_is_valid_json_and_never_emits_nonfinite_tokens():
    payload = runner._strict_json({"scores": [1.0, np.inf, np.nan], "nested": {"x": np.float64(2.0)}})
    parsed = json.loads(payload, parse_constant=lambda text: pytest.fail(f"invalid JSON constant {text}"))
    assert parsed == {"scores": [1.0, None, None], "nested": {"x": 2.0}}


def test_generator_labels_are_aligned_by_entity_and_time_after_sort_and_lag():
    class Dataset:
        prespecified_labels = np.array([10, 11, 20, 21])

    frame = pd.DataFrame({"entity_id": ["b", "b", "a", "a"], "time": [0, 1, 0, 1]})
    lagged = type("Lag", (), {"data": pd.DataFrame({"entity_id": ["a", "b"], "time": [0, 0]})})()
    result = runner._aligned_labels(Dataset(), lagged, frame, "macro",
                                    type("Spec", (), {"id_column": "entity_id", "time_column": "time"})())
    assert result.tolist() == [20, 10]


def test_trial_failure_is_retained_with_reason_and_all_metric_slots():
    record = runner.run_trial("not-a-case", "entity_holdout", 1, Path("."), {}, object())
    assert record["status"] == "failed"
    assert "Unknown DGP case" in record["failure_reason"]
    assert set(record["paired_r2"]) == {
        "selected_macro", "restricted_micro", "no_macro", "prespecified_macro",
        "flexible_micro", "matched_micro"}


def test_worker_task_preserves_stable_index_independent_of_completion_order(monkeypatch):
    monkeypatch.setattr(runner, "run_trial", lambda case, split, seed, *_:
                        {"case": case, "split": split, "seed": seed, "status": "success"})
    task = (7, "iid_null", "entity_holdout", 123)
    idx, rec = runner._worker_run_trial_local(task, Path("source"), {}, object())
    assert idx == rec["planned_index"] == 7
    assert rec["seed"] == 123

    runner._WORKER_SOURCE, runner._WORKER_PROFILE, runner._WORKER_DISCOVERY = Path("source"), {}, object()
    idx, rec = runner._worker_run_trial(task)
    assert idx == rec["planned_index"] == 7
