"""Reproduce entity-fingerprint and time-drift validation controls.

Run from the checkout with the package on PYTHONPATH::

    python examples/audit_validation_controls.py
    python examples/audit_validation_controls.py --output path/to/results.json

The default output is ``examples/validation-controls.generated.json``.
The two data-generating processes are diagnostic controls,
not estimates for an applied dataset. Legacy row-modulo scores estimate
interpolation among observed rows; entity holdout estimates prediction for unseen
entities; forward-time holdout estimates prediction at later observed times.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from causal_emergence_discovery.models import fit_linear_model, predict_r2
from causal_emergence_discovery.panel import build_lagged_table
from causal_emergence_discovery.spec import StudySpec
from causal_emergence_discovery.validation import DataSplit, ValidationPlan, build_validation_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("validation-controls.generated.json"),
        help="JSON output path (default: examples/validation-controls.generated.json).",
    )
    args = parser.parse_args()
    result = run_controls()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "scores": {
            name: {
                "row_modulo_mean_r2": value["row_modulo_baseline"]["mean_r2"],
                "planned_inner_mean_r2": value["planned_inner_folds"]["mean_r2"],
                "planned_outer_r2": value["planned_outer_holdout"]["r2"],
            }
            for name, value in result["controls"].items()
        },
    }, indent=2))


def run_controls() -> dict[str, Any]:
    rng = np.random.default_rng(123)
    spec = StudySpec.from_dict(
        {
            "dataset": {"id_column": "entity", "time_column": "time"},
            "outcomes": ["y"],
            "columns": {"tag": {"role": "individual"}, "y": {"role": "outcome"}},
        }
    )

    entity_rows = []
    for entity in range(100):
        baseline = rng.normal(0, 3.0)
        for time in range(10):
            entity_rows.append(
                {
                    "entity": entity,
                    "time": time,
                    "tag": f"entity_{entity:03d}",
                    "y": baseline + rng.normal(0, 0.1),
                }
            )
    entity_frame = build_lagged_table(
        pd.DataFrame(entity_rows), spec, outcomes=["y"], lag=1
    ).data
    target = "y__lead1"
    entity_plan = build_validation_plan(
        entity_frame,
        id_column="entity",
        time_column="time",
        target_time_column="__lead_time",
        mode="entity_holdout",
        folds=5,
        holdout_fraction=0.2,
        seed=0,
    )

    drift_rows = []
    for entity in range(100):
        baseline = rng.normal(0, 1.0)
        for time in range(10):
            drift_rows.append(
                {
                    "entity": entity,
                    "time": time,
                    "tag": f"entity_{entity:03d}",
                    "y": baseline + 0.4 * time + rng.normal(0, 0.04),
                }
            )
    drift_frame = build_lagged_table(
        pd.DataFrame(drift_rows), spec, outcomes=["y"], lag=1
    ).data
    forward_plan = build_validation_plan(
        drift_frame,
        id_column="entity",
        time_column="time",
        target_time_column="__lead_time",
        mode="forward_time",
        # Ten periods support two expanding windows after boundary purging.
        folds=2,
        holdout_fraction=0.2,
        seed=0,
    )

    entity_result = _scenario_result(entity_frame, target, entity_plan)
    drift_result = _scenario_result(drift_frame, target, forward_plan)
    return {
        "description": (
            "Synthetic adversarial validation controls. Scores answer different prediction questions "
            "and must not be compared as interchangeable estimates."
        ),
        "controls": {
            "entity_fingerprint": entity_result,
            "temporal_drift": drift_result,
        },
        "preprocessing": "Every fitted fold estimates medians and categorical levels on training rows only; test transforms use the frozen training encoder.",
        "score": "R2 is calculated on the test target values; negative scores are preserved.",
    }


def _scenario_result(frame: pd.DataFrame, target: str, plan: ValidationPlan) -> dict[str, Any]:
    row_modulo = _row_modulo_scores(frame, ["tag"], target, folds=5)
    development = frame.iloc[list(plan.outer.train_indices)].reset_index(drop=True)
    fold_evaluations = [
        {
            "name": split.name,
            "r2": _score_split(development, split, ["tag"], target),
            "boundary_and_coverage": split.audit,
        }
        for split in plan.inner
    ]
    fold_scores = [item["r2"] for item in fold_evaluations]
    outer_score = _score_indices(frame, plan.outer, ["tag"], target)
    purged_rows = [
        {
            "row_index": int(index),
            "entity": _json_value(frame.iloc[index]["entity"]),
            "predictor_time": _json_value(frame.iloc[index]["time"]),
            "target_time": _json_value(frame.iloc[index]["__lead_time"]),
        }
        for index in plan.outer.purged_indices
    ]
    return {
        "estimands": {
            "row_modulo": "Row-position modulo interpolation where the same entities can appear in train and test.",
            "planned_inner_folds": _estimand(plan.mode),
            "planned_outer_holdout": _estimand(plan.mode),
        },
        "row_modulo_baseline": row_modulo,
        "planned_inner_folds": {
            "r2": fold_scores,
            "mean_r2": float(np.mean(fold_scores)),
            "folds": fold_evaluations,
        },
        "planned_outer_holdout": {
            "r2": outer_score,
            "boundary_and_coverage": plan.audit["outer"],
            "purged_rows": purged_rows,
            "dropped_rows": len(plan.outer.dropped_indices),
            **(
                {
                    "train_entities": sorted(
                        frame.iloc[list(plan.outer.train_indices)]["entity"].unique().tolist()
                    ),
                    "test_entities": sorted(
                        frame.iloc[list(plan.outer.test_indices)]["entity"].unique().tolist()
                    ),
                }
                if plan.mode == "entity_holdout"
                else {}
            ),
        },
        "validation_mode": plan.mode,
    }


def _row_modulo_scores(
    frame: pd.DataFrame, features: list[str], target: str, folds: int
) -> dict[str, Any]:
    scores = []
    indices = np.arange(len(frame))
    for fold in range(folds):
        test = indices[indices % folds == fold]
        train = indices[indices % folds != fold]
        fit = fit_linear_model(frame.iloc[train].reset_index(drop=True), features, target)
        score = predict_r2(fit, frame.iloc[test].reset_index(drop=True), features, target)
        scores.append(float(score))
    return {"fold_r2": scores, "mean_r2": float(np.mean(scores)), "folds": folds}


def _score_split(frame: pd.DataFrame, split: DataSplit, features: list[str], target: str) -> float:
    train = frame.iloc[list(split.train_indices)].reset_index(drop=True)
    test = frame.iloc[list(split.test_indices)].reset_index(drop=True)
    fit = fit_linear_model(train, features, target)
    return float(predict_r2(fit, test, features, target))


def _score_indices(frame: pd.DataFrame, split: DataSplit, features: list[str], target: str) -> float:
    fit = fit_linear_model(frame.iloc[list(split.train_indices)].reset_index(drop=True), features, target)
    return float(predict_r2(fit, frame.iloc[list(split.test_indices)].reset_index(drop=True), features, target))


def _estimand(mode: str) -> str:
    if mode == "entity_holdout":
        return "Prediction for entities excluded from model fitting and split selection."
    if mode == "forward_time":
        return "Prediction at future predictor timestamps with training lead outcomes purged at each forecast boundary."
    return "Interpolation for already observed entities; overlapping predictor-to-target windows are quarantined."


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    return value


if __name__ == "__main__":
    main()
