"""Known-mechanism longitudinal data generators for pipeline falsification.

Every row contains predictors observed at time ``t``.  The generated outcome
at ``t + 1`` is computed from those predictors, so ``build_lagged_table(...,
lag=1)`` recovers the stated transition target.  Formula-derived oracle
columns are supplied for explicitly matched-reference analyses only and are
excluded from the ordinary StudySpec feature set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


CASES = (
    "iid_null",
    "additive_signal",
    "nonlinear_threshold",
    "sign_interaction",
    "entity_fingerprint",
    "temporal_drift",
    "rare_absent_states",
    "known_confounding",
    "correlated_treatments",
    "population_shift",
)


@dataclass(frozen=True)
class BenchmarkDataset:
    """Generated panel and its known data-generating mechanism."""

    frame: pd.DataFrame
    spec_dict: dict[str, Any]
    truth: dict[str, Any]
    oracle_features: list[str]
    prespecified_labels: np.ndarray
    case: str


def generate_case(
    case: str,
    seed: int,
    n_entities: int = 48,
    n_periods: int = 16,
    split_seed: int = 17,
    holdout_fraction: float = 0.25,
) -> BenchmarkDataset:
    """Generate one deterministic panel with a known transition mechanism.

    ``split_seed`` selects identities used by the rare/absent support control.
    Drift and population-shift boundaries are placed at the first predictor
    time in the implied forward holdout of lag-eligible rows (all but the last
    raw period).
    """
    if case not in CASES:
        raise ValueError(f"Unknown DGP case {case!r}; expected one of {CASES}.")
    if n_entities < 2 or n_periods < 3:
        raise ValueError("n_entities must be at least 2 and n_periods at least 3.")
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be strictly between 0 and 1.")

    rng = np.random.default_rng(seed)
    split_rng = np.random.default_rng(split_seed)
    shape = (n_entities, n_periods)
    x1 = rng.normal(size=shape)
    x2 = rng.normal(size=shape)
    x3 = rng.normal(size=shape)
    noise = rng.normal(size=shape)
    y = np.empty(shape, dtype=float)
    y[:, 0] = rng.normal(size=n_entities)  # initial condition, never a cause
    oracle_features: list[str] = []
    labels = np.zeros(shape, dtype=np.int64)
    columns: dict[str, np.ndarray] = {
        "x1": x1,
        "x2": x2,
        "x3": x3,
    }
    extra_columns: dict[str, np.ndarray] = {}
    treatments: list[str] = []
    adjustment: dict[str, Any] | None = None
    spec_exclude: list[str] = []
    truth: dict[str, Any] = {
        "case": case,
        "seed": int(seed),
        "target": "one-step-ahead conditional outcome y[t+1] from predictors observed at t",
        "initial_outcome_role": "initial condition only; forbidden as cause and adjustment",
        "noise": "independent standard-normal transition error unless otherwise stated",
    }

    eligible_periods = n_periods - 1
    boundary = eligible_periods - int(np.ceil(eligible_periods * holdout_fraction))
    boundary = min(max(boundary, 1), n_periods - 2)

    if case == "iid_null":
        y[:, 1:] = rng.normal(size=(n_entities, n_periods - 1))
        truth.update(mechanism="y[t+1] is IID noise independent of all predictors at t", effect=0.0)
        labels[:] = 0
    elif case == "additive_signal":
        y[:, 1:] = 1.5 * x1[:, :-1] - 0.8 * x2[:, :-1] + noise[:, :-1]
        truth.update(mechanism="y[t+1] = 1.5*x1[t] - 0.8*x2[t] + error", coefficients={"x1": 1.5, "x2": -0.8})
        labels = (x1 > 0).astype(np.int64)
    elif case == "nonlinear_threshold":
        regime = (x1 > 0).astype(np.int64)
        y[:, 1:] = 2.0 * regime[:, :-1] + noise[:, :-1]
        extra_columns["oracle_threshold"] = regime
        oracle_features.append("oracle_threshold")
        spec_exclude.append("oracle_threshold")
        truth.update(
            mechanism="y[t+1] = 2*I(x1[t] > 0) + error",
            effect="two-unit conditional mean difference across the known x1 threshold",
            matched_reference="oracle_threshold is formula-derived and excluded from ordinary discovery inputs",
            threshold=0.0,
        )
        labels = regime
    elif case == "sign_interaction":
        product = np.sign(x1) * np.sign(x2)
        y[:, 1:] = 1.5 * product[:, :-1] + noise[:, :-1]
        extra_columns["oracle_sign_product"] = product
        oracle_features.append("oracle_sign_product")
        spec_exclude.append("oracle_sign_product")
        truth.update(
            mechanism="y[t+1] = 1.5*sign(x1[t])*sign(x2[t]) + error",
            effect="conditional contrast between matching and opposing predictor signs",
            matched_reference="oracle_sign_product is formula-derived and excluded from ordinary discovery inputs",
        )
        labels = (product > 0).astype(np.int64)
    elif case == "entity_fingerprint":
        entity_effect = rng.normal(loc=0.0, scale=2.0, size=n_entities)
        marker_grid = np.broadcast_to(np.asarray([f"entity_{i:04d}" for i in range(n_entities)])[:, None], shape)
        extra_columns["entity_marker"] = marker_grid
        y[:, 1:] = entity_effect[:, None] + noise[:, :-1]
        truth.update(
            mechanism="y[e,t+1] = a_e + error, with independent random entity effects unrelated to numeric predictors",
            entity_effects={f"entity_{i:04d}": float(value) for i, value in enumerate(entity_effect)},
            entity_effect_sd=2.0,
            target_population="unseen entities receive new independent effects; categorical ID memorization cannot transfer",
        )
        labels = np.repeat(np.arange(n_entities, dtype=np.int64)[:, None], n_periods, axis=1)
    elif case == "temporal_drift":
        slope = np.where(np.arange(n_periods) < boundary, 1.5, -1.5)
        y[:, 1:] = slope[:-1][None, :] * x1[:, :-1] + noise[:, :-1]
        truth.update(
            mechanism="coefficient of x1 changes from +1.5 to -1.5 at the declared forward boundary",
            coefficients={"pre_boundary_x1": 1.5, "post_boundary_x1": -1.5},
            boundary_predictor_time=boundary,
            evaluation="rolling one-step prediction; transitions use the predictor at t and target y[t+1]",
        )
        labels = np.broadcast_to((np.arange(n_periods) >= boundary).astype(np.int64), shape).copy()
    elif case == "rare_absent_states":
        entity_order = split_rng.permutation(n_entities)
        entity_test_count = max(1, int(np.ceil(n_entities * holdout_fraction)))
        # Match validation._entity_outer: its held-out entities are the first
        # ceil(fraction * n) members of this seeded permutation.
        heldout_entities = [int(x) for x in entity_order[:entity_test_count]]
        rare_entities = {heldout_entities[0]}
        rare_entity = np.asarray([int(i in rare_entities) for i in range(n_entities)], dtype=np.int64)
        rare_tail = np.zeros(shape, dtype=np.int64)
        rare_tail_entity = min(rare_entities)
        rare_tail[rare_tail_entity, boundary] = 1  # one supported predictor in the forward holdout
        columns["rare_entity_state"] = np.repeat(rare_entity[:, None], n_periods, axis=1)
        columns["rare_tail_state"] = rare_tail
        y[:, 1:] = 0.5 * x1[:, :-1] + noise[:, :-1]
        truth.update(
            mechanism="ordinary signal plus sparse held-out support states; rare states do not alter the outcome",
            rare_entity_ids=sorted(rare_entities),
            rare_entity_row_support=int(rare_entity.sum() * n_periods),
            rare_entity_unit_support=int(rare_entity.sum()),
            rare_tail_predictor_time=boundary,
            rare_tail_row_support=1,
            rare_tail_unit_support=1,
            rare_tail_entity_id=f"e{rare_tail_entity:04d}",
            rare_entity_guaranteed_in_entity_holdout=bool(rare_tail_entity in heldout_entities),
            support_threshold_note="20 rows across at least 5 independent entities is a design-specific stress threshold from the study plan, not a universal rule",
            forward_split_boundary=boundary,
            expected_support_issue="the tail occurs only in a forward-held-out predictor period with a valid next outcome",
        )
        labels = rare_tail + 2 * np.repeat(rare_entity[:, None], n_periods, axis=1)
    elif case == "known_confounding":
        u = rng.normal(size=shape)
        treatment = u + 0.5 * rng.normal(size=shape)
        columns["u"] = u
        columns["treatment"] = treatment
        treatments = ["treatment"]
        adjustment = {"columns": ["u"], "rationale": "u is the generated common cause of treatment and next-period outcome"}
        y[:, 1:] = 1.5 * treatment[:, :-1] + 4.0 * u[:, :-1] + noise[:, :-1]
        truth.update(
            mechanism="treatment[t] = u[t] + 0.5*noise; y[t+1] = 1.5*treatment[t] + 4*u[t] + error",
            treatment_effect=1.5,
            confounder_effect=4.0,
            marginal_treatment_association="not the causal treatment coefficient because u is omitted",
            adjustment_set=["u"],
        )
        labels = (u > 0).astype(np.int64)
    elif case == "correlated_treatments":
        u = rng.normal(size=shape)
        shared_w = rng.normal(size=shape)
        treatment_a = u + 0.45 * shared_w + 0.3 * rng.normal(size=shape)
        treatment_b = 0.8 * u + 0.45 * shared_w + 0.3 * rng.normal(size=shape)
        columns.update({"u": u, "treatment_a": treatment_a, "treatment_b": treatment_b})
        treatments = ["treatment_a", "treatment_b"]
        adjustment = {"columns": ["u"], "rationale": "u is the declared adjustment covariate; the independent shared treatment shock induces residual treatment correlation but has no direct outcome effect"}
        y[:, 1:] = 1.7 * treatment_a[:, :-1] - 0.9 * treatment_b[:, :-1] + 1.2 * u[:, :-1] + noise[:, :-1]
        truth.update(
            mechanism="A[t] = U[t] + 0.45*W[t] + 0.3*error_a; B[t] = 0.8*U[t] + 0.45*W[t] + 0.3*error_b; y[t+1] = 1.7*A[t] - 0.9*B[t] + 1.2*U[t] + error_y",
            latent_shared_shock="W is independent of U and all treatment/outcome errors; it affects both treatments but has no direct effect on y",
            joint_conditional_effects={"treatment_a": 1.7, "treatment_b": -0.9},
            confounder_effect=1.2,
            estimand="joint conditional treatment coefficients given U",
            marginal_estimand_warning="separate single-treatment regressions adjusted for U omit the other declared treatment and do not recover its joint conditional coefficient",
            adjustment_set=["u"],
            treatment_correlation="shared W induces residual correlation after adjustment for U; expected residual correlation is about 0.69",
            interpretation_limit="software ground-truth control only; not a real-world causal claim",
        )
        labels = (treatment_a > treatment_b).astype(np.int64)
    elif case == "population_shift":
        post = np.broadcast_to((np.arange(n_periods) >= boundary)[None, :], shape)
        shifted_x1 = x1 + 1.5 * post
        columns["x1"] = shifted_x1
        y[:, 1:] = 1.5 * shifted_x1[:, :-1] + noise[:, :-1]
        truth.update(
            mechanism="stable y slope with x1 distribution shifted +1.5 at the forward boundary",
            coefficients={"x1": 1.5},
            mean_shift={"x1": 1.5},
            boundary_predictor_time=boundary,
            evaluation="forward-time target-population shift",
        )
        labels = post.astype(np.int64)

    frame_data: dict[str, Any] = {
        "entity_id": np.repeat([f"e{i:04d}" for i in range(n_entities)], n_periods),
        "time": np.tile(np.arange(n_periods, dtype=np.int64), n_entities),
    }
    flat_columns: dict[str, np.ndarray] = {}
    for name, values in columns.items():
        flat_columns[name] = values.reshape(-1)
    for name, values in extra_columns.items():
        flat_columns[name] = values.reshape(-1)
    if "entity_marker" not in flat_columns and case == "entity_fingerprint":
        marker_grid = np.broadcast_to(np.asarray([f"entity_{i:04d}" for i in range(n_entities)])[:, None], shape)
        flat_columns["entity_marker"] = marker_grid.reshape(-1)
    flat_columns["y"] = y.reshape(-1)
    for name, values in flat_columns.items():
        frame_data[name] = values
    frame = pd.DataFrame(frame_data)

    column_specs: dict[str, dict[str, Any]] = {
        "entity_id": {"role": "exclude", "type": "categorical", "allowed_as_cause": False, "allowed_as_adjustment": False},
        "time": {"role": "exclude", "type": "numeric", "allowed_as_cause": False, "allowed_as_adjustment": False},
        "y": {"role": "outcome", "type": "numeric", "allowed_as_cause": False, "allowed_as_effect": True, "allowed_as_adjustment": False},
    }
    if case == "entity_fingerprint":
        column_specs["entity_marker"] = {"role": "covariate", "type": "categorical", "allowed_as_cause": True, "allowed_as_adjustment": False, "tags": ["stable_entity_fingerprint_marker"]}
    for name in flat_columns:
        if name in {"entity_marker", "y"}:
            continue
        role = "intervention" if name in treatments else "covariate"
        column_specs[name] = {"role": role, "type": "numeric", "allowed_as_cause": True, "allowed_as_adjustment": name not in treatments}
    for name in oracle_features:
        column_specs[name] = {"role": "exclude", "type": "numeric", "allowed_as_cause": False, "allowed_as_adjustment": False, "tags": ["oracle_formula_feature"]}
    for name in spec_exclude:
        if name not in column_specs:
            raise AssertionError(f"Excluded feature {name!r} has no schema declaration.")

    spec_dict: dict[str, Any] = {
        "dataset": {"id_column": "entity_id", "time_column": "time"},
        "columns": column_specs,
        "outcomes": ["y"],
        "interventions": treatments,
        "exclude": ["entity_id", "time", *spec_exclude],
        "metadata": {
            "benchmark_case": case,
            "formula_oracle_features": list(oracle_features),
            "split_seed": int(split_seed),
            "holdout_fraction": float(holdout_fraction),
            "temporal_boundary": boundary if case in {"temporal_drift", "population_shift", "rare_absent_states"} else None,
        },
    }
    if adjustment is not None:
        spec_dict["adjustment"] = adjustment

    truth["oracle_features"] = list(oracle_features)
    truth["ordinary_predictors"] = [name for name in flat_columns if name not in oracle_features and name != "y"]
    truth["split_seed"] = int(split_seed)
    truth["holdout_fraction"] = float(holdout_fraction)
    truth["n_entities"] = int(n_entities)
    truth["n_periods"] = int(n_periods)
    raw_labels = labels.reshape(-1)
    eligible_labels = labels[:, :-1].reshape(-1)
    expected_states = range(4) if case == "rare_absent_states" else range(2)
    observed_states = set(int(state) for state in np.unique(raw_labels))
    truth["prespecified_state_support"] = {
        str(int(state)): {
            "raw_rows": int(np.sum(raw_labels == state)),
            "raw_entities": int(np.sum(np.any(labels == state, axis=1))),
            "lag_eligible_rows": int(np.sum(eligible_labels == state)),
            "lag_eligible_entities": int(np.sum(np.any(labels[:, :-1] == state, axis=1))),
        }
        for state in sorted(observed_states | set(expected_states))
    }
    return BenchmarkDataset(
        frame=frame,
        spec_dict=spec_dict,
        truth=truth,
        oracle_features=oracle_features,
        prespecified_labels=raw_labels.copy(),
        case=case,
    )
