"""Frozen design and replicate-level summaries for the Discovery benchmark.

Every record represents one independent, complete pipeline run. Folds are part
of that run and are never treated as independent observations for uncertainty.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any

import numpy as np


PROTOCOL_VERSION = "selection-aware-discovery-v1"
SEED_FAMILY = 20261002
BOOTSTRAP_REPLICATES = 1999
BOOTSTRAP_SEED = 20261003

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

CELLS = tuple(
    {"case": case, "split": split}
    for case, split in (
        ("iid_null", "entity_holdout"),
        ("additive_signal", "entity_holdout"),
        ("nonlinear_threshold", "entity_holdout"),
        ("sign_interaction", "entity_holdout"),
        ("entity_fingerprint", "entity_holdout"),
        ("entity_fingerprint", "interpolation"),
        ("temporal_drift", "forward_time"),
        ("temporal_drift", "interpolation"),
        ("rare_absent_states", "entity_holdout"),
        ("known_confounding", "entity_holdout"),
        ("correlated_treatments", "entity_holdout"),
        ("population_shift", "forward_time"),
    )
)

PAIRED_R2_KEYS = (
    "selected_macro",
    "restricted_micro",
    "no_macro",
    "prespecified_macro",
    "flexible_micro",
    "matched_micro",
)


def get_profile(name: str = "full") -> dict[str, Any]:
    """Return a fresh JSON-serializable frozen profile (``full`` or ``quick``)."""
    normalized = name.strip().lower()
    if normalized not in {"full", "quick"}:
        raise ValueError("profile name must be 'full' or 'quick'")
    full = normalized == "full"
    return {
        "protocol_version": PROTOCOL_VERSION,
        "name": normalized,
        "seed_family": SEED_FAMILY,
        "replicates_per_cell": 32 if full else 2,
        "n_entities": 48 if full else 24,
        "n_periods": 16 if full else 12,
        "inner_folds": 3 if full else 2,
        "lag": 1,
        "holdout_fraction": 0.25,
        "split_seed": 17,
        "max_states": 3,
        "candidate_paths": 2 if full else 1,
        "branching": 2,
        "min_state_train_rows": 20,
        "min_state_train_entities": 5,
        "state_support_threshold_role": "stress_diagnostic_not_universal_validity_rule",
        "outer_evaluation_per_run": 1,
        "retry_failed_seed": False,
        "uncertainty_unit": "independent_complete_pipeline_replicate",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "emergence_evidence_status": "not_assessed",
        "cells": deepcopy(list(CELLS)),
        "planned_runs": (32 if full else 2) * len(CELLS),
    }


def generate_seeds(profile: dict[str, Any]) -> list[int]:
    """Generate the frozen independent replicate seeds from a SeedSequence.

    Full and quick profiles share the same seed family, so quick seeds are the
    first two seeds of the full profile. A failed run is retained as a failure;
    no replacement seed is generated.
    """
    count = int(profile["replicates_per_cell"])
    entropy = int(profile["seed_family"])
    root = np.random.SeedSequence(entropy)
    children = root.spawn(count)
    seeds = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    if len(set(seeds)) != count:
        raise RuntimeError("SeedSequence unexpectedly produced duplicate replicate seeds")
    return seeds


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _mean_ci(
    values: list[float], *, bootstrap_replicates: int = 1999, bootstrap_seed: int = 0
) -> dict[str, Any]:
    n = len(values)
    if n == 0:
        return {
            "n": 0, "mean": None, "mc_se": None, "ci95_t": None,
            "bootstrap_ci95": None,
        }
    mean = math.fsum(values) / n
    if n == 1:
        return {
            "n": 1, "mean": mean, "mc_se": None, "ci95_t": None,
            "bootstrap_ci95": None,
        }
    variance = math.fsum((value - mean) ** 2 for value in values) / (n - 1)
    mc_se = math.sqrt(variance / n)
    critical = _t_critical_975(n - 1)
    half_width = critical * mc_se
    rng = np.random.default_rng(int(bootstrap_seed))
    sample = np.asarray(values, dtype=float)
    indices = rng.integers(0, n, size=(int(bootstrap_replicates), n))
    bootstrap_means = sample[indices].mean(axis=1)
    return {
        "n": n,
        "mean": mean,
        "mc_se": mc_se,
        "ci95_t": [mean - half_width, mean + half_width],
        "bootstrap_ci95": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
    }


_T975 = {
    1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582,
    6: 2.446912, 7: 2.364624, 8: 2.306004, 9: 2.262157, 10: 2.228139,
    11: 2.200985, 12: 2.178813, 13: 2.160369, 14: 2.144787, 15: 2.131450,
    16: 2.119905, 17: 2.109816, 18: 2.100922, 19: 2.093024, 20: 2.085963,
    21: 2.079614, 22: 2.073873, 23: 2.068658, 24: 2.063899, 25: 2.059539,
    26: 2.055529, 27: 2.051831, 28: 2.048407, 29: 2.045230, 30: 2.042272,
}


def _t_critical_975(df: int) -> float:
    """Two-sided 95% Student-t critical value, with a close large-df expansion."""
    if df in _T975:
        return _T975[df]
    z = 1.959963984540054
    d = float(df)
    return z + (z**3 + z) / (4 * d) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * d**2)


def _wilson(successes: int, n: int) -> dict[str, Any]:
    if n == 0:
        return {"n": 0, "successes": 0, "estimate": None, "ci95": None}
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {
        "n": n,
        "successes": successes,
        "estimate": p,
        "ci95": [max(0.0, center - radius), min(1.0, center + radius)],
    }


def _cell_key(case: str, split: str) -> str:
    return f"{case}::{split}"


def summarize_trials(records: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, Any]:
    """Summarize complete-pipeline trials by frozen cell.

    ``records`` use one row per planned seed/cell. Required keys are ``case``,
    ``split``, ``seed``, ``status`` and ``paired_r2``. Numerical effects use
    only successful runs with finite paired scores, while failure and support
    fractions retain the full planned denominator.
    """
    cells = {_cell_key(cell["case"], cell["split"]): cell for cell in profile["cells"]}
    planned = int(profile["replicates_per_cell"])
    expected_seeds = generate_seeds(profile)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, int]] = set()
    for record in records:
        case, split, seed = str(record["case"]), str(record["split"]), int(record["seed"])
        key = _cell_key(case, split)
        if key not in cells:
            raise ValueError(f"record belongs to an unfrozen cell: {key}")
        identity = (case, split, seed)
        if identity in seen:
            raise ValueError(f"duplicate replicate record: {identity}")
        seen.add(identity)
        if seed not in expected_seeds:
            raise ValueError(f"seed is outside frozen profile seed family: {seed}")
        grouped[key].append(record)

    summaries: dict[str, Any] = {}
    for key, cell in cells.items():
        rows = grouped.get(key, [])
        status_counts = Counter(str(row.get("status", "missing_status")) for row in rows)
        successful = [row for row in rows if row.get("status") == "success"]
        paired: dict[str, list[float]] = {metric: [] for metric in (
            "selected_macro_minus_restricted_micro",
            "selected_macro_minus_no_macro",
            "selected_macro_minus_prespecified_macro",
            "selected_macro_minus_flexible_micro",
            "selected_macro_minus_matched_micro",
        )}
        heldout_scores: dict[str, list[float]] = {metric: [] for metric in PAIRED_R2_KEYS}
        selection_scores: list[float] = []
        compressions: list[float] = []
        adjustment_estimates: list[float] = []
        adjustment_bootstrap_coverage: list[bool] = []
        truth_recovered_values: list[bool] = []
        oracle_micro_r2: list[float] = []
        family_counts: Counter[str] = Counter()
        state_counts: Counter[str] = Counter()
        selections_n = 0
        support_observed = 0
        unsupported_support = 0
        support_observations: list[dict[str, Any]] = []
        target_support_values: dict[str, list[float]] = defaultdict(list)
        threshold_rows = int(profile["min_state_train_rows"])
        threshold_entities = int(profile["min_state_train_entities"])
        for row in successful:
            scores = row.get("paired_r2") or {}
            for metric in PAIRED_R2_KEYS:
                score = _finite_number(scores.get(metric))
                if score is not None:
                    heldout_scores[metric].append(score)
            macro = _finite_number(scores.get("selected_macro"))
            if macro is not None:
                for key_name, reference in (
                    ("selected_macro_minus_restricted_micro", "restricted_micro"),
                    ("selected_macro_minus_no_macro", "no_macro"),
                    ("selected_macro_minus_prespecified_macro", "prespecified_macro"),
                    ("selected_macro_minus_flexible_micro", "flexible_micro"),
                    ("selected_macro_minus_matched_micro", "matched_micro"),
                ):
                    comparator = _finite_number(scores.get(reference))
                    if comparator is not None:
                        paired[key_name].append(macro - comparator)
            selection_score = _finite_number(row.get("selection_score"))
            if selection_score is not None:
                selection_scores.append(selection_score)
            compression = _finite_number(row.get("compression"))
            if compression is not None:
                compressions.append(compression)
            recovered = row.get("truth_recovered")
            if isinstance(recovered, bool):
                truth_recovered_values.append(recovered)
            oracle_micro = row.get("oracle_micro")
            if isinstance(oracle_micro, dict):
                oracle_score = _finite_number(oracle_micro.get("r2"))
                if oracle_score is not None:
                    oracle_micro_r2.append(oracle_score)
            cluster_bootstrap = row.get("adjustment_cluster_bootstrap")
            if isinstance(cluster_bootstrap, dict):
                estimate = _finite_number(cluster_bootstrap.get("estimate"))
                if estimate is not None:
                    adjustment_estimates.append(estimate)
                interval = cluster_bootstrap.get("ci95")
                true_effect = _finite_number(cluster_bootstrap.get("true_effect"))
                if (
                    isinstance(interval, (list, tuple))
                    and len(interval) == 2
                    and true_effect is not None
                ):
                    low, high = _finite_number(interval[0]), _finite_number(interval[1])
                    if low is not None and high is not None:
                        adjustment_bootstrap_coverage.append(low <= true_effect <= high)
            family = row.get("selected_family")
            state_count = row.get("selected_state_count")
            if family is not None:
                family_counts[str(family)] += 1
                selections_n += 1
            if state_count is not None:
                state_counts[str(state_count)] += 1
            support = row.get("support")
            if isinstance(support, dict):
                support_observations.append({
                    "seed": int(row["seed"]),
                    "states": deepcopy(support.get("states"))
                    if isinstance(support.get("states"), list) else None,
                })
                for support_key in (
                    "target_train_rows",
                    "target_train_entities",
                    "target_test_rows",
                    "target_test_entities",
                    "test_coverage",
                ):
                    support_value = _finite_number(support.get(support_key))
                    if support_value is not None:
                        target_support_values[support_key].append(support_value)
                state_support = support.get("states")
                if isinstance(state_support, list):
                    support_observed += 1
                    if any(
                        int(item.get("train_rows", 0)) < threshold_rows
                        or int(item.get("train_entities", 0)) < threshold_entities
                        for item in state_support
                        if isinstance(item, dict)
                    ):
                        unsupported_support += 1
        bootstrap_root = int(profile.get("bootstrap_seed", BOOTSTRAP_SEED))
        bootstrap_replicates = int(profile.get("bootstrap_replicates", BOOTSTRAP_REPLICATES))
        cell_index = list(cells).index(key)

        def replicate_mean(values: list[float], stream: int) -> dict[str, Any]:
            seed = int(np.random.SeedSequence(
                [bootstrap_root, cell_index, stream]
            ).generate_state(1, dtype=np.uint32)[0])
            return _mean_ci(
                values,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=seed,
            )

        paired_summary = {}
        for metric_index, (metric, values) in enumerate(paired.items()):
            bootstrap_seed = int(np.random.SeedSequence(
                [bootstrap_root, cell_index, metric_index]
            ).generate_state(1, dtype=np.uint32)[0])
            effect = _mean_ci(
                values,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
            )
            effect["positive_rate"] = _wilson(sum(value > 0 for value in values), len(values))
            paired_summary[metric] = effect
        raw_score_summary = {}
        for metric_index, (metric, values) in enumerate(heldout_scores.items()):
            bootstrap_seed = int(np.random.SeedSequence(
                [bootstrap_root, cell_index, 100 + metric_index]
            ).generate_state(1, dtype=np.uint32)[0])
            raw_score_summary[metric] = _mean_ci(
                values,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
            )
        failure_count = sum(count for status, count in status_counts.items() if status != "success")
        summaries[key] = {
            "case": cell["case"],
            "split": cell["split"],
            "n_planned": planned,
            "n_records": len(rows),
            "n_missing_records": max(0, planned - len(rows)),
            "status_counts": dict(status_counts),
            "n_success": len(successful),
            "n_failed_or_unsupported": failure_count,
            "failure_rate_planned": failure_count / planned if planned else None,
            "paired_r2_differences": paired_summary,
            "heldout_r2": raw_score_summary,
            "oracle_micro": {
                "n": len(oracle_micro_r2),
                "r2": replicate_mean(oracle_micro_r2, 303),
                "label": "generator-known oracle basis; excluded from Discovery selection",
            },
            "selection_score": replicate_mean(selection_scores, 300),
            "compression": replicate_mean(compressions, 301),
            "truth_recovery_rate": _wilson(
                sum(truth_recovered_values), len(truth_recovered_values)
            ),
            "adjustment_cluster_bootstrap": {
                "estimate_across_independent_replicates": replicate_mean(adjustment_estimates, 302),
                "constructed_scm_coefficient_interval_coverage": _wilson(
                    sum(adjustment_bootstrap_coverage), len(adjustment_bootstrap_coverage)
                ),
                "interval_scope": "prespecified_raw_U_adjustment_model_entity_bootstrap",
                "coverage_interpretation": "descriptive_low_precision_at_32_full_replicates_not_nominal_coverage_evidence",
            },
            "selected_family_counts": dict(sorted(family_counts.items())),
            "n_selected_family_assignments": selections_n,
            "selected_family_frequency": {
                name: count / selections_n for name, count in sorted(family_counts.items())
            } if selections_n else {},
            "selected_state_count_counts": dict(sorted(state_counts.items())),
            "n_selected_state_count_assignments": sum(state_counts.values()),
            "support_threshold": {
                "minimum_training_rows": threshold_rows,
                "minimum_training_entities": threshold_entities,
                "role": profile["state_support_threshold_role"],
                "n_success_with_state_support": support_observed,
                "n_below_threshold": unsupported_support,
                "fraction_below_threshold_of_planned": unsupported_support / planned if planned else None,
            },
            "target_support": {
                support_key: replicate_mean(values, 200 + support_index)
                for support_index, (support_key, values) in enumerate(
                    sorted(target_support_values.items())
                )
            },
            "state_support_by_replicate": support_observations,
            "failure_reasons": dict(sorted(Counter(
                str(row.get("failure_reason")) for row in rows if row.get("failure_reason") is not None
            ).items())),
        }
    return {
        "protocol_version": profile["protocol_version"],
        "profile": profile["name"],
        "replicates_per_cell": planned,
        "n_cells": len(cells),
        "n_planned_runs": planned * len(cells),
        "n_records": len(records),
        "uncertainty_unit": profile["uncertainty_unit"],
        "emergence_evidence_status": "not_assessed",
        "cells": summaries,
    }

