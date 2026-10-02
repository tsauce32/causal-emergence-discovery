"""Selection-aware, outer-holdout benchmark for causal-emergence-discovery.

Example (reported run):
  python -m benchmarks.pipeline.runner --profile full --source PATH --expected-commit SHA --output results

``--development-smoke`` permits an unpinned quick run for development only;
its manifest is marked non-reportable. Full runs require a ready status file.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import baselines
from .dgps import generate_case
from .protocol import generate_seeds, get_profile, summarize_trials

EXPECTED_BASE = "5d6ab966deba6c705f4a869006b19b73736c9824"
_WORKER_DISCOVERY = None
_WORKER_SOURCE = None
_WORKER_PROFILE = None
HARNESS_FILES = ("benchmarks/pipeline/dgps.py", "benchmarks/pipeline/baselines.py",
                 "benchmarks/pipeline/protocol.py", "benchmarks/pipeline/runner.py",
                 "benchmarks/pipeline/PROTOCOL.md")


def _git(source: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(source), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def verify_source(source: Path, expected_commit: str, *, require_clean: bool = True) -> dict[str, Any]:
    source = source.resolve()
    actual = _git(source, "rev-parse", "HEAD")
    if actual.lower() != expected_commit.lower():
        raise RuntimeError(f"source HEAD {actual} does not match --expected-commit {expected_commit}")
    changed = _git(source, "status", "--porcelain", "--", "src/causal_emergence_discovery", "pyproject.toml")
    if require_clean and changed:
        raise RuntimeError("Discovery source files are dirty; refusing benchmark run: " + changed)
    return {"source": str(source), "head": actual, "expected_commit": expected_commit,
            "relevant_source_clean": not bool(changed), "relevant_status": changed}


def protocol_hash(profile: dict[str, Any]) -> str:
    frozen = json.dumps(profile, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(frozen).hexdigest()


def harness_fingerprints() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in HARNESS_FILES}


def _load_package(source: Path):
    package_root = (source / "src").resolve()
    if not (package_root / "causal_emergence_discovery" / "__init__.py").is_file():
        raise RuntimeError(f"Discovery package not found under {package_root}")
    sys.path.insert(0, str(package_root))
    for key in list(sys.modules):
        if key == "causal_emergence_discovery" or key.startswith("causal_emergence_discovery."):
            del sys.modules[key]
    discovery = importlib.import_module("causal_emergence_discovery.discovery")
    package_file = Path(discovery.__file__).resolve()
    if package_root not in package_file.parents:
        raise RuntimeError(f"Imported Discovery from unexpected location: {package_file}")
    return discovery


def _jsonable(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [_jsonable(v) for v in value]
    if pd.isna(value):
        return None
    return str(value)


def _strict_json(value) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


def _aligned_labels(dataset, lagged, frame, name: str, spec):
    """Get DGP truth/declared labels, aligning original rows via stable row key."""
    if name in frame.columns:
        value = frame[name]
    else:
        labels = getattr(dataset, "prespecified_labels", None)
        if isinstance(labels, dict):
            value = labels.get(name)
        else:
            value = labels if name in {"macro", "prespecified_labels"} else None
    if value is None:
        return None
    if isinstance(value, str) and value in lagged.data:
        return lagged.data[value].reset_index(drop=True)
    if isinstance(value, pd.Series):
        if len(value) == len(lagged.data):
            return value.reset_index(drop=True)
        value = value.to_numpy()
    arr = np.asarray(value)
    if len(arr) == len(lagged.data):
        return pd.Series(arr)
    # build_lagged_table sorts and drops each entity's final lead. Align truth by
    # entity/time keys, not row positions.
    if len(arr) == len(frame):
        keys = list(zip(frame[spec.id_column].tolist(), frame[spec.time_column].tolist()))
        lookup = dict(zip(keys, arr.tolist()))
        lag_keys = zip(lagged.data[spec.id_column].tolist(), lagged.data[spec.time_column].tolist())
        return pd.Series([lookup[key] for key in lag_keys])
    return None


def run_trial(case: str, split: str, seed: int, source: Path, profile: dict[str, Any],
              discovery_module) -> dict[str, Any]:
    record = {"case": case, "split": split, "seed": int(seed), "status": "failed",
              "split_seed": int(profile.get("split_seed", 17)),
              "paired_r2": {key: None for key in ("selected_macro", "restricted_micro", "no_macro",
                  "prespecified_macro", "flexible_micro", "matched_micro")},
              "compression": None, "selected_family": None, "selected_state_count": None,
              "support": {}, "failure_reason": None}
    try:
        settings = profile
        data = generate_case(case, seed, n_entities=int(settings.get("n_entities", 48)),
                             n_periods=int(settings.get("n_periods", 16)),
                             split_seed=int(settings.get("split_seed", 17)),
                             holdout_fraction=float(settings.get("holdout_fraction", .25)))
        frame = data.frame.copy()
        from causal_emergence_discovery.spec import StudySpec
        from causal_emergence_discovery.panel import build_lagged_table
        from causal_emergence_discovery.macro import MacroAssignment, apply_macro

        spec = StudySpec.from_dict(data.spec_dict)
        outcomes = spec.outcome_columns(list(frame.columns))
        if len(outcomes) != 1:
            raise ValueError(f"DGP must declare exactly one outcome; got {outcomes}")
        outcome = outcomes[0]
        lag = int(settings.get("lag", 1))
        mode = {"interpolation": "within_entity_interpolation", "forward_time": "forward_time"}.get(split, "entity_holdout")
        config = discovery_module.DiscoveryConfig(
            outcome=outcome, lag=lag, max_states=int(settings.get("max_states", 3)),
            paths=int(settings.get("candidate_paths", 1)), branching_factor=int(settings.get("branching", 2)),
            folds=int(settings.get("inner_folds", 2)), top_k=1, validation_mode=mode,
            holdout_fraction=float(settings.get("holdout_fraction", .25)),
            seed=int(settings.get("split_seed", 17)))

        # This is the single full search/refit/outer evaluation for this generated dataset.
        result = discovery_module.run_discovery(frame, spec, config)
        lagged = build_lagged_table(frame, spec, outcomes=[outcome], lag=lag)
        target = lagged.target_columns[outcome]
        outer = result["validation"]["outer"]
        tr_i, te_i = outer["train_indices"], outer["test_indices"]
        train = lagged.data.iloc[tr_i].reset_index(drop=True)
        test = lagged.data.iloc[te_i].reset_index(drop=True)
        cols = result["columns"]
        micro_features = list(cols["micro_features"])
        interventions = list(cols["interventions"])
        raw_features = list(dict.fromkeys([*interventions, *micro_features]))
        selected = result["top_macros"][0]
        outer_metadata = result["outer_evaluation"]["encoder"]
        outer_encoder = outer_metadata
        if "recipe" not in outer_encoder and isinstance(outer_encoder.get("encoder"), dict):
            outer_encoder = outer_encoder["encoder"]
        recipe = outer_encoder.get("recipe")
        feature_columns = tuple(outer_encoder.get("columns", ()))
        macro = MacroAssignment(str(selected["macro_name"]), pd.Series(dtype=int),
            str(selected.get("method", recipe or "selected")), feature_columns,
            {"encoder": outer_encoder}, outer_encoder)
        train_macro, test_macro = apply_macro(macro, train), apply_macro(macro, test)
        macro_r2, _, y_test = baselines.fit_state_model(train, test, train_macro.labels,
                                                         test_macro.labels, target, extra_features=interventions)
        micro_r2 = baselines.fit_predict_r2(train, test, raw_features, target)[0]
        record["paired_r2"]["selected_macro"] = macro_r2
        record["paired_r2"]["restricted_micro"] = micro_r2
        record["paired_r2"]["no_macro"] = baselines.fit_predict_r2(train, test, [], target)[0]
        record["paired_r2"]["flexible_micro"] = baselines.fit_predict_r2(
            train, test, raw_features, target, polynomial_degree=2)[0]
        record["paired_r2"]["matched_micro"] = baselines.fit_matched_micro(
            train, test, train_macro.labels, test_macro.labels, target, extra_features=interventions)[0]

        labels = _aligned_labels(data, lagged, frame, "macro", spec)
        if labels is None and getattr(data, "prespecified_labels", None) is not None:
            labels = _aligned_labels(data, lagged, frame, "prespecified_labels", spec)
        if labels is not None:
            record["paired_r2"]["prespecified_macro"] = baselines.fit_state_model(
                train, test, labels.iloc[tr_i], labels.iloc[te_i], target,
                extra_features=interventions)[0]

        oracle_features = [name for name in getattr(data, "oracle_features", [])
                           if name in train.columns and name in test.columns]
        if oracle_features:
            record["oracle_micro"] = {"features": oracle_features,
                "label": "generator-known oracle basis; excluded from Discovery selection",
                "r2": baselines.fit_predict_r2(train, test,
                    [*interventions, *oracle_features], target)[0]}

        # Independently reproduce package-produced heldout metrics to detect alignment/API drift.
        pkg_eval = result["outer_evaluation"]
        discrepancies = {}
        for name, ours in (("macro_r2", macro_r2), ("micro_r2", micro_r2)):
            theirs = float(pkg_eval[name])
            discrepancies[name] = {"package": theirs, "independent": ours,
                                   "matches": bool(np.isclose(theirs, ours, rtol=1e-9, atol=1e-10))}
        record["package_reproduction"] = discrepancies
        record["metric_reproduction_ok"] = all(v["matches"] for v in discrepancies.values())
        if not record["metric_reproduction_ok"]:
            raise ValueError("independent outer R2 did not reproduce package result")
        primary_scores = [record["paired_r2"][key] for key in
            ("selected_macro", "restricted_micro", "no_macro", "flexible_micro", "matched_micro")]
        if not all(value is not None and np.isfinite(float(value)) for value in primary_scores):
            raise ValueError("one or more required benchmark R2 metrics are missing or nonfinite")
        selection_score = selected.get("ranking_score", selected.get("score"))
        if selection_score is None or not np.isfinite(float(selection_score)):
            raise ValueError("selected development score is nonfinite")

        train_labels = train_macro.labels.astype(str)
        test_labels = test_macro.labels.astype(str)
        state_support = []
        for state in sorted(set(train_labels)):
            tm = train_labels == state
            vm = test_labels == state
            state_support.append({"state": str(state),
                "train_rows": int(tm.sum()), "train_entities": int(train.loc[tm, spec.id_column].nunique()),
                "test_rows": int(vm.sum()), "test_entities": int(test.loc[vm, spec.id_column].nunique())})
        record["support"] = {"train_rows": len(train), "test_rows": len(test),
            "target_train_rows": int(pd.to_numeric(train[target], errors="coerce").replace([np.inf,-np.inf], np.nan).notna().sum()),
            "target_train_entities": int(train.loc[pd.to_numeric(train[target], errors="coerce").replace([np.inf,-np.inf], np.nan).notna(), spec.id_column].nunique()),
            "target_test_rows": int(pd.to_numeric(test[target], errors="coerce").replace([np.inf,-np.inf], np.nan).notna().sum()),
            "target_test_entities": int(test.loc[pd.to_numeric(test[target], errors="coerce").replace([np.inf,-np.inf], np.nan).notna(), spec.id_column].nunique()),
            "test_coverage": float(pd.to_numeric(test[target], errors="coerce").replace([np.inf,-np.inf], np.nan).notna().mean()),
            "train_entities": int(train[spec.id_column].nunique()),
            "test_entities": int(test[spec.id_column].nunique()),
            "entity_overlap": len(set(train[spec.id_column]) & set(test[spec.id_column])),
            "finite_train_targets": int(pd.to_numeric(train[target], errors="coerce").replace([np.inf,-np.inf], np.nan).notna().sum()),
            "finite_test_targets": int(pd.to_numeric(test[target], errors="coerce").replace([np.inf,-np.inf], np.nan).notna().sum()),
            "states": state_support,
            "unseen_test_state_rows": int((~test_labels.isin(set(train_labels))).sum()),
            "unsupported_train_states": [item["state"] for item in state_support
                if item["train_rows"] < int(settings.get("min_state_train_rows", 20))
                or item["train_entities"] < int(settings.get("min_state_train_entities", 5))]}
        if labels is not None:
            p_train = pd.Series(labels.iloc[tr_i].to_numpy()).astype("string")
            p_test = pd.Series(labels.iloc[te_i].to_numpy()).astype("string")
            record["support"]["prespecified_states"] = [
                {"state": state, "train_rows": int((p_train == state).sum()),
                 "train_entities": int(train.loc[p_train == state, spec.id_column].nunique()),
                 "test_rows": int((p_test == state).sum()),
                 "test_entities": int(test.loc[p_test == state, spec.id_column].nunique())}
                for state in sorted(set(p_train) | set(p_test))]
        raw_rare = [name for name in ("rare_entity_state", "rare_tail_state") if name in frame]
        for name in raw_rare:
            raw_labels = _aligned_labels(data, lagged, frame, name, spec)
            if raw_labels is not None:
                r_train = pd.Series(raw_labels.iloc[tr_i].to_numpy()).astype("string")
                r_test = pd.Series(raw_labels.iloc[te_i].to_numpy()).astype("string")
                record["support"][name] = [
                    {"state": state, "train_rows": int((r_train == state).sum()),
                     "train_entities": int(train.loc[r_train == state, spec.id_column].nunique()),
                     "test_rows": int((r_test == state).sum()),
                     "test_entities": int(test.loc[r_test == state, spec.id_column].nunique())}
                    for state in sorted(set(r_train) | set(r_test))]
        record["selected_family"] = str(selected["macro_name"]).split(":", 1)[0]
        record["selected_state_count"] = int(selected["state_count"])
        record["compression"] = _jsonable(selected.get("compression"))
        record["compression_definition"] = "package-selected candidate compression"
        if record["compression"] is None:
            record["cardinality_compression_ratio"] = float(
                1.0 - np.log2(max(2, int(selected["state_count"]))) / np.log2(max(2, len(train))))
            record["compression"] = record["cardinality_compression_ratio"]
            record["compression_definition"] = "benchmark fallback cardinality ratio; distinct from package compression"
        merge_degeneracies = outer_metadata.get("refit_merge_degeneracies", [])
        if not merge_degeneracies:
            merge_degeneracies = outer_encoder.get("refit_merge_degeneracies", [])
        record["refit_diagnostics"] = {"merge_degeneracies": merge_degeneracies,
            "recipe_estimability_failure": pkg_eval.get("recipe_estimability_failure")}
        record["selection_score"] = float(selection_score)
        record["recipe_estimable"] = bool(recipe and not pkg_eval.get("recipe_estimability_failure")
                                           and not merge_degeneracies
                                           and selected.get("recipe_estimable", True))
        if not record["recipe_estimable"]:
            raise ValueError("selected recipe carries unsupported refit diagnostics")
        record["search_audit"] = {"n_initial_candidates": len(result.get("searches", [])),
                                  "n_searches_with_best": sum(bool(s.get("best")) for s in result.get("searches", [])),
                                  "selected_macro": selected.get("macro_name"),
                                  "search_statuses": [s.get("status", "legacy_unreported") for s in result.get("searches", [])],
                                  "sampled_macro_counts": [s.get("sampled_macro_count") for s in result.get("searches", [])],
                                  "rejected_candidates": [item for s in result.get("searches", []) for item in s.get("rejected_candidates", [])],
                                  "selected_fold_refit_topologies": selected.get("fold_refit_topologies", []),
                                  "selected_development_refit_topology": outer_metadata.get("refit_topology"),
                                  "refit_encoder": outer_encoder}
        record["validation_audit"] = result["validation"].get("audit")
        # Available truths are diagnostics of recovery only; they never enter selection.
        truth = getattr(data, "truth", None)
        if isinstance(truth, dict):
            record["truth_recovered"] = _jsonable(truth.get("recovered", truth.get("macro_recovered")))
        record["causal_interpretation"] = "predictive benchmark only; observational coefficients are not identified effects"
        if case in {"known_confounding", "correlated_treatments"} and interventions and "u" in train:
            macro_frame = train.copy()
            macro_frame["__selected_macro_state"] = train_macro.labels.astype("string").to_numpy()
            names = [interventions[0]] if case == "known_confounding" else interventions
            record["confounding_diagnostics"] = _confounding_coefficients(
                train, outcome=target, treatments=names, confounder="u",
                macro_frame=macro_frame, macro_column="__selected_macro_state")
            record["package_adjustment"] = result.get("adjustment", {})
            record["package_adjustment_reproduction"] = _validate_package_adjustment(
                record["package_adjustment"],
                record["confounding_diagnostics"]["joint_raw_confounder"], names)
            if settings.get("name") == "full" and not record["package_adjustment_reproduction"]["validated"]:
                raise ValueError("package adjustment did not reproduce the declared joint conditional model")
        if case == "known_confounding" and interventions and "u" in train:
            record["adjustment_cluster_bootstrap"] = _cluster_adjustment_bootstrap(
                train, target=target, treatment=interventions[0], confounder="u",
                seed=int(seed), bootstrap_seed=int(settings.get("bootstrap_seed", 20261003)),
                draws=int(settings.get("bootstrap_replicates", 1999)),
                true_effect=(truth or {}).get("treatment_effect") if isinstance(truth, dict) else None)
        record["status"] = "success"
    except Exception as exc:
        record["failure_reason"] = f"{type(exc).__name__}: {exc}"
        record["failure_traceback"] = traceback.format_exc(limit=4)
    return _jsonable(record)


def _validate_package_adjustment(adjustment, independent_joint, treatments):
    """Compare the package-native estimand/coefficients to an independent fit."""
    paths = {p.get("intervention", p.get("treatment")): p for p in adjustment.get("pathways", [])}
    comparisons = {}
    for treatment in treatments:
        coefficient = paths.get(treatment, {}).get("coefficient")
        independent = independent_joint[treatment]
        matches = (coefficient is not None and np.isfinite(float(coefficient))
                   and bool(np.isclose(float(coefficient), independent, rtol=1e-9, atol=1e-10)))
        comparisons[treatment] = {"package": coefficient, "independent_joint": independent,
                                  "matches": matches}
    estimand = adjustment.get("estimand")
    return {"estimand": estimand, "coefficients": comparisons,
            "validated": estimand == "joint_conditional" and all(p["matches"] for p in comparisons.values()),
            "scope": "known synthetic joint conditional coefficient model; not real-world causal identification"}


def _worker_init(source: str, profile: dict[str, Any]):
    """Load the pinned package once in each spawned worker process."""
    global _WORKER_DISCOVERY, _WORKER_SOURCE, _WORKER_PROFILE
    _WORKER_SOURCE = Path(source)
    _WORKER_PROFILE = profile
    _WORKER_DISCOVERY = _load_package(_WORKER_SOURCE)


def _worker_run_trial(task: tuple[int, str, str, int]):
    """Execute one complete replicate; index remains stable under any finish order."""
    index, case, split, seed = task
    record = run_trial(case, split, seed, _WORKER_SOURCE, _WORKER_PROFILE, _WORKER_DISCOVERY)
    record["planned_index"] = index
    return index, record


def _worker_run_trial_local(task, source, profile, discovery):
    """Same stable task wrapper for serial runs, avoiding process-global state."""
    index, case, split, seed = task
    record = run_trial(case, split, seed, source, profile, discovery)
    record["planned_index"] = index
    return index, record


def _failed_worker_record(task: tuple[int, str, str, int], exc: Exception):
    index, case, split, seed = task
    record = {"case": case, "split": split, "seed": seed, "planned_index": index,
        "split_seed": int(_WORKER_PROFILE.get("split_seed", 17)) if isinstance(_WORKER_PROFILE, dict) else 17,
        "status": "failed", "paired_r2": {key: None for key in ("selected_macro", "restricted_micro",
            "no_macro", "prespecified_macro", "flexible_micro", "matched_micro")},
        "compression": None, "selected_family": None, "selected_state_count": None,
        "support": {}, "failure_reason": f"worker failure: {type(exc).__name__}: {exc}"}
    return index, record


def _confounding_coefficients(frame, *, outcome, treatments, confounder, macro_frame, macro_column):
    """Independent NumPy marginal, joint/raw-U, and coarse-macro coefficient checks."""
    clean, y = baselines._finite_targets(frame, outcome)
    result = {}
    for treatment in treatments:
        x = clean[[treatment]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        result[f"marginal_{treatment}"] = float(np.linalg.lstsq(
            np.column_stack([np.ones(len(y)), x]), y, rcond=None)[0][1])
    xcols = [*treatments, confounder]
    x = clean[xcols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    beta = np.linalg.lstsq(np.column_stack([np.ones(len(y)), x]), y, rcond=None)[0]
    result["joint_raw_confounder"] = {name: float(beta[i + 1]) for i, name in enumerate(xcols)}
    coarse = macro_frame.loc[clean.index].copy()
    macro_design = baselines._base_matrix(coarse, [*treatments, macro_column],
        *baselines._fit_encoders(coarse, [*treatments, macro_column]))
    mbeta = np.linalg.lstsq(np.column_stack([np.ones(len(y)), macro_design]), y, rcond=None)[0]
    result["coarse_macro_adjusted"] = {name: float(mbeta[i + 1]) for i, name in enumerate(treatments)}
    return result


def _cluster_adjustment_bootstrap(frame, *, target, treatment, confounder, seed, true_effect,
                                  bootstrap_seed: int = 20261003, draws: int = 1999):
    """Entity bootstrap for the known-confounder raw-U adjusted coefficient."""
    clean, y = baselines._finite_targets(frame, target)
    entities = clean.iloc[:, clean.columns.get_loc("entity_id")].to_numpy() if "entity_id" in clean else None
    if entities is None:
        raise ValueError("cluster bootstrap requires the declared entity_id column")
    groups = {key: np.flatnonzero(entities == key) for key in pd.unique(entities)}
    keys = list(groups)
    rng = np.random.default_rng((bootstrap_seed + seed) % (2**32))
    point_x = clean[[treatment, confounder]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    point = float(np.linalg.lstsq(np.column_stack([np.ones(len(y)), point_x]), y, rcond=None)[0][1])
    estimates = []
    for _ in range(draws):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        ix = np.concatenate([groups[key] for key in sampled])
        beta = np.linalg.lstsq(np.column_stack([np.ones(len(ix)), point_x[ix]]), y[ix], rcond=None)[0]
        if np.isfinite(beta[1]):
            estimates.append(float(beta[1]))
    interval = np.quantile(estimates, [.025, .975]).tolist() if len(estimates) >= int(.8 * draws) else None
    return {"estimate": point, "ci95": interval, "true_effect": true_effect,
            "interval_scope": "prespecified_raw_confounder_adjusted_model_entity_bootstrap",
            "bootstrap_unit": "entity", "bootstrap_draws": draws,
            "successful_draws": len(estimates)}


def run(profile_name: str, source: Path, expected_commit: str, output: Path,
        *, development_smoke: bool = False, readiness_status: Path | None = None,
        workers: int = 1) -> dict[str, Any]:
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    profile = get_profile(profile_name)
    source_info = verify_source(source, expected_commit)
    if profile_name == "full":
        if expected_commit.lower() == EXPECTED_BASE.lower():
            raise RuntimeError("full benchmark requires the integrated ready pinned source, not the reviewed base")
        status_path = readiness_status or (Path(__file__).resolve().parents[4] / "research" / "discovery-integrated" / "status.json")
        if not status_path.is_file():
            raise RuntimeError(f"full benchmark blocked: readiness status not found: {status_path}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        ready = bool(status.get("ready_for_integration", status.get("ready", False)))
        pinned = status.get("commit", status.get("integrated_commit", status.get("source_commit")))
        if not ready or pinned != expected_commit:
            raise RuntimeError("full benchmark requires status.json ready=true at the requested source commit")
    elif not development_smoke and expected_commit.lower() == EXPECTED_BASE.lower():
        raise RuntimeError("the reviewed base is allowed only with --development-smoke")
    discovery = _load_package(source.resolve())
    seeds = generate_seeds(profile)
    cells = list(profile["cells"])
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "profile": profile_name,
        "reportable": bool(profile_name == "full" and not development_smoke),
        "source": source_info, "protocol_sha256": protocol_hash(profile),
        "protocol": profile, "seeds": seeds, "seed_policy": "frozen independent generated dataset seeds; no retries",
        "cells": cells, "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "package_import": str(discovery.__file__),
        "development_smoke": development_smoke, "workers": workers}
    output.mkdir(parents=True, exist_ok=True)
    for filename in ("manifest.json", "trials.jsonl", "summary.json"):
        if (output / filename).exists():
            raise FileExistsError(f"refusing to overwrite existing benchmark evidence: {output / filename}")
    harness_digest = harness_fingerprints()
    harness_root = Path(__file__).resolve().parents[2]
    harness_head = _git(harness_root, "rev-parse", "HEAD")
    manifest["harness"] = {"head": harness_head, "file_sha256": harness_digest}
    (output / "manifest.json").write_text(_strict_json(manifest) + "\n", encoding="utf-8")
    trials_path = output / "trials.jsonl"
    records_by_index = {}
    tasks = [(i, str(cell["case"]), str(cell.get("split", "entity_holdout")), int(seed))
             for i, (cell, seed) in enumerate((cell, seed) for cell in cells for seed in seeds)]
    # New file per run: checkpoint each result before starting another independent dataset.
    with trials_path.open("w", encoding="utf-8", newline="\n") as stream:
        if workers == 1:
            for task in tasks:
                index, record = _worker_run_trial_local(task, source.resolve(), profile, discovery)
                records_by_index[index] = record
                stream.write(_strict_json(record) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                                     initargs=(str(source.resolve()), profile)) as pool:
                futures = {pool.submit(_worker_run_trial, task): task for task in tasks}
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        index, record = future.result()
                    except Exception as exc:
                        index, record = _failed_worker_record(task, exc)
                    records_by_index[index] = record
                    stream.write(_strict_json(record) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
    records = [records_by_index[i] for i in range(len(tasks))]
    end_source_info = verify_source(source, expected_commit)
    if end_source_info["head"] != source_info["head"]:
        raise RuntimeError("Discovery source changed during benchmark run")
    if harness_fingerprints() != harness_digest:
        raise RuntimeError("benchmark harness files changed during run")
    summary = summarize_trials(records, profile)
    summary_doc = {"profile": profile_name, "protocol_sha256": manifest["protocol_sha256"],
                   "source": source_info, "source_verified_at_end": end_source_info,
                   "reportable": manifest["reportable"], "trial_count": len(records),
                   "completed_count": sum(r["status"] == "success" for r in records),
                   "failed_count": sum(r["status"] != "success" for r in records),
                   "summary": summary}
    (output / "summary.json").write_text(_strict_json(summary_doc) + "\n", encoding="utf-8")
    return summary_doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "full"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--readiness-status", type=Path)
    parser.add_argument("--workers", type=int, default=1,
                        help="independent dataset workers (default: 1)")
    parser.add_argument("--development-smoke", action="store_true",
                        help="allow quick run on an unintegrated source; output is marked non-reportable")
    args = parser.parse_args(argv)
    summary = run(args.profile, args.source, args.expected_commit, args.output,
                  development_smoke=args.development_smoke, readiness_status=args.readiness_status,
                  workers=args.workers)
    print(json.dumps({k: summary[k] for k in ("profile", "reportable", "trial_count", "completed_count", "failed_count")}, indent=2))


if __name__ == "__main__":
    main()
