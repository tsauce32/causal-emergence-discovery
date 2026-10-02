"""Audit raw experiment records and export a compact numerical report.

This module recomputes means and Monte Carlo standard errors directly from
JSONL, without using the benchmark's summarization implementation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def moments(values):
    values = [float(x) for x in values]
    if any(not math.isfinite(x) for x in values):
        raise ValueError("nonfinite raw numerical evidence")
    return {"n": len(values), "mean": statistics.fmean(values) if values else None,
            "mc_se": statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None}


def _same_number(a, b):
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


def audit_records(records, manifest, summary):
    profile = manifest["protocol"]
    if len(set(manifest["seeds"])) != profile["replicates_per_cell"]:
        raise ValueError("manifest does not retain every frozen replicate")
    expected = {(c["case"], c["split"], int(s)) for c in profile["cells"] for s in manifest["seeds"]}
    actual = [(r["case"], r["split"], int(r["seed"])) for r in records]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("raw trials are duplicated, missing, or outside the frozen matrix")
    if not manifest["reportable"] or not summary["reportable"]:
        raise ValueError("development evidence cannot be exported as the full scientific report")
    if manifest["source"]["head"] != summary["source_verified_at_end"]["head"]:
        raise ValueError("experiment source did not remain pinned")
    checks = []
    cells = summary["summary"]["cells"]
    for key, cell in cells.items():
        rows = [r for r in records if r["case"] == cell["case"] and r["split"] == cell["split"]]
        good = [r for r in rows if r["status"] == "success"]
        if (cell["n_records"] != len(rows) or cell["n_success"] != len(good)
                or cell["n_planned"] != profile["replicates_per_cell"]
                or len(rows) != cell["n_planned"] or cell["n_missing_records"] != 0
                or cell["status_counts"] != dict(Counter(r["status"] for r in rows))
                or cell["n_failed_or_unsupported"] != len(rows) - len(good)
                or not _same_number(cell["failure_rate_planned"], (len(rows) - len(good)) / profile["replicates_per_cell"])):
            raise ValueError("summary hides a trial or failure")
        below = sum(any(s["train_rows"] < profile["min_state_train_rows"]
                        or s["train_entities"] < profile["min_state_train_entities"]
                        for s in r.get("support", {}).get("states", [])) for r in good)
        if cell["support_threshold"]["n_below_threshold"] != below:
            raise ValueError("summary changes the support denominator")
        if not _same_number(cell["support_threshold"]["fraction_below_threshold_of_planned"], below / cell["n_planned"]):
            raise ValueError("summary changes the support fraction")
        if cell["selected_family_counts"] != dict(Counter(str(r["selected_family"]) for r in good if r.get("selected_family") is not None)):
            raise ValueError("summary changes selection family frequencies")
        for baseline, stored in cell["heldout_r2"].items():
            values = [r["paired_r2"][baseline] for r in good if r["paired_r2"].get(baseline) is not None]
            independently = moments(values)
            if any(not _same_number(stored[field], independently[field]) for field in ("n", "mean", "mc_se")):
                raise ValueError(f"incorrect raw-score summary: {key}/{baseline}")
            checks.append(f"{key}/{baseline}")
        for name, stored in cell["paired_r2_differences"].items():
            reference = name.removeprefix("selected_macro_minus_")
            values = [r["paired_r2"]["selected_macro"] - r["paired_r2"][reference]
                      for r in good if r["paired_r2"].get("selected_macro") is not None
                      and r["paired_r2"].get(reference) is not None]
            independently = moments(values)
            if any(not _same_number(stored[field], independently[field]) for field in ("n", "mean", "mc_se")):
                raise ValueError(f"incorrect paired-score summary: {key}/{name}")
            if stored["positive_rate"]["n"] != len(values) or stored["positive_rate"]["successes"] != sum(x > 0 for x in values):
                raise ValueError("summary changes a descriptive positive-sign count")
            checks.append(f"{key}/{name}")
    return {"passed": True, "planned_trials": len(expected), "observed_trials": len(records),
            "independently_recomputed_numerical_summaries": len(checks),
            "failed_trials": sum(r["status"] != "success" for r in records),
            "scope": "raw means, paired differences, Monte Carlo SEs, trial identities and denominators"}


def export_report(run_dir: Path, output: Path, frozen: Path):
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()]
    freeze = json.loads(frozen.read_text(encoding="utf-8"))
    canonical = json.dumps(manifest["protocol"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    if hashlib.sha256(canonical.encode()).hexdigest() != freeze["protocol_sha256"]:
        raise ValueError("reported full profile differs from its pre-experiment freeze")
    if manifest["seeds"] != freeze["seeds"]:
        raise ValueError("reported seeds differ from the pre-experiment seed family")
    if manifest["harness"]["file_sha256"]["benchmarks/pipeline/dgps.py"] != freeze["dgp_sha256"]:
        raise ValueError("DGP changed after its pre-experiment freeze")
    audit = audit_records(records, manifest, summary)
    output.mkdir(parents=True, exist_ok=True)
    (output / "independent-numerical-audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    rows = []
    secondary = {}
    for key, cell in summary["summary"]["cells"].items():
        good = [r for r in records if r["case"] == cell["case"] and r["split"] == cell["split"] and r["status"] == "success"]
        contrast = cell["paired_r2_differences"]["selected_macro_minus_restricted_micro"]
        intervals = contrast["ci95_t"] or [None, None]
        boot = contrast.get("ci95_bootstrap") or contrast.get("bootstrap_ci95") or [None, None]
        row = {"case": cell["case"], "split": cell["split"], "planned": cell["n_planned"],
               "successful": cell["n_success"], "failed": cell["n_failed_or_unsupported"],
               "macro_n": cell["heldout_r2"]["selected_macro"]["n"],
               "restricted_micro_n": cell["heldout_r2"]["restricted_micro"]["n"],
               "paired_n": contrast["n"],
               "macro_r2": cell["heldout_r2"]["selected_macro"]["mean"],
               "restricted_micro_r2": cell["heldout_r2"]["restricted_micro"]["mean"],
               "flexible_micro_r2": cell["heldout_r2"]["flexible_micro"]["mean"],
               "paired_difference": contrast["mean"], "mc_se": contrast["mc_se"],
               "mc_ci95_low": intervals[0], "mc_ci95_high": intervals[1],
               "bootstrap_ci95_low": boot[0], "bootstrap_ci95_high": boot[1],
               "compression": cell["compression"]["mean"],
               "selected_state_support_below_threshold": cell["support_threshold"]["n_below_threshold"],
               "selection_families": json.dumps(cell["selected_family_counts"], sort_keys=True)}
        rows.append(row)
        secondary[key] = {"oracle_micro_r2": moments([r["oracle_micro"]["r2"] for r in good if r.get("oracle_micro")]),
                          "macro_minus_oracle_micro": moments([r["paired_r2"]["selected_macro"] - r["oracle_micro"]["r2"] for r in good if r.get("oracle_micro")]),
                          "failure_reasons": dict(Counter(r.get("failure_reason") for r in records if r["case"] == cell["case"] and r["split"] == cell["split"] and r["status"] != "success"))}
        if good and good[0].get("confounding_diagnostics"):
            names = good[0]["confounding_diagnostics"]["joint_raw_confounder"]
            secondary[key]["joint_raw_confounder"] = {name: moments([r["confounding_diagnostics"]["joint_raw_confounder"][name] for r in good]) for name in names}
            secondary[key]["coarse_macro_adjusted"] = {name: moments([r["confounding_diagnostics"]["coarse_macro_adjusted"][name] for r in good]) for name in good[0]["confounding_diagnostics"]["coarse_macro_adjusted"]}
            native = good[0].get("package_adjustment_reproduction", {}).get("coefficients", {})
            secondary[key]["package_native_joint_coefficients"] = {name: moments([
                r["package_adjustment_reproduction"]["coefficients"][name]["package"] for r in good]) for name in native}
            secondary[key]["package_native_joint_validation_count"] = sum(
                r.get("package_adjustment_reproduction", {}).get("validated", False) for r in good)
        secondary[key]["selected_recipe_counts"] = dict(Counter(r.get("search_audit", {}).get("selected_macro") for r in good))
        secondary[key]["candidate_refit_rejections"] = {
            "runs_with_rejections": sum(bool(r.get("search_audit", {}).get("rejected_candidates")) for r in good),
            "total_rejected_candidate_evaluations": sum(len(r.get("search_audit", {}).get("rejected_candidates", [])) for r in good)}
        if cell["case"] == "rare_absent_states":
            secondary[key]["raw_absent_state_support"] = [
                {"seed": r["seed"], "rare_entity_state": r.get("support", {}).get("rare_entity_state"),
                 "rare_tail_state": r.get("support", {}).get("rare_tail_state"),
                 "prespecified_states": r.get("support", {}).get("prespecified_states")} for r in good]
    with (output / "full-results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "secondary-controls.json").write_text(json.dumps(secondary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    def number(x):
        return "NA" if x is None else f"{x:.4f}"
    lines = ["# Frozen full-profile numerical results", "", f"Pinned integrated commit: `{manifest['source']['head']}`.", "",
             f"All {len(records)} planned records are retained. {audit['failed_trials']} trials failed or were unsupported.", "",
             "The table averages held-out R² equally across successful generated panels; each panel's R² weights its evaluation rows equally. Intervals measure Monte Carlo uncertainty over independent complete-pipeline replicates, conditional on successful evaluation; they are not fold dispersion or emergence evidence. Complete-pipeline replicate bootstrap intervals, individual baselines, state support, and selection frequencies are also saved in the full summary and CSV.", "",
             "| Case / split | Successful / planned (paired n) | Macro R² | Restricted micro R² | Flexible micro R² | Macro − restricted micro (95% MC interval) |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['case']} / {r['split']} | {r['successful']} / {r['planned']} ({r['paired_n']}) | {number(r['macro_r2'])} | {number(r['restricted_micro_r2'])} | {number(r['flexible_micro_r2'])} | {number(r['paired_difference'])} [{number(r['mc_ci95_low'])}, {number(r['mc_ci95_high'])}] |")
    lines += ["", "Read positive differences alongside absolute R² and the intercept-only control in the full summary. The drift case can favor the macro while both predictors perform poorly. For entity-fingerprint holdout, every reserved ID is unseen by the raw categorical model: zero contrasts receive the arbitrary training reference entity's effect. Its large relative gap reflects unsupported category extrapolation and does not establish useful transfer to new entities.", "",
              "The matched-recipe micro control uses the selected recipe's state-indicator basis and the same treatment terms; numerical equivalence audits representation/model capacity, rather than supplying independent evidence of emergence. Formula-derived oracle controls are excluded from selection. No rejection rule was specified; positive differences are descriptive signs, not discoveries or an empirical false-positive rate.", "",
              "Limitations: 32 replicates per cell, one panel size/noise setting and a candidate budget capped at three states. The designed scenarios do not establish performance outside this matrix, simultaneous confidence guarantees, real-data causal identification, or Hoel/CE2 emergence. Forward-time tasks condition on predictors observed at each future time; they do not evaluate recursive fixed-origin forecasts.", ""]
    (output / "NUMERICAL_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frozen", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export_report(args.run_dir, args.output, args.frozen), indent=2))
