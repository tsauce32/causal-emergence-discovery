"""Reproduce the known-confounder control: --output selects the JSON report."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from causal_emergence_discovery.adjustment import adjustment_sufficiency_audit
from causal_emergence_discovery.discovery import DiscoveryConfig, run_discovery
from causal_emergence_discovery.macro import quantile_macro
from causal_emergence_discovery.models import fit_linear_model
from causal_emergence_discovery.panel import build_lagged_table
from causal_emergence_discovery.spec import StudySpec


def make_fixture() -> tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(7701)
    rows = []
    true_effect = 1.5
    for entity in range(800):
        previous_y = 0.0
        for time in range(4):
            u = rng.normal()
            treatment = u + rng.normal(scale=0.4)
            next_y = true_effect * treatment + 4.0 * u + rng.normal(scale=0.2)
            rows.append({"entity": entity, "time": time, "u": u, "program": treatment, "y": previous_y})
            previous_y = next_y
    return pd.DataFrame(rows), true_effect


def legacy_q3_labels(values: pd.Series) -> pd.Series:
    """Retain the original rank(method='first') macro labels for baseline reproduction."""
    filled = values.fillna(float(values.median()) if values.notna().any() else 0.0)
    labels = pd.qcut(filled.rank(method="first"), q=3, labels=False, duplicates="drop")
    mapping = {}
    canonical = []
    for value in labels.tolist():
        key = value if pd.notna(value) else "__missing__"
        if key not in mapping:
            mapping[key] = len(mapping)
        canonical.append(mapping[key])
    return pd.Series(canonical, index=values.index, dtype=int)


def coefficient_record(fit, term: str) -> dict[str, float | int]:
    position = fit.feature_names.index(term)
    return {"coefficient": float(fit.coefficients[position]), "iid_stderr": float(fit.stderr[position]), "nobs": fit.nobs, "model_r2": float(fit.r2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("adjustment-controls.generated.json"))
    args = parser.parse_args()
    panel, true_effect = make_fixture()
    spec_base = StudySpec.from_dict({
        "dataset": {"id_column": "entity", "time_column": "time"},
        "outcomes": ["y"], "interventions": ["program"],
        "columns": {"u": {"role": "context"}, "program": {"role": "intervention"}, "y": {"role": "outcome"}},
    })
    lagged = build_lagged_table(panel, spec_base, outcomes=["y"], lag=1).data.reset_index(drop=True)
    target = "y__lead1"

    legacy = lagged.copy()
    legacy["__legacy_q3_u"] = legacy_q3_labels(legacy["u"]).to_numpy()
    legacy_fit = fit_linear_model(legacy, ["program", "__legacy_q3_u"], target)
    macro = quantile_macro(lagged, "u", 3)
    categorical = lagged.copy()
    categorical["__macro"] = macro.labels.astype("string").to_numpy()
    categorical_fit = fit_linear_model(categorical, ["program", "__macro"], target)

    explicit_spec = StudySpec.from_dict({
        **spec_base.to_dict(),
        "adjustment": {"columns": ["u"], "rationale": "u is generated before treatment and is a common cause of program and the next outcome.", "include_macro": False},
    })
    audit = adjustment_sufficiency_audit(lagged, explicit_spec, macro, interventions=["program"], target_column=target)

    actual = run_discovery(panel, explicit_spec, DiscoveryConfig(
        outcome="y", max_states=3, paths=2, branching_factor=2, folds=3, top_k=1,
        validation_mode="entity_holdout", holdout_fraction=0.2, seed=0,
    ))

    result = {
        "fixture": {
            "generator": "research/github_review/discovery/audit_core_validity.py confounding fixture",
            "rng_seed": 7701,
            "entities": int(panel["entity"].nunique()),
            "periods_per_entity": 4,
            "panel_rows": len(panel),
            "lagged_model_rows": len(lagged),
            "predictor": "program at time t",
            "target": target,
            "true_program_effect": true_effect,
            "structural_equation": "next_y = 1.5 * program + 4.0 * u + Normal(0, 0.2)",
            "treatment_equation": "program = u + Normal(0, 0.4)",
        },
        "full_model_frame_controls": {
            "scope": "all 2,400 lagged rows, diagnostic reproduction only",
            "legacy_rank_first_q3_macro_only": {
                "macro": "q3:u",
                "macro_encoding": "numeric ordinal states, preserved legacy baseline",
                "program_pathway": coefficient_record(legacy_fit, "program"),
                "reference_expected_coefficient": 3.7657308744,
            },
            "current_categorical_q3_macro_only": {
                "macro": macro.name,
                "macro_encoding": "categorical state indicators",
                "program_pathway": coefficient_record(categorical_fit, "program"),
            },
            "explicit_continuous_u_adjustment": {
                "adjustment_columns": ["u"],
                "include_macro": False,
                "program_pathway": audit["pathways"][0],
                "interpretation": "observed-confounder simulation control; regression alone does not certify causal identification",
            },
            "macro_encoder": macro.metadata["encoder"],
            "remaining_confounder_information": audit["declared_covariate_diagnostics"]["within_macro_residual_information"],
        },
        "actual_discovery_run": {
            "config": actual["config"],
            "panel_support": actual["panel"],
            "macro_features": actual["columns"]["macro_features"],
            "selected_macro": actual["top_macros"][0]["macro_name"],
            "development_pathways": actual["candidate_pathways"],
            "adjustment_audit": actual["adjustment"],
            "outer_predictive_evaluation": actual["outer_evaluation"],
            "validation_audit": actual["validation"]["audit"],
        },
        "scope_and_limits": {
            "pathway_estimates_use": "development split only in actual discovery; all rows only in the explicitly labeled synthetic control above",
            "holdout_outcomes_use": "selected-macro and micro-model predictive R2 only",
            "causal_certification": False,
            "inference_limit": "reported standard errors are IID OLS diagnostics and do not account for repeated entities",
        },
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output.resolve()),
        "legacy_macro_only_coefficient": result["full_model_frame_controls"]["legacy_rank_first_q3_macro_only"]["program_pathway"]["coefficient"],
        "categorical_macro_only_coefficient": result["full_model_frame_controls"]["current_categorical_q3_macro_only"]["program_pathway"]["coefficient"],
        "continuous_adjustment_coefficient": audit["pathways"][0]["coefficient"],
        "development_adjustment_coefficient": actual["candidate_pathways"][0]["coefficient"],
        "holdout_macro_r2": actual["outer_evaluation"]["macro_r2"],
        "holdout_micro_r2": actual["outer_evaluation"]["micro_r2"],
    }, indent=2))


if __name__ == "__main__":
    main()
