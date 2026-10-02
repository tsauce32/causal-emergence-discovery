"""Command-line interface for causal-emergence-discovery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from causal_emergence_discovery.discovery import DiscoveryConfig, discover_from_csv
from causal_emergence_discovery.panel import load_panel_csv, validate_panel
from causal_emergence_discovery.spec import load_spec


def validate_command(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    df = load_panel_csv(args.csv)
    summary = validate_panel(df, spec)
    print(f"PASS {args.csv}")
    print(f"rows: {summary.row_count}")
    print(f"entities: {summary.entity_count}")
    print(f"columns: {len(summary.observed_columns)}")
    for warning in summary.warnings:
        print(f"warning: {warning}")
    return 0


def discover_command(args: argparse.Namespace) -> int:
    config = DiscoveryConfig(
        outcome=args.outcome,
        lag=args.lag,
        max_states=args.max_states,
        paths=args.paths,
        branching_factor=args.branching_factor,
        folds=args.folds,
        top_k=args.top_k,
        validation_mode=args.validation_mode,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )
    result = discover_from_csv(args.csv, args.spec, config)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_summary(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ced",
        description="Discover emergent macro-causal hypotheses in longitudinal tabular data.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a CSV and study spec.")
    validate_parser.add_argument("csv")
    validate_parser.add_argument("spec")
    validate_parser.set_defaults(func=validate_command)

    discover_parser = subparsers.add_parser("discover", help="Run outcome-oriented discovery.")
    discover_parser.add_argument("csv")
    discover_parser.add_argument("spec")
    discover_parser.add_argument("--outcome", default=None)
    discover_parser.add_argument("--lag", type=int, default=1)
    discover_parser.add_argument("--max-states", type=int, default=6)
    discover_parser.add_argument("--paths", type=int, default=10)
    discover_parser.add_argument("--branching-factor", type=int, default=2)
    discover_parser.add_argument("--folds", type=int, default=5)
    discover_parser.add_argument("--top-k", type=int, default=10)
    discover_parser.add_argument(
        "--validation-mode", choices=("entity_holdout", "forward_time", "within_entity_interpolation"),
        default="entity_holdout", help="Predictive estimand: new entities, future periods, or observed-entity interpolation.",
    )
    discover_parser.add_argument("--holdout-fraction", type=float, default=0.2)
    discover_parser.add_argument("--seed", type=int, default=0)
    discover_parser.add_argument("--json", action="store_true", help="Print the full JSON result.")
    discover_parser.add_argument("--output", help="Write the full JSON result to a file.")
    discover_parser.set_defaults(func=discover_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def _print_summary(result: dict[str, object]) -> None:
    config = result["config"]
    panel = result["panel"]
    columns = result["columns"]
    top_macros = result["top_macros"]
    print("causal-emergence discovery")
    print(f"status: {result['status']}")
    print(f"rows: {panel['row_count']}  entities: {panel['entity_count']}")
    print(f"outcome: {config['outcome']}  lag: {config['lag']}  target: {columns['target']}")
    if "validation_mode" in config:
        fraction = config.get("holdout_fraction")
        holdout_text = "n/a" if fraction is None else f"{fraction:.0%}"
        print(f"validation: {config['validation_mode']}  outer holdout: {holdout_text}")
    print(f"candidate macro features: {len(columns['macro_features'])}")
    if top_macros:
        best = top_macros[0]
        ranking_score = best.get("ranking_score", best.get("score"))
        score_text = "n/a" if ranking_score is None else f"{ranking_score:.4f}"
        print(
            "best macro: "
            f"{best['macro_name']} "
            f"ranking_score={score_text} "
            f"states={best['state_count']}"
        )
        ranking_components = best.get("ranking_components")
        if ranking_components:
            component_text = ", ".join(
                f"{name}={value:.4f}" for name, value in ranking_components.items()
            )
            print(f"  ranking components: {component_text}")
        if best.get("macro_r2") is not None and best.get("micro_r2") is not None:
            print(
                "  raw predictive R2: "
                f"macro={best['macro_r2']:.4f}, micro={best['micro_r2']:.4f}"
            )
        difference = best.get("predictive_r2_difference")
        if difference is not None:
            print(
                "  predictive R2 difference (macro - micro, mean paired fold gap): "
                f"{difference:.4f}"
            )
        if best.get("predictive_evaluation_scope") is not None:
            print(f"  predictive evaluation scope: {best['predictive_evaluation_scope']}")
        if best.get("emergence_evidence_status") is not None:
            print(f"  emergence evidence: {best['emergence_evidence_status']}")
        validation_specificity = best.get("validation_specificity")
        if validation_specificity is not None:
            print(
                "  validation specificity "
                f"({best['specificity_definition']}): {validation_specificity:.4f}"
            )
        outcome_specificity = best.get("outcome_specificity")
        if outcome_specificity is not None:
            print(
                "  descriptive in-sample outcome specificity: "
                f"{outcome_specificity:.4f}"
            )
    evaluation = result.get("outer_evaluation")
    if evaluation is not None:
        evaluation_status = evaluation.get("status", "evaluated")
        macro_r2 = evaluation.get("macro_r2")
        micro_r2 = evaluation.get("micro_r2")
        if macro_r2 is None or micro_r2 is None:
            reason = evaluation.get("reason", evaluation.get("unavailable_reason"))
            suffix = f" reason={reason}" if reason else ""
            print(f"untouched holdout: status={evaluation_status}{suffix}")
            support = evaluation.get("target_support")
            if isinstance(support, dict):
                reserved = support.get("reserved_rows", support.get("reserved"))
                eligible = support.get("eligible_rows", support.get("eligible"))
                if reserved is not None or eligible is not None:
                    print(f"  holdout target support: reserved={reserved}, eligible={eligible}")
        else:
            rows = evaluation.get("test_rows", "n/a")
            difference = evaluation.get("predictive_r2_difference")
            difference_text = "" if difference is None else f", macro-micro={difference:.4f}"
            print(
                f"untouched holdout: macro_r2={macro_r2:.4f}, "
                f"micro_r2={micro_r2:.4f}{difference_text}, rows={rows}"
            )
    pathways = result["candidate_pathways"]
    if pathways:
        print("candidate pathways:")
        for pathway in pathways[:5]:
            coefficient = pathway["coefficient"]
            coefficient_text = "n/a" if coefficient is None else f"{coefficient:.4f}"
            print(
                f"  {pathway['intervention']} -> {columns['target']}: "
                f"associational coef={coefficient_text}"
            )
    for warning in result["warnings"]:
        print(f"warning: {warning}")


if __name__ == "__main__":
    raise SystemExit(main())
