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
    print(f"candidate macro features: {len(columns['macro_features'])}")
    if top_macros:
        best = top_macros[0]
        print(
            "best macro: "
            f"{best['macro_name']} "
            f"score={best['score']:.4f} "
            f"delta={best['emergence_delta']:.4f} "
            f"states={best['state_count']}"
        )
        print(
            "  components: "
            f"macro_r2={best['macro_r2']:.4f}, "
            f"micro_r2={best['micro_r2']:.4f}, "
            f"specificity={best['specificity']:.4f}, "
            f"stability={best['stability']:.4f}, "
            f"compression={best['compression']:.4f}"
        )
    pathways = result["candidate_pathways"]
    if pathways:
        print("candidate pathways:")
        for pathway in pathways[:5]:
            t_stat = pathway["t_stat"]
            t_text = "n/a" if t_stat is None else f"{t_stat:.3f}"
            coefficient = pathway["coefficient"]
            coefficient_text = "n/a" if coefficient is None else f"{coefficient:.4f}"
            print(
                f"  {pathway['intervention']} -> {pathway['target']}: "
                f"coef={coefficient_text}, t={t_text}"
            )
    for warning in result["warnings"]:
        print(f"warning: {warning}")


if __name__ == "__main__":
    raise SystemExit(main())
