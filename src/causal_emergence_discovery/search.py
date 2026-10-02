"""Branching greedy search over discrete macro-variable state merges."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from causal_emergence_discovery.macro import MacroAssignment, possible_state_merges
from causal_emergence_discovery.scoring import MacroScore, MacroScoreRefitError, score_macro


@dataclass(frozen=True)
class SearchNode:
    macro: MacroAssignment
    score: MacroScore
    path: tuple[dict[str, object], ...]


def greedy_macro_search(
    df: pd.DataFrame,
    initial_macro: MacroAssignment,
    *,
    outcome: str,
    target_column: str,
    micro_feature_columns: list[str],
    intervention_columns: list[str],
    folds: int = 5,
    n_paths: int = 10,
    branching_factor: int = 2,
    min_states: int = 2,
    validation_splits=None,
    column_specs=None,
) -> dict[str, object]:
    """Search for coarser macro variables by merging macro states."""
    if n_paths < 1:
        raise ValueError("n_paths must be positive.")
    if branching_factor < 1:
        raise ValueError("branching_factor must be positive.")
    if min_states < 1:
        raise ValueError("min_states must be positive.")

    rejected: dict[tuple[int, ...], dict[str, object]] = {}
    try:
        start_score = _score(df, initial_macro, outcome, target_column, micro_feature_columns,
                             intervention_columns, folds, validation_splits, column_specs)
    except MacroScoreRefitError as exc:
        rejected[_signature(initial_macro)] = _rejection(initial_macro, exc)
        return {
            "initial_macro": initial_macro.name,
            "status": "no_valid_candidate",
            "best": None,
            "sampled_macro_count": 0,
            "sampled_macros": [],
            "paths": [],
            "rejected_candidates": list(rejected.values()),
            "ranking_scope": "candidates supported on every required development fold",
        }
    start_record = _record(initial_macro, start_score)
    frontier = [SearchNode(initial_macro, start_score, (start_record,))]
    sampled: dict[tuple[int, ...], SearchNode] = {_signature(initial_macro): frontier[0]}
    terminal: set[tuple[int, ...]] = set()

    while any(node.macro.state_count > min_states and _signature(node.macro) not in terminal for node in frontier):
        expanded: list[SearchNode] = []
        for node in frontier:
            signature = _signature(node.macro)
            if node.macro.state_count <= min_states or signature in terminal:
                expanded.append(node)
                continue
            candidates = []
            for candidate in possible_state_merges(node.macro):
                try:
                    candidate_score = _score(df, candidate, outcome, target_column,
                                             micro_feature_columns, intervention_columns,
                                             folds, validation_splits, column_specs)
                except MacroScoreRefitError as exc:
                    rejected[_signature(candidate)] = _rejection(candidate, exc)
                    continue
                candidate_node = SearchNode(
                    candidate,
                    candidate_score,
                    (*node.path, _record(candidate, candidate_score)),
                )
                candidates.append(candidate_node)
                signature = _signature(candidate)
                existing = sampled.get(signature)
                if existing is None or _rank_key(candidate_node) > _rank_key(existing):
                    sampled[signature] = candidate_node
            if not candidates:
                # A valid current recipe with no supported merge remains a
                # terminal search path; it cannot be ranked against rejects.
                terminal.add(signature)
                expanded.append(node)
            else:
                expanded.extend(sorted(candidates, key=_rank_key, reverse=True)[:branching_factor])
        if not expanded:
            break
        frontier = sorted(expanded, key=_rank_key, reverse=True)[:n_paths]

    best = max(sampled.values(), key=_rank_key)
    sampled_nodes = sorted(sampled.values(), key=_rank_key, reverse=True)
    return {
        "initial_macro": initial_macro.name,
        "best": _record(best.macro, best.score),
        "sampled_macro_count": len(sampled_nodes),
        "sampled_macros": [_record(node.macro, node.score) for node in sampled_nodes],
        "paths": [list(node.path) for node in sorted(frontier, key=_rank_key, reverse=True)],
        "status": "partial_topology_rejections" if rejected else "complete",
        "rejected_candidates": list(rejected.values()),
        "ranking_scope": "candidates supported on every required development fold",
    }


def _score(df, macro, outcome, target_column, micro_feature_columns,
           intervention_columns, folds, validation_splits, column_specs):
    return score_macro(
        df, macro, outcome=outcome, target_column=target_column,
        micro_feature_columns=micro_feature_columns,
        intervention_columns=intervention_columns, folds=folds,
        validation_splits=validation_splits, column_specs=column_specs,
    )


def _rejection(macro: MacroAssignment, exc: MacroScoreRefitError) -> dict[str, object]:
    return {
        "macro_name": macro.name,
        "method": macro.method,
        "status": exc.status,
        "comparable": False,
        "diagnostics": exc.diagnostics,
    }


def _record(macro: MacroAssignment, score: MacroScore) -> dict[str, object]:
    counts = macro.labels.value_counts().sort_index()
    encoder = macro.encoder or {}
    record = score.to_dict()
    record.update(
        {
            "status": "valid",
            "comparable": True,
            "method": macro.method,
            "feature_columns": list(macro.feature_columns),
            "metadata": macro.metadata,
            "recipe_identity": encoder.get("identity"),
            "recipe_algorithm_version": encoder.get("algorithm_version"),
            "state_sizes": {str(int(index)): int(value) for index, value in counts.items()},
        }
    )
    return record


def _rank_key(node: SearchNode) -> tuple[float, float, int, str]:
    return (
        node.score.ranking_score,
        node.score.specificity,
        node.score.state_count,
        node.macro.name,
    )


def _signature(macro: MacroAssignment) -> tuple[int, ...]:
    return tuple(int(value) for value in macro.labels.tolist())
