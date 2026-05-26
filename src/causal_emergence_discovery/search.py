"""Branching greedy search over discrete macro-variable state merges."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from causal_emergence_discovery.macro import MacroAssignment, possible_state_merges
from causal_emergence_discovery.scoring import MacroScore, score_macro


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
) -> dict[str, object]:
    """Search for coarser macro variables by merging macro states."""
    if n_paths < 1:
        raise ValueError("n_paths must be positive.")
    if branching_factor < 1:
        raise ValueError("branching_factor must be positive.")
    if min_states < 1:
        raise ValueError("min_states must be positive.")

    start_score = score_macro(
        df,
        initial_macro,
        outcome=outcome,
        target_column=target_column,
        micro_feature_columns=micro_feature_columns,
        intervention_columns=intervention_columns,
        folds=folds,
    )
    start_record = _record(initial_macro, start_score)
    frontier = [SearchNode(initial_macro, start_score, (start_record,))]
    sampled: dict[tuple[int, ...], SearchNode] = {_signature(initial_macro): frontier[0]}

    while any(node.macro.state_count > min_states for node in frontier):
        expanded: list[SearchNode] = []
        for node in frontier:
            if node.macro.state_count <= min_states:
                expanded.append(node)
                continue
            candidates = []
            for candidate in possible_state_merges(node.macro):
                candidate_score = score_macro(
                    df,
                    candidate,
                    outcome=outcome,
                    target_column=target_column,
                    micro_feature_columns=micro_feature_columns,
                    intervention_columns=intervention_columns,
                    folds=folds,
                )
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
    }


def _record(macro: MacroAssignment, score: MacroScore) -> dict[str, object]:
    counts = macro.labels.value_counts().sort_index()
    record = score.to_dict()
    record.update(
        {
            "method": macro.method,
            "feature_columns": list(macro.feature_columns),
            "metadata": macro.metadata,
            "state_sizes": {str(int(index)): int(value) for index, value in counts.items()},
        }
    )
    return record


def _rank_key(node: SearchNode) -> tuple[float, float, float, int, str]:
    return (
        node.score.score,
        node.score.emergence_delta,
        node.score.specificity,
        node.score.state_count,
        node.macro.name,
    )


def _signature(macro: MacroAssignment) -> tuple[int, ...]:
    return tuple(int(value) for value in macro.labels.tolist())
