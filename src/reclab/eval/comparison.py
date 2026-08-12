"""Compares the reasoning engine's shortlist against measured eval results.

Extracted from scripts/run_benchmark.py's inline reporting logic so the CLI
and the API's /compare endpoint produce the same verdict from the same rule,
rather than two implementations that can silently drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

from reclab.eval.harness import EvalResult
from reclab.reasoning_engine import Recommendation


@dataclass
class ComparisonSummary:
    """The Phase 0 question, answered for one run: did the shortlist's #1
    pick match what actually won? Neither field is ever hidden or softened
    when they disagree — see benchmarks/README.md for why that matters."""

    shortlist_pick: str
    measured_best_recall: str | None
    measured_best_cold_start_recall: str | None
    matches_on_recall: bool | None  # None if nothing was measured (no timestamp column)
    note: str | None  # extra context, e.g. the cold-start-recall caveat


def summarize_comparison(
    shortlist: list[Recommendation], eval_results: dict[str, EvalResult]
) -> ComparisonSummary:
    """`eval_results` maps architecture name -> EvalResult for every
    architecture that trained successfully (skip entries some callers may
    track separately for architectures that raised during training)."""
    shortlist_pick = shortlist[0].architecture

    if not eval_results:
        return ComparisonSummary(
            shortlist_pick=shortlist_pick,
            measured_best_recall=None,
            measured_best_cold_start_recall=None,
            matches_on_recall=None,
            note=None,
        )

    best_recall = max(eval_results, key=lambda n: eval_results[n].recall_at_k)

    cold_scored: dict[str, float] = {
        n: r.cold_start_recall_at_k
        for n, r in eval_results.items()
        if r.cold_start_recall_at_k is not None
    }
    best_cold_recall = max(cold_scored, key=lambda n: cold_scored[n]) if cold_scored else None

    matches = shortlist_pick == best_recall
    note = None
    if not matches and shortlist_pick == best_cold_recall:
        note = (
            "the shortlist's pick does win on cold-start recall, which is plausibly what "
            "its rationale was actually optimizing for — a single Recall@K comparison "
            "doesn't capture that trade-off"
        )

    return ComparisonSummary(
        shortlist_pick=shortlist_pick,
        measured_best_recall=best_recall,
        measured_best_cold_start_recall=best_cold_recall,
        matches_on_recall=matches,
        note=note,
    )
