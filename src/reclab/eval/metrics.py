"""Standard top-K recommendation metrics used across every architecture in the
eval harness, so results are directly comparable regardless of which
architecture produced them."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def recall_at_k(recommended: Sequence[Any], relevant: set[Any], k: int) -> float:
    """Fraction of relevant items captured in the top-k recommendations."""
    if not relevant:
        return 0.0
    top_k = set(recommended[:k])
    return len(top_k & relevant) / len(relevant)


def ndcg_at_k(recommended: Sequence[Any], relevant: set[Any], k: int) -> float:
    """Normalized discounted cumulative gain over the top-k recommendations."""
    if not relevant:
        return 0.0

    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 2)  # i is 0-indexed, rank is i+1

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def coverage_at_k(all_recommendations: Sequence[Sequence[Any]], catalog: set[Any], k: int) -> float:
    """Fraction of the item catalog that appears in at least one user's
    top-k recommendations. Low coverage with high recall/NDCG is a sign of a
    model that's over-fit to a small popular slice of the catalog.

    Takes the actual catalog *set*, not just its size — a real bug found by
    running this against Amazon Reviews data: hybrid_llm's candidate pool
    isn't limited to the train/test interaction catalog (it can recommend
    items with zero training interactions by design, see
    architectures/hybrid_llm.py), so items outside the catalog were
    inflating the numerator and coverage_at_k could exceed 1.0 — nonsensical
    for a fraction. Recommendations are now intersected with the catalog
    before counting: you can't "cover" catalog items you were never asked
    about, so recommending items outside it doesn't count toward coverage
    of it either way. See benchmarks/README.md for the real-data finding
    this fixes."""
    if not catalog:
        return 0.0
    recommended_items: set[Any] = set()
    for recs in all_recommendations:
        recommended_items.update(recs[:k])
    return len(recommended_items & catalog) / len(catalog)
