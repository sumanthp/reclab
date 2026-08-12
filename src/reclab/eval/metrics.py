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


def coverage_at_k(all_recommendations: Sequence[Sequence[Any]], catalog_size: int, k: int) -> float:
    """Fraction of the item catalog that appears in at least one user's
    top-k recommendations. Low coverage with high recall/NDCG is a sign of a
    model that's over-fit to a small popular slice of the catalog."""
    if catalog_size == 0:
        return 0.0
    recommended_items: set[Any] = set()
    for recs in all_recommendations:
        recommended_items.update(recs[:k])
    return len(recommended_items) / catalog_size
