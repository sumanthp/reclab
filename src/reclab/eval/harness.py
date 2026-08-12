"""Offline evaluation harness: temporal train/test split, fit-and-evaluate for
a single architecture, and cold-start slice performance — the actual
mechanism that checks whether the reasoning engine's shortlist means anything
(see docs/architecture/mvp-plan.md section 5, the "working" bar for Phase 0)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from reclab.architectures.base import Architecture
from reclab.eval.metrics import coverage_at_k, ndcg_at_k, recall_at_k


def temporal_train_test_split(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    timestamp_col: str = "timestamp",
    holdout_n: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-last-N-out split per user, ordered by timestamp.

    Users with `holdout_n` or fewer total interactions contribute nothing to
    the test set (there's nothing meaningful to hold out) — all of their rows
    stay in train. This matches how temporal splits are normally done for
    sequential recommenders: you evaluate "did we predict what the user did
    next," not a random held-out interaction from the middle of their history.
    """
    if timestamp_col not in interactions.columns:
        raise ValueError(
            f"temporal_train_test_split requires a '{timestamp_col}' column; "
            "pass timestamp_col= if it's named differently"
        )

    sorted_df = interactions.sort_values([user_col, timestamp_col])
    group_sizes = sorted_df.groupby(user_col)[user_col].transform("size")
    rank_from_end = sorted_df.groupby(user_col).cumcount(ascending=False)

    is_test = (rank_from_end < holdout_n) & (group_sizes > holdout_n)
    return sorted_df[~is_test].reset_index(drop=True), sorted_df[is_test].reset_index(drop=True)


@dataclass
class ExampleRecommendation:
    """One test user's actual recommendations, held-out item(s), and
    whether it was a hit — makes "Recall@10 = 0.14" concrete instead of
    only an aggregate number. See docs/architecture/ui-ux-plan.md section
    3.4, the "sample of actual recommended items" this fills in."""

    user_id: Any
    recommended: list[Any]
    held_out: list[Any]
    hit: bool


# Small and fixed, not sampled: the first N test users in a deterministic
# (groupby) order, so the same dataset always produces the same examples —
# reproducibility matters more here than a "representative" sample would,
# and N is small enough that showing all hits or all misses by bad luck is
# an acceptable, honestly-reported outcome rather than something to correct
# for by cherry-picking.
MAX_EXAMPLE_RECOMMENDATIONS = 5


@dataclass
class EvalResult:
    architecture: str
    k: int
    n_test_users: int
    recall_at_k: float
    ndcg_at_k: float
    coverage_at_k: float
    cold_start_recall_at_k: float | None  # None if no cold-start items appeared in test
    # Fraction of test users whose top-k includes *any* cold-start item — a
    # softer, higher-sample-size companion to cold_start_recall_at_k (which
    # requires exactly predicting the specific held-out item, a very hard,
    # high-variance target when few test users happen to have a cold-start
    # item as their held-out interaction). This asks "does this architecture
    # surface cold items at all," which is the capability hybrid_llm actually
    # claims over two_tower/sasrec — see benchmarks/README.md.
    cold_start_surfaced_rate: float
    example_recommendations: list[ExampleRecommendation] = field(default_factory=list)


def run_eval(
    architecture: Architecture,
    train: pd.DataFrame,
    test: pd.DataFrame,
    item_metadata: pd.DataFrame | None = None,
    k: int = 10,
    cold_start_threshold: int = 5,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> EvalResult:
    """Fit `architecture` on `train` and evaluate against held-out `test`
    interactions. Assumes `architecture` is unfitted (a fresh instance) —
    the harness owns the fit/eval lifecycle so results aren't accidentally
    computed against data the model already trained on.

    Every `Architecture` implementation reads fixed `"user_id"`/`"item_id"`
    column names internally (see e.g. architectures/two_tower.py) rather
    than accepting `user_col`/`item_col` themselves — so if the caller's
    columns are named differently, `train`/`test` are renamed to the
    canonical names before being handed to the architecture. `user_col`/
    `item_col` still control what columns this function reads *its own*
    inputs from; they don't need to already be "user_id"/"item_id"."""
    if user_col != "user_id" or item_col != "item_id":
        train = train.rename(columns={user_col: "user_id", item_col: "item_id"})
        test = test.rename(columns={user_col: "user_id", item_col: "item_id"})
    user_col, item_col = "user_id", "item_id"

    architecture.fit(train, item_metadata)

    train_item_counts = train[item_col].value_counts()
    cold_items = set(train_item_counts[train_item_counts < cold_start_threshold].index)
    catalog = set(pd.concat([train[item_col], test[item_col]]))

    test_relevant_by_user = test.groupby(user_col)[item_col].apply(set).to_dict()

    all_recommendations: list[list[Any]] = []
    recall_scores: list[float] = []
    ndcg_scores: list[float] = []
    cold_start_hits = 0
    cold_start_relevant_total = 0
    users_with_cold_item_surfaced = 0
    examples: list[ExampleRecommendation] = []

    for user_id, relevant in test_relevant_by_user.items():
        recs = architecture.recommend(user_id, k)
        all_recommendations.append(recs)
        recall_scores.append(recall_at_k(recs, relevant, k))
        ndcg_scores.append(ndcg_at_k(recs, relevant, k))

        if set(recs[:k]) & cold_items:
            users_with_cold_item_surfaced += 1

        cold_relevant = relevant & cold_items
        if cold_relevant:
            cold_start_hits += len(set(recs[:k]) & cold_relevant)
            cold_start_relevant_total += len(cold_relevant)

        if len(examples) < MAX_EXAMPLE_RECOMMENDATIONS:
            examples.append(
                ExampleRecommendation(
                    user_id=user_id,
                    recommended=list(recs[:k]),
                    held_out=sorted(relevant, key=str),
                    hit=bool(set(recs[:k]) & relevant),
                )
            )

    n_test_users = len(test_relevant_by_user)
    coverage = coverage_at_k(all_recommendations, catalog, k) if all_recommendations else 0.0
    cold_start_recall = (
        cold_start_hits / cold_start_relevant_total if cold_start_relevant_total > 0 else None
    )

    return EvalResult(
        architecture=type(architecture).__name__,
        k=k,
        n_test_users=n_test_users,
        recall_at_k=sum(recall_scores) / n_test_users if n_test_users else 0.0,
        ndcg_at_k=sum(ndcg_scores) / n_test_users if n_test_users else 0.0,
        coverage_at_k=coverage,
        cold_start_recall_at_k=cold_start_recall,
        cold_start_surfaced_rate=users_with_cold_item_surfaced / n_test_users
        if n_test_users
        else 0.0,
        example_recommendations=examples,
    )
