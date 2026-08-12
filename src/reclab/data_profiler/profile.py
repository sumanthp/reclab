"""Compute a DataProfile from a raw interactions table.

This is the input the reasoning engine's planner reasons over. Every field here
should be something that plausibly changes which architecture wins, not just a
generic dataset stat.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DataProfile:
    """Summary statistics used by the reasoning engine to shortlist architectures.

    Attributes:
        n_users: distinct users in the interaction log.
        n_items: distinct items in the interaction log.
        n_interactions: total interaction rows.
        sparsity: 1 - (n_interactions / (n_users * n_items)). Higher = sparser.
        cold_start_ratio: fraction of items with fewer than `cold_start_threshold`
            interactions. High values favor content/LLM-assisted approaches over
            pure collaborative filtering.
        median_sequence_length: median number of interactions per user. Short
            sequences weaken session/sequence-based models like SASRec.
        has_item_text: whether item metadata includes usable free text.
        has_item_image: whether item metadata includes usable images.
    """

    n_users: int
    n_items: int
    n_interactions: int
    sparsity: float
    cold_start_ratio: float
    median_sequence_length: float
    has_item_text: bool = False
    has_item_image: bool = False


def profile_interactions(
    interactions: pd.DataFrame,
    *,
    user_col: str = "user_id",
    item_col: str = "item_id",
    cold_start_threshold: int = 5,
    item_metadata: pd.DataFrame | None = None,
    text_col: str | None = None,
    image_col: str | None = None,
) -> DataProfile:
    """Profile a raw interactions table.

    `interactions` must have at least `user_col` and `item_col`. Additional
    columns (timestamp, rating, etc.) are ignored for now — Phase 0 only needs
    the signals above to test the reasoning engine's shortlist logic.
    """
    if interactions.empty:
        raise ValueError("interactions dataframe is empty")

    n_users = interactions[user_col].nunique()
    n_items = interactions[item_col].nunique()
    n_interactions = len(interactions)

    sparsity = 1.0 - (n_interactions / (n_users * n_items))

    item_counts = interactions.groupby(item_col).size()
    cold_start_ratio = float((item_counts < cold_start_threshold).mean())

    median_sequence_length = float(interactions.groupby(user_col).size().median())

    has_item_text = bool(
        item_metadata is not None and text_col is not None and text_col in item_metadata.columns
    )
    has_item_image = bool(
        item_metadata is not None
        and image_col is not None
        and image_col in item_metadata.columns
    )

    return DataProfile(
        n_users=n_users,
        n_items=n_items,
        n_interactions=n_interactions,
        sparsity=sparsity,
        cold_start_ratio=cold_start_ratio,
        median_sequence_length=median_sequence_length,
        has_item_text=has_item_text,
        has_item_image=has_item_image,
    )
