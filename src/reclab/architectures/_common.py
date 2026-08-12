"""Shared helpers for architecture implementations: vocab building, popularity
fallback for cold users, and per-user interaction sequences. Kept private
(leading underscore) since these are implementation details, not part of the
public `Architecture` interface."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_vocab(ids: pd.Series) -> tuple[dict[Any, int], list[Any]]:
    """Map raw ids to dense 0..n-1 indices, preserving first-seen order."""
    unique_ids = list(dict.fromkeys(ids.tolist()))
    id_to_idx = {uid: idx for idx, uid in enumerate(unique_ids)}
    return id_to_idx, unique_ids


def popularity_ranking(interactions: pd.DataFrame, item_col: str = "item_id") -> list[Any]:
    """Item ids ranked by interaction count, most popular first. Used as the
    fallback recommendation for users not seen during training — every
    architecture in this repo needs *some* answer for a brand-new user, and
    "most popular" is the standard, honest baseline for that case."""
    return interactions[item_col].value_counts().index.tolist()


def user_sequences(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    timestamp_col: str | None = "timestamp",
) -> dict[Any, list[Any]]:
    """Each user's items in chronological order (or input order if no
    timestamp column is present)."""
    if timestamp_col in interactions.columns:
        interactions = interactions.sort_values(timestamp_col)
    return interactions.groupby(user_col)[item_col].apply(list).to_dict()
