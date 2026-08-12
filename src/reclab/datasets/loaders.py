"""Loaders for real public benchmark datasets.

`load_movielens_100k` has been run end-to-end against the real
files.grouplens.org download (see benchmarks/README.md for results) — no
parsing changes were needed versus the fixture-tested version.
`load_amazon_reviews_category` has not been run against the real dataset yet
and remains a stub; see CONTRIBUTING.md.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def load_movielens_100k(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the MovieLens 100K dataset.

    `path` may be either the extracted `ml-100k/` directory or the
    downloaded `ml-100k.zip` archive (not included in this repo — see
    MOVIELENS_100K_URL). Expects the standard ml-100k layout:
      - u.data: user id \\t item id \\t rating \\t timestamp
      - u.item: item id | title | release date | ... | genre flags (pipe-separated,
        latin-1 encoded)

    Returns (interactions, item_metadata) in reclab's standard shape:
      interactions: user_id, item_id, timestamp
      item_metadata: item_id, description (movie title, used as the item text
        signal for hybrid_llm's re-ranker)
    """
    path = Path(path)

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            with zf.open("ml-100k/u.data") as f:
                interactions_raw = pd.read_csv(
                    f, sep="\t", names=["user_id", "item_id", "rating", "timestamp"]
                )
            with zf.open("ml-100k/u.item") as f:
                items_raw = pd.read_csv(
                    f,
                    sep="|",
                    encoding="latin-1",
                    header=None,
                    usecols=[0, 1],
                    names=["item_id", "title"],
                )
    else:
        interactions_raw = pd.read_csv(
            path / "u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"]
        )
        items_raw = pd.read_csv(
            path / "u.item",
            sep="|",
            encoding="latin-1",
            header=None,
            usecols=[0, 1],
            names=["item_id", "title"],
        )

    interactions = interactions_raw[["user_id", "item_id", "timestamp"]].copy()
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["item_id"] = interactions["item_id"].astype(str)

    item_metadata = items_raw.rename(columns={"title": "description"}).copy()
    item_metadata["item_id"] = item_metadata["item_id"].astype(str)

    return interactions, item_metadata


def load_amazon_reviews_category(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a single-category Amazon Reviews file (2023 release format: one
    gzipped JSONL file per category, e.g. "Video_Games.jsonl.gz").

    Expects each JSON line to have at minimum: `user_id`, `asin` (item id),
    `timestamp` (ms epoch), and ideally `title` and/or `text` for item text.
    Since item metadata (title) lives in a *separate* per-category metadata
    file in the real dataset (`meta_<category>.jsonl.gz`), pass that file's
    path as `metadata_path` if you have it — otherwise item_metadata falls
    back to review text truncated per item, which is a weaker text signal.
    """
    raise NotImplementedError(
        "Amazon Reviews loading needs the real file to nail down field names "
        "and confirm the metadata-join logic — flagged as the first thing to "
        "implement against a real download rather than guess further. See "
        "benchmarks/README.md and CONTRIBUTING.md."
    )
