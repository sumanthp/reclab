"""Loaders for real public benchmark datasets.

Both loaders have been run end-to-end against real downloads (see
benchmarks/README.md for results).
"""

from __future__ import annotations

import gzip
import json
import zipfile
from pathlib import Path

import pandas as pd

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

# McAuley Lab's 2023 release. The per-category review files ship as plain
# .jsonl (not .jsonl.gz, despite what an earlier version of this docstring
# assumed before being run against the real files) — see
# https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023. The
# `benchmark/5core/rating_only/<Category>.csv` files are a much smaller
# pre-filtered alternative (just user_id, parent_asin, rating, timestamp)
# and are what this loader's tests and the checked-in benchmark run use.
AMAZON_REVIEWS_2023_BASE_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"
)


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


def _read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_amazon_reviews_category(
    reviews_path: str | Path, metadata_path: str | Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load a single-category slice of the Amazon Reviews 2023 dataset.

    `reviews_path` accepts two real formats from
    https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023:
      - `benchmark/5core/rating_only/<Category>.csv` — small, pre-filtered
        (5-core: every user/item has >=5 interactions), columns
        `user_id,parent_asin,rating,timestamp`. No review text, so pass
        `metadata_path` for item text or item_metadata will be None.
      - `raw/review_categories/<Category>.jsonl` (optionally gzipped) — the
        full per-review records, one JSON object per line, with `user_id`,
        `parent_asin` (item id; `asin` is used as a fallback for older
        dumps), `rating`, `timestamp` (ms epoch), `title`/`text` (the
        *review's* title/text, not the product's — a weaker per-item text
        signal than real product metadata, used only when `metadata_path`
        isn't given).

    `metadata_path`, if given, points at
    `raw/meta_categories/meta_<Category>.jsonl` (optionally gzipped): one
    JSON object per product with `parent_asin` and `title` (the actual
    product title — the real item text signal for hybrid_llm's re-ranker).

    Returns (interactions, item_metadata) in reclab's standard shape:
      interactions: user_id, item_id, timestamp
      item_metadata: item_id, description — or None if neither
        `metadata_path` nor per-review `title`/`text` fields are available.
    """
    reviews_path = Path(reviews_path)

    if reviews_path.suffix == ".csv":
        reviews = pd.read_csv(reviews_path)
        item_col = "parent_asin"
    else:
        reviews = pd.DataFrame.from_records(_read_jsonl(reviews_path))
        item_col = "parent_asin" if "parent_asin" in reviews.columns else "asin"

    interactions = reviews.rename(columns={item_col: "item_id"})[
        ["user_id", "item_id", "timestamp"]
    ].copy()
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["item_id"] = interactions["item_id"].astype(str)

    item_metadata: pd.DataFrame | None = None
    if metadata_path is not None:
        # Built via explicit column construction, not rename+select: the raw
        # meta_<Category>.jsonl records already have their own "description"
        # field (a list of bullet points) distinct from "title" — renaming
        # "title" to "description" alongside it produces two same-named
        # columns and downstream `df["description"]` silently returns a
        # DataFrame instead of a Series. Hit this against the real file.
        meta = pd.DataFrame.from_records(_read_jsonl(Path(metadata_path)))
        item_metadata = pd.DataFrame(
            {
                "item_id": meta["parent_asin"].astype(str),
                "description": meta["title"],
            }
        )
        item_metadata = item_metadata.drop_duplicates(subset="item_id").reset_index(drop=True)
    elif "title" in reviews.columns:
        item_metadata = pd.DataFrame(
            {
                "item_id": reviews[item_col].astype(str),
                "description": reviews["title"],
            }
        )
        item_metadata = item_metadata.drop_duplicates(subset="item_id").reset_index(drop=True)

    return interactions, item_metadata
