"""Tests against small locally-built fixtures that mimic the real MovieLens
100K and Amazon Reviews 2023 file formats — both loaders have also been run
against real downloads (see benchmarks/README.md)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from reclab.datasets.loaders import load_amazon_reviews_category, load_movielens_100k

U_DATA_FIXTURE = "196\t242\t3\t881250949\n186\t302\t3\t891717742\n22\t377\t1\t878887116\n"
U_ITEM_FIXTURE = (
    "242|Kolya (1996)|01-Jan-1997||http://example.com|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0\n"
    "302|L.A. Confidential (1997)|01-Jan-1997||http://example.com|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0\n"
    "377|Heavyweights (1994)|01-Jan-1994||http://example.com|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0\n"
)


def _write_fixture_dir(tmp_path: Path) -> Path:
    ml_dir = tmp_path / "ml-100k"
    ml_dir.mkdir()
    (ml_dir / "u.data").write_text(U_DATA_FIXTURE)
    (ml_dir / "u.item").write_bytes(U_ITEM_FIXTURE.encode("latin-1"))
    return ml_dir


def test_load_movielens_100k_from_directory(tmp_path):
    ml_dir = _write_fixture_dir(tmp_path)
    interactions, item_metadata = load_movielens_100k(ml_dir)

    assert list(interactions.columns) == ["user_id", "item_id", "timestamp"]
    assert len(interactions) == 3
    assert set(interactions["user_id"]) == {"196", "186", "22"}

    assert list(item_metadata.columns) == ["item_id", "description"]
    assert len(item_metadata) == 3
    row = item_metadata[item_metadata["item_id"] == "242"].iloc[0]
    assert row["description"] == "Kolya (1996)"


def test_load_movielens_100k_from_zip(tmp_path):
    import zipfile

    ml_dir = _write_fixture_dir(tmp_path)
    zip_path = tmp_path / "ml-100k.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(ml_dir / "u.data", arcname="ml-100k/u.data")
        zf.write(ml_dir / "u.item", arcname="ml-100k/u.item")

    interactions, item_metadata = load_movielens_100k(zip_path)

    assert len(interactions) == 3
    assert len(item_metadata) == 3


AMAZON_5CORE_CSV_FIXTURE = (
    "user_id,parent_asin,rating,timestamp\n"
    "u1,B001,5.0,1547589356557\n"
    "u1,B002,4.0,1593352422858\n"
    "u2,B001,3.0,1596473351088\n"
)

AMAZON_RAW_REVIEWS_FIXTURE = [
    {
        "rating": 5.0,
        "title": "Great!",
        "text": "Loved it.",
        "asin": "B001",
        "parent_asin": "B001",
        "user_id": "u1",
        "timestamp": 1547589356557,
    },
    {
        "rating": 4.0,
        "title": "Pretty good",
        "text": "Works fine.",
        "asin": "B002",
        "parent_asin": "B002",
        "user_id": "u1",
        "timestamp": 1593352422858,
    },
]

AMAZON_META_FIXTURE = [
    {"parent_asin": "B001", "title": "Widget Pro", "average_rating": 4.5},
    {"parent_asin": "B002", "title": "Widget Lite", "average_rating": 4.0},
]


def test_load_amazon_reviews_from_5core_csv(tmp_path):
    reviews_path = tmp_path / "All_Beauty.csv"
    reviews_path.write_text(AMAZON_5CORE_CSV_FIXTURE)

    interactions, item_metadata = load_amazon_reviews_category(reviews_path)

    assert list(interactions.columns) == ["user_id", "item_id", "timestamp"]
    assert len(interactions) == 3
    assert set(interactions["item_id"]) == {"B001", "B002"}
    assert item_metadata is None  # no title/text in the 5-core rating-only format


def test_load_amazon_reviews_from_5core_csv_with_metadata(tmp_path):
    reviews_path = tmp_path / "All_Beauty.csv"
    reviews_path.write_text(AMAZON_5CORE_CSV_FIXTURE)
    metadata_path = tmp_path / "meta_All_Beauty.jsonl"
    metadata_path.write_text("\n".join(json.dumps(r) for r in AMAZON_META_FIXTURE))

    interactions, item_metadata = load_amazon_reviews_category(reviews_path, metadata_path)

    assert list(item_metadata.columns) == ["item_id", "description"]
    assert len(item_metadata) == 2
    row = item_metadata[item_metadata["item_id"] == "B001"].iloc[0]
    assert row["description"] == "Widget Pro"


def test_load_amazon_reviews_from_raw_jsonl_falls_back_to_review_title(tmp_path):
    reviews_path = tmp_path / "All_Beauty.jsonl"
    reviews_path.write_text("\n".join(json.dumps(r) for r in AMAZON_RAW_REVIEWS_FIXTURE))

    interactions, item_metadata = load_amazon_reviews_category(reviews_path)

    assert len(interactions) == 2
    assert item_metadata is not None
    row = item_metadata[item_metadata["item_id"] == "B001"].iloc[0]
    assert row["description"] == "Great!"


def test_load_amazon_reviews_from_gzipped_jsonl(tmp_path):
    reviews_path = tmp_path / "All_Beauty.jsonl.gz"
    with gzip.open(reviews_path, "wt", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r) for r in AMAZON_RAW_REVIEWS_FIXTURE))

    interactions, _ = load_amazon_reviews_category(reviews_path)

    assert len(interactions) == 2
