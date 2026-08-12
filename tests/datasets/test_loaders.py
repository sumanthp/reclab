"""Tests against small locally-built fixtures that mimic the real MovieLens
100K file format, since the real dataset isn't reachable from this
environment (see loaders.py module docstring)."""

from __future__ import annotations

from pathlib import Path

from reclab.datasets.loaders import load_movielens_100k

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
