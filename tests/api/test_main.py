from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from reclab.api.main import app
from reclab.datasets.synthetic import SyntheticConfig, generate_synthetic_dataset


@pytest.fixture(autouse=True)
def _isolated_job_store(tmp_path, monkeypatch):
    monkeypatch.setenv("RECLAB_STORAGE", f"sqlite:///{tmp_path}/test-runs.db")


@pytest.fixture
def client():
    return TestClient(app)


def _tiny_dataset():
    cfg = SyntheticConfig(n_users=15, n_items=20, median_sequence_length=6, seed=1)
    return generate_synthetic_dataset(cfg)


def _csv_bytes(df) -> bytes:
    return df.to_csv(index=False).encode()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_architectures(client):
    resp = client.get("/architectures")
    assert resp.status_code == 200
    names = {a["name"] for a in resp.json()}
    assert names == {"two_tower", "sasrec", "hybrid_llm"}


def test_profile_rejects_missing_columns(client):
    csv = b"a,b\n1,2\n"
    resp = client.post(
        "/profile", files={"interactions_csv": ("bad.csv", io.BytesIO(csv), "text/csv")}
    )
    assert resp.status_code == 400
    assert "user_id" in resp.json()["detail"]


def test_profile_returns_shortlist(client):
    interactions, _ = _tiny_dataset()
    resp = client.post(
        "/profile",
        files={
            "interactions_csv": (
                "interactions.csv",
                io.BytesIO(_csv_bytes(interactions)),
                "text/csv",
            )
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["profile"]["n_users"] == 15
    assert {r["architecture"] for r in body["recommendations"]} == {
        "two_tower",
        "sasrec",
        "hybrid_llm",
    }
    ranks = [r["rank"] for r in body["recommendations"]]
    assert ranks == sorted(ranks)


def test_compare_requires_timestamp_column(client):
    csv = b"user_id,item_id\nu1,i1\n"
    resp = client.post(
        "/compare", files={"interactions_csv": ("no_ts.csv", io.BytesIO(csv), "text/csv")}
    )
    assert resp.status_code == 400
    assert "timestamp" in resp.json()["detail"]


def test_compare_end_to_end_and_poll_run(client):
    interactions, item_metadata = _tiny_dataset()

    start = client.post(
        "/compare",
        files={
            "interactions_csv": (
                "interactions.csv",
                io.BytesIO(_csv_bytes(interactions)),
                "text/csv",
            ),
            "item_metadata_csv": (
                "meta.csv",
                io.BytesIO(_csv_bytes(item_metadata)),
                "text/csv",
            ),
        },
    )
    assert start.status_code == 202
    job_id = start.json()["job_id"]

    # TestClient runs the ASGI app (including BackgroundTasks) synchronously
    # within the request, so the job is already done by the time this returns.
    run = client.get(f"/runs/{job_id}")
    assert run.status_code == 200
    body = run.json()
    assert body["status"] == "done"
    assert body["error"] is None

    result = body["result"]
    assert set(result["eval_results"]) == {"two_tower", "sasrec", "hybrid_llm"}
    for arch_result in result["eval_results"].values():
        assert "recall_at_k" in arch_result

    comparison = result["comparison"]
    assert comparison["shortlist_pick"] in {"two_tower", "sasrec", "hybrid_llm"}
    assert comparison["matches_on_recall"] in (True, False)


def test_run_not_found(client):
    resp = client.get("/runs/does-not-exist")
    assert resp.status_code == 404
