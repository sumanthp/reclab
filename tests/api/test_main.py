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


def test_profile_without_metadata_has_no_item_text(client):
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
    assert resp.json()["profile"]["has_item_text"] is False


def test_profile_with_metadata_reflects_item_text(client):
    # Regression test: /profile used to have no item_metadata_csv parameter
    # at all, so has_item_text was always False and hybrid_llm's rationale
    # was always penalized for "no item text metadata" even when the caller
    # was about to provide it to /compare a moment later.
    interactions, item_metadata = _tiny_dataset()
    resp = client.post(
        "/profile",
        files={
            "interactions_csv": (
                "interactions.csv",
                io.BytesIO(_csv_bytes(interactions)),
                "text/csv",
            ),
            "item_metadata_csv": ("meta.csv", io.BytesIO(_csv_bytes(item_metadata)), "text/csv"),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["profile"]["has_item_text"] is True
    hybrid = next(r for r in body["recommendations"] if r["architecture"] == "hybrid_llm")
    assert "no item text metadata" not in hybrid["rationale"]


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


def test_list_runs_returns_summaries_newest_first(client):
    interactions, _ = _tiny_dataset()
    csv_bytes = _csv_bytes(interactions)

    for _ in range(2):
        client.post(
            "/compare",
            files={"interactions_csv": ("interactions.csv", io.BytesIO(csv_bytes), "text/csv")},
        )

    resp = client.get("/runs")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    # summaries, not full results
    assert "result" not in body[0]
    assert "error" not in body[0]
    assert body[0]["status"] == "done"
    timestamps = [r["created_at"] for r in body]
    assert timestamps == sorted(timestamps, reverse=True)


def test_cancel_unknown_run_404s(client):
    resp = client.post("/runs/does-not-exist/cancel")
    assert resp.status_code == 404


def test_cancel_already_finished_run_409s(client):
    interactions, _ = _tiny_dataset()
    start = client.post(
        "/compare",
        files={
            "interactions_csv": (
                "interactions.csv",
                io.BytesIO(_csv_bytes(interactions)),
                "text/csv",
            )
        },
    )
    job_id = start.json()["job_id"]  # already "done" — TestClient runs it synchronously

    resp = client.post(f"/runs/{job_id}/cancel")
    assert resp.status_code == 409
    assert "already finished" in resp.json()["detail"]


def test_cancel_pending_run_marks_it_cancelled(client):
    from reclab.api import jobs as jobs_module

    job_id = jobs_module.create_job(dataset_label="never-started.csv")

    resp = client.post(f"/runs/{job_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"

    # cancelling an already-cancelled run is a 409, not silently accepted twice
    again = client.post(f"/runs/{job_id}/cancel")
    assert again.status_code == 409


def test_compare_job_does_not_start_if_cancelled_before_it_ran():
    from reclab.api import jobs as jobs_module
    from reclab.api.main import _run_compare_job

    interactions, _ = _tiny_dataset()
    job_id = jobs_module.create_job(dataset_label="test")
    jobs_module.mark_cancelled(job_id)  # simulates a cancel request beating the thread pool

    _run_compare_job(job_id, interactions, None, "user_id", "item_id", 10)

    job = jobs_module.get_job(job_id)
    assert job.status == "cancelled"
    assert job.result is None  # never touched — mark_running must not have run


def test_compare_job_stops_between_architectures_when_cancelled(monkeypatch):
    from reclab.api import jobs as jobs_module
    from reclab.api.main import _run_compare_job

    interactions, item_metadata = _tiny_dataset()
    job_id = jobs_module.create_job(dataset_label="test")
    real_get_job = jobs_module.get_job
    calls = {"n": 0}

    def fake_get_job(jid):
        calls["n"] += 1
        # Let the top-of-function check and the first in-loop check see the
        # real (not-yet-cancelled) state so one architecture actually
        # trains, then report "cancelled" from then on.
        if calls["n"] > 2:
            return jobs_module.Job(
                id=jid,
                status="cancelled",
                dataset_label="test",
                created_at="x",
                updated_at="x",
                result=None,
                error=None,
            )
        return real_get_job(jid)

    monkeypatch.setattr(jobs_module, "get_job", fake_get_job)

    _run_compare_job(job_id, interactions, item_metadata, "user_id", "item_id", 10)

    final = real_get_job(job_id)
    assert final.status == "cancelled"
    assert final.result is not None
    assert 0 < len(final.result["eval_results"]) < 3
    assert final.result["comparison"] is None
