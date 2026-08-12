from __future__ import annotations

import pytest

from reclab.api import jobs


@pytest.fixture(autouse=True)
def _isolated_job_store(tmp_path, monkeypatch):
    monkeypatch.setenv("RECLAB_STORAGE", f"sqlite:///{tmp_path}/jobs-test.db")


def test_create_and_get_job():
    job_id = jobs.create_job(dataset_label="foo.csv")
    job = jobs.get_job(job_id)

    assert job is not None
    assert job.status == "pending"
    assert job.dataset_label == "foo.csv"
    assert job.result is None
    assert job.error is None


def test_get_unknown_job_returns_none():
    assert jobs.get_job("does-not-exist") is None


def test_mark_running_then_done():
    job_id = jobs.create_job()
    jobs.mark_running(job_id)
    assert jobs.get_job(job_id).status == "running"

    jobs.mark_done(job_id, {"profile": {"n_users": 1}})
    job = jobs.get_job(job_id)
    assert job.status == "done"
    assert job.result == {"profile": {"n_users": 1}}


def test_mark_error():
    job_id = jobs.create_job()
    jobs.mark_error(job_id, "boom")
    job = jobs.get_job(job_id)
    assert job.status == "error"
    assert job.error == "boom"


def test_mark_cancelled_without_result_is_a_request_marker():
    job_id = jobs.create_job()
    jobs.mark_cancelled(job_id)
    job = jobs.get_job(job_id)
    assert job.status == "cancelled"
    assert job.result is None


def test_mark_cancelled_with_result_persists_partial_progress():
    job_id = jobs.create_job()
    jobs.mark_cancelled(job_id, {"eval_results": {"two_tower": {"recall_at_k": 0.1}}})
    job = jobs.get_job(job_id)
    assert job.status == "cancelled"
    assert job.result == {"eval_results": {"two_tower": {"recall_at_k": 0.1}}}


def test_list_jobs_orders_newest_first():
    first = jobs.create_job(dataset_label="a.csv")
    second = jobs.create_job(dataset_label="b.csv")

    listed = jobs.list_jobs()

    assert [j.id for j in listed] == [second, first]


def test_list_jobs_respects_limit():
    for i in range(5):
        jobs.create_job(dataset_label=f"{i}.csv")

    assert len(jobs.list_jobs(limit=2)) == 2


def test_create_job_prunes_beyond_retention_limit(monkeypatch):
    monkeypatch.setattr(jobs, "MAX_RETAINED_RUNS", 3)

    ids = [jobs.create_job(dataset_label=f"{i}.csv") for i in range(5)]

    remaining = {j.id for j in jobs.list_jobs(limit=100)}
    assert remaining == set(ids[-3:])  # oldest 2 pruned, newest 3 kept
    assert jobs.get_job(ids[0]) is None
