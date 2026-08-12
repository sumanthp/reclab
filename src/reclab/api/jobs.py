"""SQLite-backed job store for long-running training+eval runs.

The /compare endpoint trains and evaluates every registered architecture,
which is too slow to do inline in a single HTTP request (see
docs/architecture/ui-ux-plan.md section 4 — MovieLens 100K took long enough
during Phase 0 validation that it had to run in the background). A
single-process, single-user local tool doesn't need a real task queue;
SQLite is enough to serve GET /runs/{id} without holding job state only in
memory, and to survive a server restart mid-job (the job is just reported
as still "running" — no resume, which is fine at this scale).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

JobStatus = Literal["pending", "running", "done", "error"]

_lock = threading.Lock()


def _db_path() -> Path:
    storage = os.environ.get("RECLAB_STORAGE", "sqlite:///./reclab.db")
    prefix = "sqlite:///"
    if not storage.startswith(prefix):
        raise ValueError(f"RECLAB_STORAGE must be a sqlite:/// URL, got: {storage!r}")
    return Path(storage[len(prefix) :])


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                dataset_label TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                result_json TEXT,
                error TEXT
            )
            """
        )
        yield conn
        conn.commit()
    finally:
        conn.close()


@dataclass
class Job:
    id: str
    status: JobStatus
    dataset_label: str | None
    created_at: str
    updated_at: str
    result: dict[str, Any] | None
    error: str | None


def create_job(dataset_label: str | None = None) -> str:
    job_id = uuid.uuid4().hex
    now = datetime.now(UTC).isoformat()
    with _lock, _conn() as conn:
        conn.execute(
            "INSERT INTO runs (id, status, dataset_label, created_at, updated_at) "
            "VALUES (?, 'pending', ?, ?, ?)",
            (job_id, dataset_label, now, now),
        )
    return job_id


def mark_running(job_id: str) -> None:
    with _lock, _conn() as conn:
        conn.execute(
            "UPDATE runs SET status = 'running', updated_at = ? WHERE id = ?",
            (datetime.now(UTC).isoformat(), job_id),
        )


def mark_done(job_id: str, result: dict[str, Any]) -> None:
    with _lock, _conn() as conn:
        conn.execute(
            "UPDATE runs SET status = 'done', result_json = ?, updated_at = ? WHERE id = ?",
            (json.dumps(result), datetime.now(UTC).isoformat(), job_id),
        )


def mark_error(job_id: str, error: str) -> None:
    with _lock, _conn() as conn:
        conn.execute(
            "UPDATE runs SET status = 'error', error = ?, updated_at = ? WHERE id = ?",
            (error, datetime.now(UTC).isoformat(), job_id),
        )


def get_job(job_id: str) -> Job | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT id, status, dataset_label, created_at, updated_at, result_json, error "
            "FROM runs WHERE id = ?",
            (job_id,),
        ).fetchone()
    if row is None:
        return None
    return Job(
        id=row[0],
        status=row[1],
        dataset_label=row[2],
        created_at=row[3],
        updated_at=row[4],
        result=json.loads(row[5]) if row[5] else None,
        error=row[6],
    )
