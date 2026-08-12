"""FastAPI app entrypoint.

Run with: uv run uvicorn reclab.api.main:app --reload
"""

from __future__ import annotations

import io
import os
from dataclasses import asdict

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from reclab.api import jobs
from reclab.architectures import REGISTRY
from reclab.data_profiler import DataProfile, profile_interactions
from reclab.eval import EvalResult, run_eval, summarize_comparison, temporal_train_test_split
from reclab.reasoning_engine import Recommendation, recommend_architectures

app = FastAPI(
    title="reclab",
    description=(
        "Reason about, configure, and evaluate recommendation system "
        "architectures on your own data."
    ),
    version="0.1.0",
)

# Permissive by default: this is a self-hosted, single-user local tool with
# no auth and no sensitive data beyond what the user uploads themselves —
# not a multi-tenant service. Override RECLAB_CORS_ORIGINS (comma-separated)
# for a locked-down deployment.
_cors_origins = os.environ.get("RECLAB_CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _cors_origins == "*" else _cors_origins.split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProfileResponse(BaseModel):
    profile: dict
    recommendations: list[dict]


class CompareStartResponse(BaseModel):
    job_id: str


class RunResponse(BaseModel):
    id: str
    status: str
    dataset_label: str | None
    created_at: str
    updated_at: str
    result: dict | None
    error: str | None


class RunSummary(BaseModel):
    """Lightweight — no `result`/`error` — so GET /runs stays cheap to list
    even once individual runs' eval_results grow. Fetch GET /runs/{id} for
    the full detail of one run."""

    id: str
    status: str
    dataset_label: str | None
    created_at: str
    updated_at: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/architectures")
def list_architectures() -> list[dict]:
    """List every registered candidate architecture with its static info."""
    return [arch.info().__dict__ for arch in REGISTRY.values()]


async def _read_csv(upload: UploadFile, label: str) -> pd.DataFrame:
    raw = await upload.read()
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"could not parse {label}: {exc}") from exc


@app.post("/profile", response_model=ProfileResponse)
async def profile_dataset(
    interactions_csv: UploadFile,
    item_metadata_csv: UploadFile | None = None,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> ProfileResponse:
    """Profile an uploaded interactions CSV and return the reasoning engine's
    ranked architecture shortlist for it.

    `item_metadata_csv` is optional but matters: without it, `has_item_text`
    is always false and the shortlist's `hybrid_llm` rationale is always
    penalized for "no item text metadata" — even if the caller is about to
    provide metadata to /compare a moment later. Expects the same `item_id`/
    `description` shape /compare does.

    Phase 0 scope: local CSV upload only. Cloud/warehouse connectors come
    later — see docs/architecture/mvp-plan.md.
    """
    df = await _read_csv(interactions_csv, "interactions CSV")

    if user_col not in df.columns or item_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"expected columns '{user_col}' and '{item_col}' in uploaded CSV",
        )

    item_metadata = None
    if item_metadata_csv is not None:
        item_metadata = await _read_csv(item_metadata_csv, "item metadata CSV")

    try:
        profile: DataProfile = profile_interactions(
            df,
            user_col=user_col,
            item_col=item_col,
            item_metadata=item_metadata,
            text_col="description" if item_metadata is not None else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    recommendations: list[Recommendation] = recommend_architectures(profile)

    return ProfileResponse(
        profile=profile.__dict__,
        recommendations=[r.__dict__ for r in recommendations],
    )


def _run_compare_job(
    job_id: str,
    interactions: pd.DataFrame,
    item_metadata: pd.DataFrame | None,
    user_col: str,
    item_col: str,
    k: int,
) -> None:
    """Runs in a background thread (via FastAPI's BackgroundTasks) — this is
    the same profile -> shortlist -> train -> eval -> compare pipeline as
    scripts/run_benchmark.py, just reporting into the job store instead of
    stdout. Any exception here must be captured, not raised, since nothing
    downstream is listening for it — mark_error is the only way the caller
    finds out.

    Cancellation is cooperative and coarse-grained: a single architecture's
    training loop can't be interrupted mid-flight (that would need
    restructuring every architecture's fit() to check a flag between
    epochs), but the job checks for a cancellation request before starting
    each architecture and stops there, persisting whatever finished so far.
    """
    if (current := jobs.get_job(job_id)) is not None and current.status == "cancelled":
        return  # cancelled before this background task even started running
    jobs.mark_running(job_id)
    try:
        profile = profile_interactions(
            interactions,
            user_col=user_col,
            item_col=item_col,
            item_metadata=item_metadata,
            text_col="description" if item_metadata is not None else None,
        )
        shortlist = recommend_architectures(profile)
        train, test = temporal_train_test_split(
            interactions, user_col=user_col, item_col=item_col
        )

        eval_results: dict[str, dict] = {}
        eval_results_objs: dict[str, EvalResult] = {}
        for name, arch_cls in REGISTRY.items():
            if (current := jobs.get_job(job_id)) is not None and current.status == "cancelled":
                jobs.mark_cancelled(
                    job_id,
                    {
                        "profile": asdict(profile),
                        "reasoning_engine_shortlist": [asdict(r) for r in shortlist],
                        "eval_results": eval_results,
                        # Not enough architectures ran for a fair verdict —
                        # a "measured best" out of one or two partial
                        # results would misrepresent the comparison.
                        "comparison": None,
                    },
                )
                return
            try:
                result = run_eval(
                    arch_cls(),
                    train,
                    test,
                    item_metadata=item_metadata,
                    k=k,
                    user_col=user_col,
                    item_col=item_col,
                )
                eval_results[name] = asdict(result)
                eval_results_objs[name] = result
            except ValueError as exc:
                eval_results[name] = {"skipped": str(exc)}

        comparison = summarize_comparison(shortlist, eval_results_objs)

        jobs.mark_done(
            job_id,
            {
                "profile": asdict(profile),
                "reasoning_engine_shortlist": [asdict(r) for r in shortlist],
                "eval_results": eval_results,
                "comparison": asdict(comparison),
            },
        )
    except Exception as exc:  # noqa: BLE001 - must not raise inside a background task
        jobs.mark_error(job_id, str(exc))


@app.post("/compare", response_model=CompareStartResponse, status_code=202)
async def start_compare(
    background_tasks: BackgroundTasks,
    interactions_csv: UploadFile,
    item_metadata_csv: UploadFile | None = None,
    user_col: str = "user_id",
    item_col: str = "item_id",
    k: int = 10,
) -> CompareStartResponse:
    """Kick off training + evaluating every registered architecture on the
    uploaded data, and check the reasoning engine's shortlist against what
    actually wins. Returns immediately with a job id — poll GET /runs/{id}
    for status and results, since this can take well over a minute (see
    docs/architecture/ui-ux-plan.md section 4)."""
    interactions = await _read_csv(interactions_csv, "interactions CSV")

    if user_col not in interactions.columns or item_col not in interactions.columns:
        raise HTTPException(
            status_code=400,
            detail=f"expected columns '{user_col}' and '{item_col}' in uploaded CSV",
        )
    if "timestamp" not in interactions.columns:
        raise HTTPException(
            status_code=400,
            detail="interactions CSV needs a 'timestamp' column for /compare "
            "(temporal_train_test_split requires one)",
        )

    item_metadata = None
    if item_metadata_csv is not None:
        item_metadata = await _read_csv(item_metadata_csv, "item metadata CSV")

    job_id = jobs.create_job(dataset_label=interactions_csv.filename)
    background_tasks.add_task(
        _run_compare_job, job_id, interactions, item_metadata, user_col, item_col, k
    )
    return CompareStartResponse(job_id=job_id)


@app.get("/runs", response_model=list[RunSummary])
def list_runs(limit: int = 50) -> list[RunSummary]:
    return [
        RunSummary(
            id=j.id,
            status=j.status,
            dataset_label=j.dataset_label,
            created_at=j.created_at,
            updated_at=j.updated_at,
        )
        for j in jobs.list_jobs(limit=limit)
    ]


@app.get("/runs/{job_id}", response_model=RunResponse)
def get_run(job_id: str) -> RunResponse:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no run with id {job_id}")
    return RunResponse(**asdict(job))


@app.post("/runs/{job_id}/cancel", response_model=RunResponse)
def cancel_run(job_id: str) -> RunResponse:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no run with id {job_id}")
    if job.status in ("done", "error", "cancelled"):
        raise HTTPException(
            status_code=409, detail=f"run already finished with status '{job.status}'"
        )
    jobs.mark_cancelled(job_id)
    updated = jobs.get_job(job_id)
    assert updated is not None  # just wrote it above
    return RunResponse(**asdict(updated))
