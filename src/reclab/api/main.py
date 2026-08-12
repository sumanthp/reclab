"""FastAPI app entrypoint.

Run with: uv run uvicorn reclab.api.main:app --reload
"""

from __future__ import annotations

import io

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from reclab.architectures import REGISTRY
from reclab.data_profiler import DataProfile, profile_interactions
from reclab.reasoning_engine import Recommendation, recommend_architectures

app = FastAPI(
    title="reclab",
    description=(
        "Reason about, configure, and evaluate recommendation system "
        "architectures on your own data."
    ),
    version="0.1.0",
)


class ProfileResponse(BaseModel):
    profile: dict
    recommendations: list[dict]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/architectures")
def list_architectures() -> list[dict]:
    """List every registered candidate architecture with its static info."""
    return [arch.info().__dict__ for arch in REGISTRY.values()]


@app.post("/profile", response_model=ProfileResponse)
async def profile_dataset(
    interactions_csv: UploadFile,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> ProfileResponse:
    """Profile an uploaded interactions CSV and return the reasoning engine's
    ranked architecture shortlist for it.

    Phase 0 scope: local CSV upload only. Cloud/warehouse connectors come
    later — see docs/architecture/mvp-plan.md.
    """
    raw = await interactions_csv.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"could not parse CSV: {exc}") from exc

    if user_col not in df.columns or item_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"expected columns '{user_col}' and '{item_col}' in uploaded CSV",
        )

    try:
        profile: DataProfile = profile_interactions(df, user_col=user_col, item_col=item_col)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    recommendations: list[Recommendation] = recommend_architectures(profile)

    return ProfileResponse(
        profile=profile.__dict__,
        recommendations=[r.__dict__ for r in recommendations],
    )
