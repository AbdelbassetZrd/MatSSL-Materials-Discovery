"""Design Lab endpoints: score a composition for an application, or get ranked candidates."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from app.services import design as service

router = APIRouter()


class FormulaInput(BaseModel):
    formula: str = Field(..., min_length=1, max_length=100)

    @field_validator("formula")
    @classmethod
    def strip(cls, v: str) -> str:
        v = v.strip().replace(" ", "")
        if not v:
            raise ValueError("formula cannot be empty")
        return v


@router.get("/applications")
async def applications():
    """Target applications the Design Lab can score, and how many strong candidates exist."""
    return {"applications": service.applications()}


@router.post("/score")
async def score(body: FormulaInput):
    """Score one composition for every application, with the property values behind it."""
    try:
        return await run_in_threadpool(service.score_formula, body.formula)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/candidates")
async def candidates(
    application: str = Query(..., description="superconductor, solar_absorber, …"),
    limit: int = Query(40, ge=1, le=400),
    elements: str | None = Query(None, description="comma-separated, must all be present"),
    exclude: str | None = Query(None, description="comma-separated, must all be absent"),
):
    """Best unlabeled GNoME compositions for an application, filterable by chemistry."""
    split = lambda s: [e.strip() for e in s.split(",") if e.strip()] if s else None  # noqa: E731
    try:
        return await run_in_threadpool(
            service.top_candidates, application, limit, split(elements), split(exclude)
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
