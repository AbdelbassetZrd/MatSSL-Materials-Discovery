"""Materials prediction endpoints (counterpart of the friend's /qsar routes)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from app.services import model as service
from app.services import structure as structures

logger = logging.getLogger(__name__)
router = APIRouter()


class FormulaInput(BaseModel):
    formula: str = Field(
        ..., min_length=1, max_length=100, description="Chemical formula, e.g. LiFePO4"
    )

    @field_validator("formula")
    @classmethod
    def strip(cls, v: str) -> str:
        v = v.strip().replace(" ", "")
        if not v:
            raise ValueError("formula cannot be empty")
        return v


class BatchInput(BaseModel):
    formulas: list[str] = Field(..., min_length=1, max_length=100)


@router.get("/models-info")
async def models_info():
    """Deployed model for every task, with its method and test metrics."""
    return {"models": [m.model_dump() for m in service.models_info()]}


@router.post("/predict")
async def predict(body: FormulaInput):
    """Predict metallicity, magnetism, stability and PV band-gap class for a formula."""
    try:
        return await run_in_threadpool(service.predict, body.formula)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/batch-predict")
async def batch_predict(body: BatchInput):
    results, errors = [], []
    for f in body.formulas:
        try:
            results.append(service.predict(f.strip()))
        except ValueError as exc:
            errors.append({"formula": f, "error": str(exc)})
    return {"results": results, "errors": errors}


@router.post("/analyze")
async def analyze(body: FormulaInput):
    """Composition analysis: element fractions, descriptors, chemical/association rules."""
    try:
        return service.analyze(body.formula)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/similar")
async def similar(
    formula: str = Query(..., min_length=1, max_length=100), k: int = Query(8, ge=1, le=25)
):
    """Nearest known (labeled) compositions in the self-supervised embedding space."""
    try:
        return service.similar(formula.strip(), k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/experiments")
async def experiments():
    """All experiment results (datasets, SSL comparison, GA, unsupervised) for the dashboard."""
    try:
        return service.experiments()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/structure")
async def structure(formula: str = Query(..., min_length=1, max_length=100)):
    """Lowest-energy crystal structure for the formula from the Alexandria OPTIMADE API
    (used by the 3D viewer and the structure-aware models). 404 if the composition is
    unknown, 503 if the database cannot be reached."""
    try:
        reduced, _ = structures.names(formula)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="invalid formula") from exc
    try:
        s = await run_in_threadpool(structures.ground_state, formula)
    except structures.StructureServiceError as exc:
        logger.warning("OPTIMADE lookup failed: %s", exc)
        raise HTTPException(status_code=503, detail="structure service unavailable") from exc
    if s is None:
        raise HTTPException(status_code=404, detail=f"no known structure for {reduced}")
    return {
        "formula": reduced,
        "id": s["id"],
        "xyz": structures.xyz(s),
        "lattice_vectors": s["lattice_vectors"],
        "e_above_hull": s.get("e_above_hull"),
    }
