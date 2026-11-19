"""Materials prediction endpoints (counterpart of the friend's /qsar routes)."""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from app.services import model as service

logger = logging.getLogger(__name__)
router = APIRouter()

OPTIMADE = "https://alexandria.icams.rub.de/pbe/v1/structures"


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
        return service.predict(body.formula)
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
    (used by the 3D viewer). Returns 404 if the composition is unknown."""
    from pymatgen.core import Composition

    try:
        reduced = Composition(formula).reduced_formula
        amounts = Composition(formula).reduced_composition.get_el_amt_dict()
        # OPTIMADE reduced formula: elements in alphabetical order, counts of 1 omitted
        hill = "".join(el + ("" if n == 1 else f"{n:g}") for el, n in sorted(amounts.items()))
    except Exception as exc:
        raise HTTPException(status_code=422, detail="invalid formula") from exc
    params = {
        "filter": f'chemical_formula_reduced="{hill}"',
        "response_fields": "lattice_vectors,cartesian_site_positions,species_at_sites,"
        "_alexandria_hull_distance",
        "page_limit": 20,
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(OPTIMADE, params=params)
            r.raise_for_status()
            data = r.json().get("data", [])
    except Exception as exc:
        logger.warning("OPTIMADE lookup failed: %s", exc)
        raise HTTPException(status_code=503, detail="structure service unavailable") from exc
    if not data:
        raise HTTPException(status_code=404, detail=f"no known structure for {reduced}")
    best = min(data, key=lambda d: d["attributes"].get("_alexandria_hull_distance") or 9)
    a = best["attributes"]
    lines = [str(len(a["species_at_sites"])), f"{reduced} {best['id']}"]
    lines += [
        f"{s} {x:.5f} {y:.5f} {z:.5f}"
        for s, (x, y, z) in zip(a["species_at_sites"], a["cartesian_site_positions"], strict=True)
    ]
    return {
        "formula": reduced,
        "id": best["id"],
        "xyz": "\n".join(lines),
        "lattice_vectors": a["lattice_vectors"],
        "e_above_hull": a.get("_alexandria_hull_distance"),
    }
