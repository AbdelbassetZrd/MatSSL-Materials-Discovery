"""Molecular toxicity endpoints (SMILES in, predicted toxicity profile out)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from app.services import molecules as service

router = APIRouter()


class SMILESInput(BaseModel):
    smiles: str = Field(
        ...,
        min_length=1,
        max_length=600,
        description="SMILES string, e.g. CC(=O)Oc1ccccc1C(=O)O (aspirin)",
    )

    @field_validator("smiles")
    @classmethod
    def strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("smiles cannot be empty")
        return v


class BatchSMILESInput(BaseModel):
    smiles: list[str] = Field(..., min_length=1, max_length=50)


@router.get("/models-info")
async def models_info():
    """Deployed model for every toxicity endpoint, with its method and test metrics."""
    return {"models": service.models_info()}


@router.post("/predict")
async def predict(body: SMILESInput):
    """Toxicity profile of one molecule: 12 Tox21 assays, clinical toxicity and approval."""
    try:
        return await run_in_threadpool(service.predict, body.smiles)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/batch-predict")
async def batch_predict(body: BatchSMILESInput):
    results, errors = [], []
    for s in body.smiles:
        try:
            results.append(await run_in_threadpool(service.predict, s.strip()))
        except ValueError as exc:
            errors.append({"smiles": s, "error": str(exc)})
    return {"results": results, "errors": errors}


@router.post("/properties")
async def properties(body: SMILESInput):
    """Drug-likeness properties only (no model needed): weight, logP, TPSA, QED, Lipinski."""
    try:
        return await run_in_threadpool(service.properties, body.smiles)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
