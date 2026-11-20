"""Illness screening endpoints: predicted activity against the proteins linked to a disease."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from app.services import indications as service

router = APIRouter()


class ScreenInput(BaseModel):
    smiles: str = Field(..., min_length=1, max_length=600)
    disease_id: str | None = Field(None, description="omit to rank every illness")

    @field_validator("smiles")
    @classmethod
    def strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("smiles cannot be empty")
        return v


class MatrixInput(BaseModel):
    smiles: list[str] = Field(..., min_length=1, max_length=25)


@router.get("/diseases")
async def diseases():
    """The illnesses in the knowledge base, with how many of their proteins have a model."""
    return {"diseases": service.diseases(), "disclaimer": service.DISCLAIMER}


@router.get("/diseases/{disease_id}")
async def disease_detail(disease_id: str):
    """Associated proteins with their evidence scores, and the drugs already approved for it."""
    try:
        return await run_in_threadpool(service.disease_detail, disease_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/screen")
async def screen(body: ScreenInput):
    """Predicted protein-target activity for one molecule, per illness.

    With `disease_id`: the breakdown for that illness. Without: every illness ranked.
    """
    try:
        return await run_in_threadpool(service.screen, body.smiles, body.disease_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/matrix")
async def matrix(body: MatrixInput):
    """Molecules x illnesses screening grid."""
    try:
        return await run_in_threadpool(service.matrix, body.smiles)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
