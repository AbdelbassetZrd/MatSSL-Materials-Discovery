"""Design Lab service: score a composition for a target application, and suggest candidates.

Two questions:
  * "How good is *my* composition as a superconductor / cathode / absorber / magnet?"
    -> every deployed property model is run on the formula and combined by the application
       score (pipeline/industries/applications.py), which always returns its own breakdown.
  * "Suggest me one."
    -> the 554k unlabeled GNoME compositions are pre-scored offline by
       pipeline/industries/rank_candidates.py; this service just serves the ranking.
"""

from __future__ import annotations

import json
import logging
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "app" / "models"
DESIGN_DIR = ROOT / "data" / "predictions" / "design"
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "pipeline" / "industries"))

from applications import APPLICATIONS  # noqa: E402
from applications import score as app_score  # noqa: E402
from featurize import parse_formula  # noqa: E402
from hiacc_model import StackedRegressor  # noqa: E402

from app.services import model as materials  # noqa: E402  (after sys.path setup)

logger = logging.getLogger(__name__)
_main = sys.modules["__main__"]
if not hasattr(_main, "StackedRegressor"):
    _main.StackedRegressor = StackedRegressor

PROPERTY_NAME = {
    "metallicity": "metal",
    "magnetism": "magnetic",
    "stability": "stable",
    "band_gap": "pv_gap",
}


@lru_cache(maxsize=1)
def superconductor_model() -> dict | None:
    f = MODELS_DIR / "superconductor_model.pkl"
    if not f.exists():
        return None
    try:
        return joblib.load(f)
    except Exception as exc:  # pragma: no cover - logged for ops
        logger.error("could not load the superconductor model: %s", exc)
        return None


@lru_cache(maxsize=1)
def candidates() -> pd.DataFrame | None:
    f = DESIGN_DIR / "candidates.parquet"
    return pd.read_parquet(f) if f.exists() else None


def applications() -> list[dict]:
    """The catalogue shown by the Design Lab."""
    cand = candidates()
    # the stored table only keeps the best few hundred per application; the count of strong
    # candidates over the whole pool is recorded by the ranking script
    try:
        counts = {k: v["n_strong"] for k, v in summary()["applications"].items()}
    except (FileNotFoundError, KeyError):
        counts = {}
    out = []
    for key, cfg in APPLICATIONS.items():
        n = counts.get(key, 0)
        out.append(
            {
                "id": key,
                "title": cfg["title"],
                "subtitle": cfg["subtitle"],
                "question": cfg["question"],
                "note": cfg["note"],
                "examples": cfg["examples"],
                "needs": cfg["needs"],
                "n_strong_candidates": n,
                "available": cand is not None,
            }
        )
    return out


def _properties(formula: str) -> dict:
    """Every deployed property model evaluated on one composition."""
    reg = materials.get_registry()
    raw = materials._features(formula)
    props: dict[str, float] = {}
    for task in reg.models:
        X = materials._task_matrix(task, raw)
        props[PROPERTY_NAME.get(task, task)] = float(
            np.clip(reg.models[task]["model"].predict_proba(X)[0, 1], 0, 1)
        )
    pkg = superconductor_model()
    if pkg is not None:
        tc = float(pkg["model"].predict(pkg["preprocessor"].transform(raw))[0])
        props["tc"] = max(tc, 0.0)
    return props


def score_formula(formula: str) -> dict:
    """Score one composition for every application, with the property values behind it."""
    frac = parse_formula(formula)
    if frac is None:
        raise ValueError(f"Could not parse formula '{formula}'")
    props = _properties(formula)
    payload = {**props, "elements": set(frac)}
    results = [app_score(a, payload) for a in APPLICATIONS]
    results.sort(key=lambda r: -r["score"])
    return {
        "formula": formula,
        "elements": sorted(frac),
        "properties": {
            "stable": props.get("stable"),
            "metal": props.get("metal"),
            "magnetic": props.get("magnetic"),
            "pv_gap": props.get("pv_gap"),
            "tc_kelvin": props.get("tc"),
        },
        "applications": results,
        "best": results[0]["application"] if results else None,
    }


def top_candidates(
    application: str,
    limit: int = 40,
    elements: list[str] | None = None,
    exclude: list[str] | None = None,
) -> dict:
    """Best GNoME compositions for an application, optionally filtered by chemistry."""
    if application not in APPLICATIONS:
        raise ValueError(f"unknown application '{application}'")
    cand = candidates()
    if cand is None:
        raise FileNotFoundError(
            "candidate ranking not available - run pipeline/industries/rank_candidates.py"
        )
    sub = cand[cand.application == application].copy()
    if elements or exclude:
        els = sub["formula"].map(lambda f: set(parse_formula(f) or {}))
        if elements:
            sub = sub[els.map(lambda s: set(elements) <= s)]
        if exclude:
            sub = sub[els.map(lambda s: not (set(exclude) & s))]
    sub = sub.nlargest(limit, "score")
    rows = []
    for r in sub.itertuples(index=False):
        props = {k: getattr(r, k, None) for k in ("stable", "metal", "magnetic", "pv_gap", "tc")}
        detail = app_score(application, {**props, "elements": set(parse_formula(r.formula) or {})})
        rows.append(
            {
                "formula": r.formula,
                "score": float(r.score),
                "headline": detail["headline"],
                "verdict": detail["verdict"],
                "components": detail["components"],
                "properties": {k: (None if v is None else float(v)) for k, v in props.items()},
            }
        )
    cfg = APPLICATIONS[application]
    return {
        "application": application,
        "title": cfg["title"],
        "note": cfg["note"],
        "pool": "GNoME (554k unlabeled compositions proposed by a generative search)",
        "n_returned": len(rows),
        "candidates": rows,
    }


def summary() -> dict:
    f = DESIGN_DIR / "design_summary.json"
    if not f.exists():
        raise FileNotFoundError("run pipeline/industries/rank_candidates.py first")
    return json.loads(f.read_text(encoding="utf-8"))
