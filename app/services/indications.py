"""Screen a molecule against the proteins associated with an illness.

The service holds one activity model per protein target. A molecule is featurised once, every
target model is applied, and the resulting **target-activity profile** is combined into an
illness ranking by `pipeline/molecules/indications.py`.

Because the profile is computed per target rather than per illness, asking "and what about
this other illness?" costs nothing: it is a different weighted sum over the same profile.
"""

from __future__ import annotations

import json
import logging
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "app" / "models"
PROC = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "pipeline" / "molecules"))

from indications import (  # noqa: E402
    DISCLAIMER,
    combine_with_toxicity,
    rank_diseases,
    score_disease,
)

from app.services import molecules as mol  # noqa: E402  (after sys.path setup)

logger = logging.getLogger(__name__)


class _Registry:
    def __init__(self) -> None:
        self.diseases: list[dict] = []
        self.targets: dict[str, dict] = {}
        f = PROC / "diseases.json"
        if f.exists():
            kb = json.loads(f.read_text(encoding="utf-8"))
            self.diseases = kb["diseases"]
        for p in sorted(MODELS_DIR.glob("target_*.pkl")):
            try:
                pkg = joblib.load(p)
                pkg["active_bits"] = np.unpackbits(pkg["active_fingerprints"], axis=1)
                self.targets[pkg["chembl_id"]] = pkg
            except Exception as exc:  # pragma: no cover - logged for ops
                logger.error("could not load %s: %s", p.name, exc)
        if self.targets:
            logger.info("loaded %d target activity models", len(self.targets))

    def disease(self, disease_id: str) -> dict | None:
        return next((d for d in self.diseases if d["id"] == disease_id), None)


@lru_cache(maxsize=1)
def get_registry() -> _Registry:
    return _Registry()


def _nearest_active(pkg: dict, bits: np.ndarray) -> tuple[float | None, str | None]:
    """Tanimoto similarity to the closest molecule the model saw as active, and which one."""
    act = pkg.get("active_bits")
    if act is None or len(act) == 0:
        return None, None
    inter = act @ bits
    union = act.sum(1) + bits.sum() - inter
    sim = inter / np.maximum(union, 1e-9)
    i = int(np.argmax(sim))
    smiles = pkg.get("active_smiles")
    return float(sim[i]), (str(smiles[i]) if smiles is not None and i < len(smiles) else None)


def target_profile(smiles: str) -> dict:
    """Predicted activity of one molecule against every modelled protein target."""
    reg = get_registry()
    m = mol.parse(smiles)
    desc = mol._descriptors(m)
    bits, _ = mol._fingerprint(m)
    profile = {}
    for cid, pkg in reg.targets.items():
        X = mol._matrix(pkg, desc, bits)
        p = float(np.clip(pkg["model"].predict_proba(X)[0, 1], 0, 1))
        sim, nearest = _nearest_active(pkg, bits)
        profile[cid] = {
            "chembl_id": cid, "symbol": pkg["symbol"], "name": pkg.get("name"),
            "probability": p, "threshold": float(pkg["threshold"]),
            "active": int(p >= pkg["threshold"]),
            "similarity": None if sim is None else round(sim, 3),
            "nearest_active": nearest,
            "decoy_fraction": pkg.get("decoy_fraction"),
            "test_roc_auc": (pkg.get("test_metrics") or {}).get("ROC-AUC"),
        }
    return profile


def diseases() -> list[dict]:
    reg = get_registry()
    out = []
    for d in reg.diseases:
        modelled = [t for t in d["targets"] if t["chembl_id"] in reg.targets]
        out.append({"id": d["id"], "label": d["label"], "name": d["name"],
                    "description": d["description"],
                    "n_targets": len(d["targets"]), "n_modelled": len(modelled),
                    "targets": [t["symbol"] for t in d["targets"]],
                    "n_known_drugs": len(d["known_drugs"])})
    return out


def disease_detail(disease_id: str) -> dict:
    reg = get_registry()
    d = reg.disease(disease_id)
    if d is None:
        raise ValueError(f"unknown illness '{disease_id}'")
    targets = []
    for t in d["targets"]:
        pkg = reg.targets.get(t["chembl_id"])
        targets.append({**t, "modelled": pkg is not None,
                        "test_roc_auc": (pkg or {}).get("test_metrics", {}).get("ROC-AUC"),
                        "decoy_fraction": (pkg or {}).get("decoy_fraction")})
    return {"id": d["id"], "label": d["label"], "name": d["name"],
            "description": d["description"], "targets": targets,
            "known_drugs": d["known_drugs"], "disclaimer": DISCLAIMER}


def _toxicity_flags(smiles: str) -> int | None:
    try:
        return mol.predict(smiles)["risk"]["n_flagged"]
    except Exception:  # noqa: BLE001 - toxicity models are optional here
        return None


def screen(smiles: str, disease_id: str | None = None, with_toxicity: bool = True) -> dict:
    """One molecule against one illness, or against all of them ranked best first."""
    reg = get_registry()
    if not reg.targets:
        raise FileNotFoundError("no target models - run pipeline/molecules/train_targets.py")
    profile = target_profile(smiles)
    flags = _toxicity_flags(smiles) if with_toxicity else None

    if disease_id:
        d = reg.disease(disease_id)
        if d is None:
            raise ValueError(f"unknown illness '{disease_id}'")
        results = [combine_with_toxicity(score_disease(d, profile), flags)]
    else:
        results = [combine_with_toxicity(r, flags) for r in rank_diseases(reg.diseases, profile)]

    hits = sorted((p for p in profile.values() if p["active"]),
                  key=lambda p: -p["probability"])
    return {
        "smiles": mol.properties(smiles)["smiles"],
        "results": results,
        "target_hits": hits,
        "n_targets_modelled": len(profile),
        "toxicity_flags": flags,
        "disclaimer": DISCLAIMER,
    }


def matrix(smiles_list: list[str]) -> dict:
    """Molecules x illnesses score grid - the heat map behind the screening page."""
    reg = get_registry()
    if not reg.targets:
        raise FileNotFoundError("no target models - run pipeline/molecules/train_targets.py")
    labels = [{"id": d["id"], "label": d["label"]} for d in reg.diseases]
    rows, errors = [], []
    for s in smiles_list:
        try:
            profile = target_profile(s)
        except ValueError as exc:
            errors.append({"smiles": s, "error": str(exc)})
            continue
        by_id = {r["disease_id"]: r for r in rank_diseases(reg.diseases, profile)}
        rows.append({
            "smiles": s,
            "scores": [{"disease_id": c["id"], "score": by_id[c["id"]]["score"],
                        "verdict": by_id[c["id"]]["verdict"],
                        "confidence": by_id[c["id"]].get("confidence")} for c in labels],
            "best": max((r for r in by_id.values() if r["score"] is not None),
                        key=lambda r: r["score"], default={}).get("label"),
        })
    return {"diseases": labels, "molecules": rows, "errors": errors, "disclaimer": DISCLAIMER}
