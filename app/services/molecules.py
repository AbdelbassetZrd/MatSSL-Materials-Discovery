"""Molecular toxicity service.

Loads the per-endpoint models trained by pipeline/molecules/train_toxicity.py, featurises a
SMILES exactly as the training pipeline does, and returns one prediction per endpoint with a
calibrated threshold, an explanation and standard drug-likeness properties.

Explanations are exact SHAP contributions from the LightGBM member of each stacked ensemble.
When a fingerprint bit is among the top contributors, the substructure it stands for is
decoded back to a SMILES fragment, so the explanation stays chemically readable.
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "app" / "models"
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "pipeline" / "molecules"))

from featurize_mol import DESC_NAMES, N_BITS, RADIUS, scaffold  # noqa: E402
from hiacc_model import StackedEnsemble  # noqa: E402
from run_ssl_experiments import Preprocessor  # noqa: E402

logger = logging.getLogger(__name__)

# pickles written by scripts run as __main__ reference __main__.<Class>
_main = sys.modules["__main__"]
for _cls in (Preprocessor, StackedEnsemble):
    if not hasattr(_main, _cls.__name__):
        setattr(_main, _cls.__name__, _cls)

TOX21 = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]
ENDPOINTS = [*TOX21, "CT_TOX", "FDA_APPROVED"]
# endpoints that count toward the overall toxicity flag (FDA_APPROVED is not a toxicity signal)
RISK_ENDPOINTS = [*TOX21, "CT_TOX"]
GROUPS = {
    **{e: "Nuclear receptor" for e in TOX21 if e.startswith("NR-")},
    **{e: "Stress response" for e in TOX21 if e.startswith("SR-")},
    "CT_TOX": "Clinical",
    "FDA_APPROVED": "Clinical",
}
PRETTY_DESC = {
    "MolWt": "molecular weight",
    "MolLogP": "lipophilicity (logP)",
    "TPSA": "polar surface area",
    "qed": "drug-likeness (QED)",
    "NumHDonors": "H-bond donors",
    "NumHAcceptors": "H-bond acceptors",
    "NumRotatableBonds": "rotatable bonds",
    "RingCount": "rings",
    "NumAromaticRings": "aromatic rings",
    "FractionCSP3": "fraction of sp3 carbons",
    "HeavyAtomCount": "heavy atoms",
    "MolMR": "molar refractivity",
    "BalabanJ": "Balaban index",
    "BertzCT": "molecular complexity",
    "MaxPartialCharge": "most positive partial charge",
    "MinPartialCharge": "most negative partial charge",
}


class EndpointPrediction(BaseModel):
    endpoint: str
    title: str
    group: str
    prediction: int
    prediction_label: str
    probability: float
    threshold: float
    test_metrics: dict | None = None
    explanation: dict | None = None


def safe(name: str) -> str:
    return name.replace("-", "_").lower()


def pretty(name: str) -> str:
    if name in PRETTY_DESC:
        return PRETTY_DESC[name]
    if name.startswith("fr_"):
        return f"{name[3:].replace('_', ' ')} group count"
    return name


class _MolRegistry:
    def __init__(self) -> None:
        self.models: dict[str, dict] = {}
        for e in ENDPOINTS:
            f = MODELS_DIR / f"mol_{safe(e)}_model.pkl"
            if f.exists():
                try:
                    self.models[e] = joblib.load(f)
                except Exception as exc:  # pragma: no cover - logged for ops
                    logger.error("could not load molecule model %s: %s", e, exc)
        if self.models:
            logger.info("loaded %d molecular endpoints", len(self.models))


@lru_cache(maxsize=1)
def get_registry() -> _MolRegistry:
    return _MolRegistry()


def parse(smiles: str):
    """RDKit molecule, or ValueError for an unparseable SMILES."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES '{smiles}'")
    return mol


def _fingerprint(mol):
    """(bit vector, {bit: readable fragment}) for the top-level Morgan fingerprint."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator as fpgen  # noqa: N813

    gen = fpgen.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    ao = fpgen.AdditionalOutput()
    ao.AllocateBitInfoMap()
    bits = np.asarray(gen.GetFingerprint(mol, additionalOutput=ao), dtype=np.float32)
    frags = {}
    for bit, envs in (ao.GetBitInfoMap() or {}).items():
        atom, radius = envs[0]
        try:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom) if radius else None
            if env:
                amap: dict[int, int] = {}
                sub = Chem.PathToSubmol(mol, env, atomMap=amap)
                frags[int(bit)] = Chem.MolToSmiles(sub)
            else:
                frags[int(bit)] = mol.GetAtomWithIdx(atom).GetSymbol()
        except Exception as exc:  # noqa: BLE001 - a bit we cannot decode is simply unnamed
            logger.debug("could not decode fingerprint bit %s: %s", bit, exc)
    return bits, frags


def _descriptors(mol) -> dict:
    from rdkit.Chem import Descriptors

    d = Descriptors.CalcMolDescriptors(mol)
    return {k: (float(v) if np.isfinite(v) else np.nan) for k, v in d.items()}


def properties(smiles: str) -> dict:
    """Drug-likeness summary shown next to the predictions."""
    from rdkit import Chem

    mol = parse(smiles)
    d = _descriptors(mol)
    rules = {
        "MW <= 500": d["MolWt"] <= 500,
        "logP <= 5": d["MolLogP"] <= 5,
        "H-bond donors <= 5": d["NumHDonors"] <= 5,
        "H-bond acceptors <= 10": d["NumHAcceptors"] <= 10,
    }
    return {
        "smiles": Chem.MolToSmiles(mol),
        "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
        "scaffold": scaffold(Chem.MolToSmiles(mol)),
        "n_atoms": mol.GetNumHeavyAtoms(),
        "descriptors": {
            "molecular_weight": d["MolWt"],
            "logP": d["MolLogP"],
            "TPSA": d["TPSA"],
            "QED": d["qed"],
            "h_bond_donors": d["NumHDonors"],
            "h_bond_acceptors": d["NumHAcceptors"],
            "rotatable_bonds": d["NumRotatableBonds"],
            "aromatic_rings": d["NumAromaticRings"],
            "rings": d["RingCount"],
            "fraction_sp3": d["FractionCSP3"],
        },
        "lipinski": {
            "rules": rules,
            "violations": int(sum(not v for v in rules.values())),
            "passes": bool(sum(not v for v in rules.values()) <= 1),
        },
    }


def _matrix(pkg: dict, desc: dict, bits: np.ndarray) -> np.ndarray:
    row = pd.DataFrame([[desc.get(n, np.nan) for n in DESC_NAMES]], columns=DESC_NAMES)
    return np.hstack([pkg["preprocessor"].transform(row), bits.reshape(1, -1)])


def _explain(pkg: dict, X: np.ndarray, desc: dict, frags: dict, top: int = 6) -> dict | None:
    est = pkg.get("lgbm")
    if est is None:
        return None
    try:
        contrib = np.asarray(est.predict(X, pred_contrib=True))[0, :-1]
    except Exception as exc:  # noqa: BLE001 - explanation is best effort
        logger.warning("molecule contribution failed: %s", exc)
        return None
    names = pkg["feature_columns"]
    if len(contrib) != len(names):
        return None
    items = []
    n_desc = len(pkg["descriptor_columns"])  # constant descriptors were dropped at fit time
    for i in np.argsort(-np.abs(contrib))[:top]:
        if i < n_desc:
            items.append(
                {
                    "feature": pretty(names[i]),
                    "value": desc.get(names[i]),
                    "contribution": float(contrib[i]),
                    "kind": "descriptor",
                }
            )
        else:
            bit = int(i - n_desc)
            frag = frags.get(bit)
            items.append(
                {
                    "feature": f"substructure {frag}" if frag else f"substructure #{bit}",
                    "value": float(X[0, i]),
                    "contribution": float(contrib[i]),
                    "kind": "substructure",
                }
            )
    return {"type": "shap", "items": items}


def predict(smiles: str) -> dict:
    """Toxicity profile of one molecule: one prediction per available endpoint."""
    reg = get_registry()
    mol = parse(smiles)
    desc = _descriptors(mol)
    bits, frags = _fingerprint(mol)
    preds, flagged = [], []
    for e in ENDPOINTS:
        pkg = reg.models.get(e)
        if pkg is None:
            continue
        X = _matrix(pkg, desc, bits)
        p = float(np.clip(pkg["model"].predict_proba(X)[0, 1], 0, 1))
        t = float(pkg["threshold"])
        y = int(p >= t)
        if y and e in RISK_ENDPOINTS:
            flagged.append(e)
        positive = "Approved" if e == "FDA_APPROVED" else "Active / toxic"
        negative = "Not approved" if e == "FDA_APPROVED" else "Inactive"
        preds.append(
            EndpointPrediction(
                endpoint=e,
                title=pkg.get("title", e),
                group=GROUPS.get(e, "Other"),
                prediction=y,
                prediction_label=positive if y else negative,
                probability=p,
                threshold=t,
                test_metrics=pkg.get("test_metrics"),
                explanation=_explain(pkg, X, desc, frags),
            ).model_dump()
        )
    n = len(flagged)
    level = "Low" if n == 0 else "Moderate" if n <= 2 else "High"
    return {
        **properties(smiles),
        "predictions": preds,
        "risk": {
            "level": level,
            "n_flagged": n,
            "n_endpoints": len(RISK_ENDPOINTS),
            "flagged": flagged,
            "summary": f"{n} of {len(RISK_ENDPOINTS)} toxicity endpoints predicted active",
        },
    }


def models_info() -> list[dict]:
    reg = get_registry()
    out = []
    for e in ENDPOINTS:
        pkg = reg.models.get(e)
        out.append(
            {
                "endpoint": e,
                "title": (pkg or {}).get("title", e),
                "group": GROUPS.get(e, "Other"),
                "status": "Loaded" if pkg else "Not loaded",
                "method": (pkg or {}).get("method"),
                "config": (pkg or {}).get("config"),
                "test_metrics": (pkg or {}).get("test_metrics"),
            }
        )
    return out
