"""Materials prediction service.

Loads the deployed models (one per task) selected by the experiment campaign
(``app/models/registry.json``), featurises a chemical formula exactly like the
training pipeline, and returns predictions with probabilities, the selected
semi-supervised method, an applicability-domain check and chemistry checks.
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
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "app" / "models"
PRED_DIR = ROOT / "data" / "predictions"
sys.path.insert(0, str(ROOT / "pipeline"))

from featurize import featurize_formulas, parse_formula  # noqa: E402
from run_ssl_experiments import Preprocessor  # noqa: E402
from self_supervised import MaskedAutoencoder  # noqa: E402

logger = logging.getLogger(__name__)

# Pickles written by scripts run as __main__ reference __main__.<Class>.
_main = sys.modules["__main__"]
for _cls in (Preprocessor, MaskedAutoencoder):
    if not hasattr(_main, _cls.__name__):
        setattr(_main, _cls.__name__, _cls)

TASKS = {
    "metallicity": {"title": "Metallicity", "positive": "Metal", "negative": "Non-metal",
                    "source": "Materials Project"},
    "magnetism": {"title": "Magnetism", "positive": "Magnetic", "negative": "Non-magnetic",
                  "source": "JARVIS-DFT"},
    "stability": {"title": "Thermodynamic stability", "positive": "Stable / near hull",
                  "negative": "Unstable", "source": "Alexandria"},
    "band_gap": {"title": "Photovoltaic band gap", "positive": "In PV window (1.1-1.7 eV)",
                 "negative": "Outside PV window", "source": "Materials Project"},
}


class ModelInfo(BaseModel):
    task: str
    title: str
    status: str
    method: str | None = None
    config: str | None = None
    source: str | None = None
    test_metrics: dict | None = None


class TaskPrediction(BaseModel):
    task: str
    title: str
    prediction: int
    prediction_label: str
    probability: float
    confidence: float
    method: str | None = None


class _Registry:
    def __init__(self) -> None:
        self.models: dict[str, dict] = {}
        self.scalers: dict[str, object] = {}
        self.ae = None
        self.rules: dict[str, pd.DataFrame] = {}
        self.load()

    def load(self) -> None:
        reg_path = MODELS_DIR / "registry.json"
        registry = json.loads(reg_path.read_text()) if reg_path.exists() else {}
        for task in TASKS:
            entry = registry.get(task, {})
            mfile = MODELS_DIR / entry.get("model_file", f"{task}_model.pkl")
            sfile = MODELS_DIR / entry.get("scaler_file", f"{task}_scaler.pkl")
            if mfile.exists() and sfile.exists():
                try:
                    pkg = joblib.load(mfile)
                    pkg["registry"] = entry
                    self.models[task] = pkg
                    self.scalers[task] = joblib.load(sfile)
                    logger.info("loaded %s model (%s)", task, pkg.get("config"))
                except Exception as exc:  # pragma: no cover - logged for ops
                    logger.error("could not load %s model: %s", task, exc)
        ae_path = MODELS_DIR / "self_supervised_autoencoder.pkl"
        if ae_path.exists():
            self.ae = joblib.load(ae_path)
        for task in TASKS:
            f = PRED_DIR / "unsupervised" / f"association_rules_{task}.csv"
            if f.exists():
                self.rules[task] = pd.read_csv(f)


@lru_cache(maxsize=1)
def get_registry() -> _Registry:
    return _Registry()


def models_info() -> list[ModelInfo]:
    reg = get_registry()
    out = []
    for task, meta in TASKS.items():
        pkg = reg.models.get(task)
        out.append(ModelInfo(
            task=task, title=meta["title"], source=meta["source"],
            status="Loaded" if pkg else "Not loaded",
            method=(pkg or {}).get("method") or (pkg or {}).get("registry", {}).get("method"),
            config=(pkg or {}).get("config"),
            test_metrics=(pkg or {}).get("test_metrics") or
            (pkg or {}).get("registry", {}).get("test_metrics")))
    return out


def _features(formula: str) -> pd.DataFrame:
    feats, ok = featurize_formulas(pd.Series([formula]))
    if not bool(ok.iloc[0]):
        raise ValueError(f"Could not parse formula '{formula}'")
    return feats


def _task_matrix(task: str, raw: pd.DataFrame) -> np.ndarray:
    reg = get_registry()
    pkg, prep = reg.models[task], reg.scalers[task]
    X = prep.transform(raw)
    if task == "band_gap":
        return X[:, pkg["selected_idx"]]
    space = pkg.get("feature_space", "raw")
    if space == "raw":
        return X
    Z = reg.ae.encode(raw)
    return Z if space == "ae" else np.hstack([X, Z])


def predict(formula: str) -> dict:
    """Predict every available property for one formula."""
    reg = get_registry()
    raw = _features(formula)
    preds = []
    for task, meta in TASKS.items():
        if task not in reg.models:
            continue
        pkg = reg.models[task]
        X = _task_matrix(task, raw)
        p = float(np.clip(pkg["model"].predict_proba(X)[0, 1], 0, 1))
        y = int(p >= 0.5)
        preds.append(TaskPrediction(
            task=task, title=meta["title"], prediction=y,
            prediction_label=meta["positive"] if y else meta["negative"], probability=p,
            confidence=abs(p - 0.5) * 2,
            method=pkg.get("method") or ("Genetic-algorithm descriptors + "
                                         "gradient boosting" if task == "band_gap" else None)
        ).model_dump())
    return {"formula": formula, "predictions": preds,
            "applicability_domain": applicability_domain(raw)}


def applicability_domain(raw: pd.DataFrame) -> dict | None:
    reg = get_registry()
    pkg = reg.models.get("band_gap")
    if not pkg or "nn" not in pkg:
        return None
    X = _task_matrix("band_gap", raw)
    d = float(pkg["nn"].kneighbors(X, n_neighbors=5)[0].mean())
    pc = pkg["pca"].transform(X)[0]
    return {"distance": d, "threshold": float(pkg["ad_threshold"]),
            "inside": bool(d <= pkg["ad_threshold"]),
            "cluster": int(pkg["kmeans"].predict(pkg["pca"].transform(X))[0]),
            "pca": [float(pc[0]), float(pc[1])]}


def analyze(formula: str) -> dict:
    """Composition analysis: fractions, key descriptors, chemical rules."""
    from pymatgen.core import Composition

    frac = parse_formula(formula)
    if frac is None:
        raise ValueError(f"Could not parse formula '{formula}'")
    comp = Composition(formula)
    raw = _features(formula)
    key = ["MagpieData mean Electronegativity", "MagpieData range Electronegativity",
           "MagpieData mean CovalentRadius", "MagpieData mean NValence",
           "MagpieData mean GSbandgap", "MagpieData mean GSmagmom",
           "MagpieData mean MeltingT", "MagpieData mean AtomicWeight", "0-norm"]
    try:
        guesses = comp.oxi_state_guesses(max_sites=-50)
    except Exception:
        guesses = []
    elements = sorted(frac)
    matched = []
    for task, rules in get_registry().rules.items():
        for _, r in rules.iterrows():
            ante = [a.strip() for a in str(r["antecedents"]).split(",")]
            if all(a in elements for a in ante):
                matched.append({"task": task, "rule": f"{{{r['antecedents']}}} -> "
                                f"{r['consequents']}", "confidence": float(r["confidence"]),
                                "lift": float(r["lift"])})
    matched = sorted(matched, key=lambda m: -m["lift"])[:8]
    return {
        "formula": formula,
        "reduced_formula": comp.reduced_formula,
        "elements": [{"symbol": el, "fraction": float(f)} for el, f in frac.items()],
        "n_elements": len(elements),
        "n_atoms_formula_unit": float(comp.num_atoms),
        "weight": float(comp.weight),
        "descriptors": {k: (None if pd.isna(raw[k].iloc[0]) else float(raw[k].iloc[0]))
                        for k in key if k in raw},
        "chemical_rules": {
            "charge_balanced": bool(guesses),
            "oxidation_states": {k: float(v) for k, v in guesses[0].items()} if guesses
            else None,
            "n_elements_ok": len(elements) <= 6,
        },
        "association_rules_matched": matched,
    }
