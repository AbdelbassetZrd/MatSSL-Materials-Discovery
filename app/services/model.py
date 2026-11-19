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
    "metallicity": {
        "title": "Metallicity",
        "positive": "Metal",
        "negative": "Non-metal",
        "source": "Materials Project",
    },
    "magnetism": {
        "title": "Magnetism",
        "positive": "Magnetic",
        "negative": "Non-magnetic",
        "source": "JARVIS-DFT",
    },
    "stability": {
        "title": "Thermodynamic stability",
        "positive": "Stable / near hull",
        "negative": "Unstable",
        "source": "Alexandria",
    },
    "band_gap": {
        "title": "Photovoltaic band gap (DFT-PBE)",
        "positive": "PBE gap in 1.1-1.7 eV window",
        "negative": "PBE gap outside 1.1-1.7 eV",
        "source": "Materials Project",
    },
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
    threshold: float = 0.5
    method: str | None = None
    config: str | None = None
    test_metrics: dict | None = None
    explanation: dict | None = None


PRETTY = {
    "Electronegativity": "electronegativity",
    "CovalentRadius": "covalent radius",
    "NValence": "valence electrons",
    "NsValence": "s valence electrons",
    "NpValence": "p valence electrons",
    "NdValence": "d valence electrons",
    "NfValence": "f valence electrons",
    "NUnfilled": "unfilled orbitals",
    "NsUnfilled": "unfilled s orbitals",
    "NpUnfilled": "unfilled p orbitals",
    "NdUnfilled": "unfilled d orbitals",
    "NfUnfilled": "unfilled f orbitals",
    "GSmagmom": "elemental magnetic moment",
    "GSbandgap": "elemental band gap",
    "GSvolume_pa": "elemental volume per atom",
    "MeltingT": "melting temperature",
    "MendeleevNumber": "Mendeleev number",
    "AtomicWeight": "atomic weight",
    "SpaceGroupNumber": "elemental space group",
    "Number": "atomic number",
    "Column": "periodic-table column",
    "Row": "periodic-table row",
}


def pretty(name: str) -> str:
    """Human-readable descriptor name, e.g. 'MagpieData mean NdValence' -> 'mean d valence…'."""
    if name.startswith("MagpieData "):
        _, stat, prop = name.split(" ", 2)
        return f"{stat.replace('_', ' ')} {PRETTY.get(prop, prop)}"
    if name == "0-norm":
        return "number of elements"
    if name.endswith("-norm"):
        return f"stoichiometry {name}"
    if name.startswith("AE "):
        return f"self-supervised embedding {name[3:]}"
    return name


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
        out.append(
            ModelInfo(
                task=task,
                title=meta["title"],
                source=meta["source"],
                status="Loaded" if pkg else "Not loaded",
                method=(pkg or {}).get("method") or (pkg or {}).get("registry", {}).get("method"),
                config=(pkg or {}).get("config"),
                test_metrics=(pkg or {}).get("test_metrics")
                or (pkg or {}).get("registry", {}).get("test_metrics"),
            )
        )
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


def _feature_names(task: str) -> list[str]:
    reg = get_registry()
    pkg, prep = reg.models[task], reg.scalers[task]
    if task == "band_gap":
        return list(pkg["selected_features"])
    ae_names = [f"AE {i + 1}" for i in range(32)]
    space = pkg.get("feature_space", "raw")
    return {"raw": list(prep.cols), "ae": ae_names, "raw+ae": list(prep.cols) + ae_names}[space]


def _tree_contrib(est, X: np.ndarray) -> np.ndarray | None:
    """Exact SHAP contributions (log-odds) of a LightGBM / XGBoost model, else None."""
    name = type(est).__name__
    try:
        if name == "LGBMClassifier":
            return np.asarray(est.predict(X, pred_contrib=True))[0, :-1]
        if name == "XGBClassifier":
            import xgboost

            return est.get_booster().predict(xgboost.DMatrix(X), pred_contribs=True)[0, :-1]
    except Exception as exc:  # pragma: no cover - explanation is best effort
        logger.warning("contribution failed: %s", exc)
    return None


def explain(task: str, raw: pd.DataFrame, X: np.ndarray, top: int = 6) -> dict | None:
    """Why this prediction: SHAP contributions for tree-boosting based models, otherwise the
    position of the composition's key descriptors in the positive/negative training classes."""
    reg = get_registry()
    model = reg.models[task]["model"]
    names = _feature_names(task)
    contrib = None
    kind = type(model).__name__
    if kind == "SelfTrainingClassifier":
        contrib = _tree_contrib(model.estimator_, X)
    elif kind == "ClusterThenLabel":
        contrib = _tree_contrib(model.model_, X)
    elif kind == "CoTrainingClassifier":
        c1, c2 = (
            _tree_contrib(model.c1_, X[:, model.view1]),
            _tree_contrib(model.c2_, X[:, model.view2]),
        )
        if c1 is not None and c2 is not None:
            contrib = np.zeros(X.shape[1])
            contrib[model.view1] += 0.5 * c1
            contrib[model.view2] += 0.5 * c2
    elif kind in ("LGBMClassifier", "XGBClassifier"):
        contrib = _tree_contrib(model, X)
    if contrib is not None and len(contrib) == len(names):
        order = np.argsort(-np.abs(contrib))[:top]
        items = []
        for i in order:
            n = names[i]
            val = float(raw[n].iloc[0]) if n in raw else None
            items.append({"feature": pretty(n), "value": val, "contribution": float(contrib[i])})
        return {"type": "shap", "items": items}
    idx = neighbor_index()
    if idx is None or task not in idx["percentiles"]:
        return None
    items = []
    for n, q in idx["percentiles"][task].items():
        v = float(raw[n].iloc[0])
        p_neg = float(np.searchsorted(q[0], v) / len(q[0]) * 100)
        p_pos = float(np.searchsorted(q[1], v) / len(q[1]) * 100)
        items.append(
            {"feature": pretty(n), "value": v, "pct_positive": p_pos, "pct_negative": p_neg}
        )
    items.sort(key=lambda d: -abs(d["pct_positive"] - d["pct_negative"]))
    return {"type": "percentile", "items": items[:top]}


@lru_cache(maxsize=1)
def neighbor_index() -> dict | None:
    f = MODELS_DIR / "neighbor_index.joblib"
    if not f.exists():
        return None
    from sklearn.neighbors import NearestNeighbors

    idx = joblib.load(f)
    idx["nn"] = NearestNeighbors(n_neighbors=10).fit(idx["embedding"].astype(np.float32))
    return idx


def similar(formula: str, k: int = 8) -> dict:
    """Nearest labeled compositions in the self-supervised embedding space."""
    reg, idx = get_registry(), neighbor_index()
    if idx is None or reg.ae is None:
        raise RuntimeError("neighbor index not available")
    raw = _features(formula)
    dist, ind = idx["nn"].kneighbors(reg.ae.encode(raw), n_neighbors=k)
    out = []
    for d, i in zip(dist[0], ind[0], strict=True):
        row = {"formula": str(idx["formula"][i]), "distance": float(d)}
        for key, values in idx["labels"].items():
            v = values[i]
            row[key] = None if v is None or (isinstance(v, float) and np.isnan(v)) else v
        out.append(row)
    from pymatgen.core import Composition

    reduced = Composition(formula).reduced_formula
    return {
        "formula": reduced,
        "known": any(r["formula"] == reduced for r in out),
        "neighbors": out,
    }


def experiments() -> dict:
    """Dashboard data exported by pipeline/export_dashboard.py."""
    for f in (
        ROOT / "frontend" / "public" / "data" / "dashboard.json",
        ROOT / "frontend" / "dist" / "data" / "dashboard.json",
    ):
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
    raise FileNotFoundError("dashboard.json not found - run pipeline/export_dashboard.py")


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
        t = float(pkg.get("threshold", 0.5))  # calibrated for rare classes
        y = int(p >= t)
        entry = pkg.get("registry", {})
        preds.append(
            TaskPrediction(
                task=task,
                title=meta["title"],
                prediction=y,
                prediction_label=meta["positive"] if y else meta["negative"],
                probability=p,
                confidence=(p - t) / (1 - t) if y else (t - p) / t,
                threshold=t,
                method=pkg.get("method") or entry.get("method"),
                config=entry.get("config") or pkg.get("config"),
                test_metrics=entry.get("test_metrics") or pkg.get("test_metrics"),
                explanation=explain(task, raw, X),
            ).model_dump()
        )
    return {
        "formula": formula,
        "predictions": preds,
        "applicability_domain": applicability_domain(raw),
    }


def applicability_domain(raw: pd.DataFrame) -> dict | None:
    reg = get_registry()
    pkg = reg.models.get("band_gap")
    if not pkg or "nn" not in pkg:
        return None
    X = _task_matrix("band_gap", raw)
    d = float(pkg["nn"].kneighbors(X, n_neighbors=5)[0].mean())
    pc = pkg["pca"].transform(X)[0]
    return {
        "distance": d,
        "threshold": float(pkg["ad_threshold"]),
        "inside": bool(d <= pkg["ad_threshold"]),
        "cluster": int(pkg["kmeans"].predict(pkg["pca"].transform(X))[0]),
        "pca": [float(pc[0]), float(pc[1])],
    }


def analyze(formula: str) -> dict:
    """Composition analysis: fractions, key descriptors, chemical rules."""
    from pymatgen.core import Composition

    frac = parse_formula(formula)
    if frac is None:
        raise ValueError(f"Could not parse formula '{formula}'")
    comp = Composition(formula)
    raw = _features(formula)
    key = [
        "MagpieData mean Electronegativity",
        "MagpieData range Electronegativity",
        "MagpieData mean CovalentRadius",
        "MagpieData mean NValence",
        "MagpieData mean GSbandgap",
        "MagpieData mean GSmagmom",
        "MagpieData mean MeltingT",
        "MagpieData mean AtomicWeight",
        "0-norm",
    ]
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
                matched.append(
                    {
                        "task": task,
                        "rule": f"{{{r['antecedents']}}} -> {r['consequents']}",
                        "confidence": float(r["confidence"]),
                        "lift": float(r["lift"]),
                    }
                )
    matched = sorted(matched, key=lambda m: -m["lift"])[:8]
    return {
        "formula": formula,
        "reduced_formula": comp.reduced_formula,
        "elements": [{"symbol": el, "fraction": float(f)} for el, f in frac.items()],
        "n_elements": len(elements),
        "n_atoms_formula_unit": float(comp.num_atoms),
        "weight": float(comp.weight),
        "descriptors": {
            k: (None if pd.isna(raw[k].iloc[0]) else float(raw[k].iloc[0])) for k in key if k in raw
        },
        "chemical_rules": {
            "charge_balanced": bool(guesses),
            "oxidation_states": {k: float(v) for k, v in guesses[0].items()} if guesses else None,
            "n_elements_ok": len(elements) <= 6,
        },
        "association_rules_matched": matched,
    }
