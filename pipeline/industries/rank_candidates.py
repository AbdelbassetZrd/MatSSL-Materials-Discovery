"""Rank the 554k unlabeled GNoME compositions for every Design Lab application.

Every deployed model is run once over the whole pool, the application scores are computed from
the resulting probabilities, and the best candidates of each application are stored so the web
application can answer "suggest me a superconductor / a cathode" instantly.

The pool is the *unlabeled* set of the semi-supervised study: these are compositions proposed by
a generative search, so none of them carries a measured label. That is exactly the discovery
setting -- the models are used to triage candidates nobody has computed yet.

Usage:  uv run python pipeline/industries/rank_candidates.py [--top 400]
Output: data/predictions/design/candidates.parquet, design_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "pipeline" / "industries"))
from applications import APPLICATIONS, score  # noqa: E402
from featurize import parse_formula  # noqa: E402
from run_ssl_experiments import Preprocessor  # noqa: E402
from self_supervised import MaskedAutoencoder  # noqa: E402

_main = sys.modules["__main__"]
for _cls in (Preprocessor, MaskedAutoencoder):
    if not hasattr(_main, _cls.__name__):
        setattr(_main, _cls.__name__, _cls)

PROC = ROOT / "data" / "processed"
MODELS = ROOT / "app" / "models"
OUT = ROOT / "data" / "predictions" / "design"
CHUNK = 50_000
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c.startswith(("MagpieData", "frac ")) or c.endswith("-norm")]


def property_predictions(raw: pd.DataFrame, reg: dict, ae) -> dict[str, np.ndarray]:
    """Probabilities / values of every deployed model for a block of compositions."""
    out = {}
    for task, entry in reg.items():
        pkg = joblib.load(MODELS / entry["model_file"])
        prep = joblib.load(MODELS / entry["scaler_file"])
        X = prep.transform(raw)
        if task == "band_gap":
            X = X[:, pkg["selected_idx"]]
        else:
            space = pkg.get("feature_space", "raw")
            if space != "raw":
                Z = ae.encode(raw)
                X = Z if space == "ae" else np.hstack([X, Z])
        out[task] = pkg["model"].predict_proba(X)[:, 1]
    sc = MODELS / "superconductor_model.pkl"
    if sc.exists():
        pkg = joblib.load(sc)
        out["tc"] = pkg["model"].predict(pkg["preprocessor"].transform(raw))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=400, help="candidates kept per application")
    ap.add_argument("--limit", type=int, default=0, help="only score the first N compositions")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    reg = json.loads((MODELS / "registry.json").read_text())
    ae = joblib.load(MODELS / "self_supervised_autoencoder.pkl")
    pool = pd.read_parquet(PROC / "unlabeled_gnome_features.parquet")
    if args.limit:
        pool = pool.head(args.limit)
    cols = feat_cols(pool)
    log(f"scoring {len(pool):,} GNoME compositions with {len(reg)} property models + Tc")

    blocks = []
    for start in range(0, len(pool), CHUNK):
        block = pool.iloc[start:start + CHUNK]
        preds = property_predictions(block[cols], reg, ae)
        blocks.append(pd.DataFrame({"formula": block["formula"].to_numpy(), **preds}))
        log(f"   {min(start + CHUNK, len(pool)):,}/{len(pool):,}")
    df = pd.concat(blocks, ignore_index=True)

    name = {"metallicity": "metal", "magnetism": "magnetic", "stability": "stable",
            "band_gap": "pv_gap"}
    df = df.rename(columns=name)
    elements = {f: set(parse_formula(f) or {}) for f in df["formula"].unique()}
    log(f"parsed {len(elements):,} formulas")

    rows, strong = [], {}
    for app in APPLICATIONS:
        sub = df.copy()
        needs_elements = app == "battery_cathode" or app == "permanent_magnet"
        vals = []
        for r in sub.itertuples(index=False):
            p = {k: getattr(r, k, None) for k in ("stable", "metal", "magnetic", "pv_gap", "tc")}
            if needs_elements:
                p["elements"] = elements.get(r.formula, set())
            vals.append(score(app, p)["score"])
        sub["score"] = vals
        strong[app] = int((sub["score"] >= 0.6).sum())  # over the whole pool, not the top-N
        best = sub.nlargest(args.top, "score").assign(application=app)
        rows.append(best)
        log(f"   {app:<18} best {best['score'].max():.3f} | "
            f"top-{args.top} above {best['score'].min():.3f} | "
            f"{strong[app]:,} strong candidates in the pool")

    out = pd.concat(rows, ignore_index=True).round(4)
    out.to_parquet(OUT / "candidates.parquet", index=False)
    summary = {
        "pool": {"source": "GNoME (unlabeled)", "n_scored": int(len(df))},
        "applications": {
            a: {"title": APPLICATIONS[a]["title"], "subtitle": APPLICATIONS[a]["subtitle"],
                "question": APPLICATIONS[a]["question"], "note": APPLICATIONS[a]["note"],
                "examples": APPLICATIONS[a]["examples"],
                "n_strong": strong[a],
                "top": out[out.application == a].head(50).to_dict(orient="records")}
            for a in APPLICATIONS},
    }
    (OUT / "design_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    log(f"DONE -> {OUT / 'candidates.parquet'}")


if __name__ == "__main__":
    main()
