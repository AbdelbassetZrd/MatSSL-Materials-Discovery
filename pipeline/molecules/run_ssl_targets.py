"""The semi-supervised comparison, repeated on protein-target activity.

This is the third domain the same protocol is applied to:

    materials      1,000 labels + unlabeled GNoME compositions
    toxicity       1,000 labels + unlabeled ZINC molecules
    target activity   few hundred labels + unlabeled ZINC molecules   <- here

Targets with little measured data are exactly the case semi-supervised learning is supposed
to help with, so the comparison is run on the targets with the fewest labelled molecules.
Decoys are excluded from the labelled pool: they are assumed inactive rather than measured,
and mixing them in would make the task artificially easy.

Usage:  uv run python pipeline/molecules/run_ssl_targets.py [--n-targets 4] [--budget 400]
Output: data/predictions/targets/ssl_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "pipeline" / "molecules"))
import ssl_core as core  # noqa: E402
from featurize_mol import DESC_NAMES, feature_matrix, featurize_smiles, view_columns  # noqa: E402
from run_ssl_experiments import Preprocessor  # noqa: E402

warnings.filterwarnings("ignore")
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "predictions" / "targets"
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def stratified_take(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n_pos = max(2, int(round(n * len(pos) / len(y))))
    n_pos, n_neg = min(n_pos, len(pos)), min(n - n_pos, len(neg))
    return np.concatenate([rng.choice(pos, n_pos, replace=False),
                           rng.choice(neg, n_neg, replace=False)])


def run_target(cid: str, symbol: str, pool_raw: np.ndarray, args) -> dict | None:
    df = pd.read_parquet(PROC / "targets" / f"{cid}.parquet")
    df = df[~df["decoy"]].reset_index(drop=True)  # measured molecules only
    if len(df) < 300 or df["label"].nunique() < 2:
        log(f"{symbol:<9} only {len(df)} measured molecules - skipped")
        return None
    feat = featurize_smiles(df["smiles"].tolist(), n_jobs=args.n_jobs, quiet=True)
    feat["label"] = df["label"].to_numpy()
    feat = feat[feat["scaffold"] != ""].reset_index(drop=True)
    X, y = feature_matrix(feat), feat["label"].to_numpy(int)
    groups = feat["scaffold"].to_numpy()
    tr, te = next(StratifiedGroupKFold(4, shuffle=True, random_state=42).split(X, y, groups))
    if min(len(tr), len(te)) < 100 or len(set(y[te].tolist())) < 2:
        log(f"{symbol:<9} split too small - skipped")
        return None

    prep = Preprocessor().fit(pd.DataFrame(X[tr][:, :len(DESC_NAMES)], columns=DESC_NAMES))

    def tf(arr):
        d = pd.DataFrame(arr[:, :len(DESC_NAMES)], columns=DESC_NAMES)
        return np.hstack([prep.transform(d), arr[:, len(DESC_NAMES):]])

    Xtr, Xte, pool = tf(X[tr]), tf(X[te]), tf(pool_raw)
    v1, v2 = view_columns()  # descriptors vs fingerprint, the two Co-Training views
    v1 = [i for i in range(len(prep.cols))]
    v2 = [len(prep.cols) + i for i in range(len(v2))]

    budget = min(args.budget, int(0.8 * len(tr)))
    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        li = stratified_take(y[tr], budget, rng)
        d = core.Data(Xtr[li], y[tr][li],
                      pool[rng.permutation(len(pool))[:args.unlabeled]], Xte, y[te])
        runs = [
            core.run_baseline(d, "LightGBM", seed=seed)[0],
            core.run_label_propagation(d, "knn", n_neighbors=10)[0],
            core.run_label_propagation(d, "knn", n_neighbors=10, spreading=True)[0],
            core.run_self_training(d, "LightGBM", threshold=0.9, max_iter=3, seed=seed)[0],
            core.run_co_training(d, v1, v2, "LightGBM", threshold=0.85, n_rounds=5, seed=seed)[0],
            core.run_cluster_then_label(d, "LightGBM", n_clusters=50, seed=seed)[0],
        ]
        for r in runs:
            rows.append({**{k: r[k] for k in ("Model", "F1-Score", "ROC-AUC", "Composite_Score")},
                         "seed": seed})
    agg = (pd.DataFrame(rows).groupby("Model")[["F1-Score", "ROC-AUC", "Composite_Score"]]
           .agg(["mean", "std"]).round(4))
    best = agg[("Composite_Score", "mean")].idxmax()
    base = agg.loc[[m for m in agg.index if m.startswith("Baseline")][0],
                   ("Composite_Score", "mean")]
    gain = 100 * (agg.loc[best, ("Composite_Score", "mean")] - base) / base
    log(f"{symbol:<9} {len(feat):>5,} measured, {budget} labels | best {best} "
        f"{agg.loc[best, ('Composite_Score', 'mean')]:.4f} vs baseline {base:.4f} ({gain:+.2f}%)")
    return {"chembl_id": cid, "symbol": symbol, "n_measured": int(len(feat)),
            "n_labels": int(budget), "n_unlabeled": int(args.unlabeled), "seeds": args.seeds,
            "best_method": best, "baseline_composite": float(base),
            "improvement_pct": float(gain),
            "per_method": [{"method": m,
                            "composite_mean": float(agg.loc[m, ("Composite_Score", "mean")]),
                            "composite_std": float(agg.loc[m, ("Composite_Score", "std")] or 0),
                            "f1_mean": float(agg.loc[m, ("F1-Score", "mean")]),
                            "auc_mean": float(agg.loc[m, ("ROC-AUC", "mean")])}
                           for m in agg.index]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-targets", type=int, default=4)
    ap.add_argument("--budget", type=int, default=400, help="labelled molecules kept")
    ap.add_argument("--unlabeled", type=int, default=5000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()

    kb = json.loads((PROC / "diseases.json").read_text(encoding="utf-8"))
    sizes = []
    for t in kb["targets"]:
        f = PROC / "targets" / f"{t['chembl_id']}.parquet"
        if f.exists():
            d = pd.read_parquet(f, columns=["label", "decoy"])
            n = int((~d["decoy"]).sum())
            if n >= 400:
                sizes.append((n, t))
    sizes.sort(key=lambda x: x[0])  # the least data first: where SSL should matter most
    chosen = [t for _, t in sizes[:args.n_targets]]
    log(f"{len(chosen)} low-data targets: "
        f"{', '.join(f'{t['symbol']}' for t in chosen)}")

    pool_df = pd.read_parquet(PROC / "molecules_unlabeled_feat.parquet").sample(
        args.unlabeled * 2, random_state=11)
    pool_raw = feature_matrix(pool_df)

    results = [r for t in chosen
               if (r := run_target(t["chembl_id"], t["symbol"], pool_raw, args))]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ssl_comparison.json").write_text(
        json.dumps({"setup": {"budget": args.budget, "unlabeled": args.unlabeled,
                              "seeds": args.seeds, "pool": "ZINC"},
                    "targets": results}, indent=1), encoding="utf-8")
    if results:
        g = [r["improvement_pct"] for r in results]
        log(f"DONE {len(results)} targets | best SSL vs baseline: "
            f"{min(g):+.2f}% to {max(g):+.2f}% (mean {np.mean(g):+.2f}%)")


if __name__ == "__main__":
    main()
