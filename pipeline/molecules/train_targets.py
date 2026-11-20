"""Train one activity classifier per protein target.

Same recipe as the toxicity models (pipeline/molecules/train_toxicity.py): RDKit descriptors +
Morgan fingerprint, a train/test split grouped by Bemis-Murcko scaffold so no scaffold is on
both sides, a stacked LightGBM / XGBoost / HistGB ensemble and a threshold calibrated on
out-of-fold training predictions.

Two things are measured beyond the usual metrics, because they are what actually matters when
the model is used for screening:

  * **enrichment** - do known actives rank above random ZINC molecules? Reported as the
    enrichment factor in the top 1 % of a ranked list, where 1.0 means no better than chance.
  * **reliability vs similarity** - held-out molecules are binned by their Tanimoto similarity
    to the nearest training active, and accuracy is measured per bin. The applicability domain
    is then read off that curve instead of being fixed at an arbitrary cutoff.

Per-target Optuna is deliberately skipped: on both earlier tracks tuning did not improve the
held-out score, and it would cost hours across ~50 targets.

Usage:  uv run python pipeline/molecules/train_targets.py [--targets CHEMBL203 ...]
Output: app/models/target_{chembl_id}.pkl, data/predictions/targets/{chembl_id}_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "pipeline" / "molecules"))
from featurize_mol import DESC_NAMES, feature_matrix, feature_names, featurize_smiles  # noqa: E402
from hiacc_model import StackedEnsemble, _logit  # noqa: E402
from run_ssl_experiments import Preprocessor  # noqa: E402
from ssl_core import evaluate  # noqa: E402

warnings.filterwarnings("ignore")
PROC = ROOT / "data" / "processed"
TARGETS = PROC / "targets"
OUT = ROOT / "data" / "predictions" / "targets"
MODELS = ROOT / "app" / "models"
SIM_BINS = [0.0, 0.3, 0.4, 0.5, 0.7, 1.01]
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def bases(nj: int):
    return [
        ("LightGBM", lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                                        colsample_bytree=0.3, subsample=0.8, subsample_freq=1,
                                        min_child_samples=10, n_jobs=nj, verbose=-1,
                                        random_state=42)),
        ("XGBoost", XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                                  colsample_bytree=0.3, subsample=0.8, tree_method="hist",
                                  n_jobs=nj, random_state=42, eval_metric="logloss")),
        ("HistGB", HistGradientBoostingClassifier(max_iter=350, learning_rate=0.08,
                                                  max_leaf_nodes=31, random_state=42)),
    ]


def tanimoto_to_actives(fp: np.ndarray, actives: np.ndarray) -> np.ndarray:
    """Highest Tanimoto similarity of each row of `fp` to any active fingerprint."""
    inter = fp @ actives.T
    union = fp.sum(1)[:, None] + actives.sum(1)[None, :] - inter
    return (inter / np.maximum(union, 1e-9)).max(1)


def enrichment_factor(scores_active: np.ndarray, scores_background: np.ndarray,
                      top: float = 0.01) -> float:
    """Enrichment of known actives in the top fraction of a realistic screening list.

    Measured against a large background of random ZINC molecules rather than against the test
    set itself: with a test set that is 70 % active by construction, the best achievable
    enrichment would be 1.4, which says nothing. 1.0 means no better than random ranking.
    """
    y = np.concatenate([np.ones(len(scores_active)), np.zeros(len(scores_background))])
    p = np.concatenate([scores_active, scores_background])
    n = max(1, int(len(p) * top))
    order = np.argsort(-p)[:n]
    base = y.mean()
    return float(y[order].mean() / base) if base > 0 else float("nan")


def train_one(cid: str, meta: dict, args, background: np.ndarray | None = None) -> dict | None:
    df = pd.read_parquet(TARGETS / f"{cid}.parquet")
    if len(df) < 300 or df.label.nunique() < 2:
        log(f"{meta['symbol']:<9} too little data ({len(df)})")
        return None
    feat = featurize_smiles(df["smiles"].tolist(), n_jobs=args.n_jobs, quiet=True)
    feat["label"] = df["label"].to_numpy()
    feat["decoy"] = df["decoy"].to_numpy()
    feat = feat[feat["scaffold"] != ""].reset_index(drop=True)

    X = feature_matrix(feat)
    y = feat["label"].to_numpy(int)
    groups = feat["scaffold"].to_numpy()
    nj = args.n_jobs
    tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, groups))
    assert not set(groups[tr]) & set(groups[te]), "scaffold leakage between train and test"

    prep = Preprocessor().fit(pd.DataFrame(X[tr][:, :len(DESC_NAMES)], columns=DESC_NAMES))
    names = list(prep.cols) + [n for n in feature_names() if n.startswith("bit")]

    def transform(arr: np.ndarray) -> np.ndarray:
        """Scale the descriptor block (dropping the constants the Preprocessor removed) and
        leave the fingerprint bits as they are."""
        d = pd.DataFrame(arr[:, :len(DESC_NAMES)], columns=DESC_NAMES)
        return np.hstack([prep.transform(d), arr[:, len(DESC_NAMES):]])

    Xtr, Xte, ytr, yte = transform(X[tr]), transform(X[te]), y[tr], y[te]
    base_names = [n for n, _ in bases(nj)]
    oof = np.zeros((len(tr), len(base_names)))
    for a, b in StratifiedGroupKFold(3, shuffle=True, random_state=3).split(Xtr, ytr, groups[tr]):
        for j, (_, m) in enumerate(bases(nj)):
            m.fit(Xtr[a], ytr[a])
            oof[b, j] = m.predict_proba(Xtr[b])[:, 1]
    meta_lr = LogisticRegression(C=1.0, max_iter=1000).fit(_logit(oof), ytr)
    stack = StackedEnsemble([m.fit(Xtr, ytr) for _, m in bases(nj)], meta_lr, base_names)
    p = stack.predict_proba(Xte)[:, 1]

    oof_meta = np.zeros(len(tr))
    for a, b in StratifiedGroupKFold(3, shuffle=True, random_state=4).split(oof, ytr, groups[tr]):
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(_logit(oof[a]), ytr[a])
        oof_meta[b] = lr.predict_proba(_logit(oof[b]))[:, 1]
    grid = np.linspace(0.1, 0.9, 161)
    thr = float(grid[int(np.argmax([balanced_accuracy_score(ytr, (oof_meta >= t).astype(int))
                                    for t in grid]))])

    pred = (p >= thr).astype(int)
    res = evaluate(yte, pred, p, meta["symbol"])
    res["PR-AUC"] = float(average_precision_score(yte, p))
    res["Balanced_Accuracy"] = float(balanced_accuracy_score(yte, pred))
    res["Threshold"] = thr

    # ZINC decoys are assumed inactive, not measured, and they are easy to separate from
    # screened compounds. The honest headline number ignores them.
    is_decoy = feat["decoy"].to_numpy()[te]
    measured = ~is_decoy
    res_measured = None
    if measured.sum() >= 50 and len(set(yte[measured].tolist())) > 1:
        res_measured = {
            "n": int(measured.sum()),
            "ROC-AUC": float(roc_auc_score(yte[measured], p[measured])),
            "PR-AUC": float(average_precision_score(yte[measured], p[measured])),
            "Balanced_Accuracy": float(balanced_accuracy_score(yte[measured], pred[measured])),
        }

    # reliability against similarity to the nearest training active
    fp_all = X[:, len(DESC_NAMES):]
    act_fp = fp_all[tr][ytr == 1]
    sim = tanimoto_to_actives(fp_all[te], act_fp)
    reliability = []
    for lo, hi in zip(SIM_BINS[:-1], SIM_BINS[1:], strict=True):
        m = (sim >= lo) & (sim < hi)
        if m.sum() >= 20:
            reliability.append({"min_similarity": lo, "n": int(m.sum()),
                                "accuracy": float((pred[m] == yte[m]).mean()),
                                "roc_auc": (float(roc_auc_score(yte[m], p[m]))
                                            if len(set(yte[m].tolist())) > 1 else None)})
    ef = float("nan")
    if background is not None:
        bg = stack.predict_proba(transform(background))[:, 1]
        ef = enrichment_factor(p[(yte == 1) & measured], bg)
    auc_m = res_measured["ROC-AUC"] if res_measured else float("nan")
    log(f"{meta['symbol']:<9} {len(feat):>6,} mols  AUC={res['ROC-AUC']:.3f} "
        f"(measured only {auc_m:.3f})  bal={res['Balanced_Accuracy']:.3f}  "
        f"EF1%={ef:.0f}x  decoys={feat['decoy'].mean():.0%}")

    summary = {
        "chembl_id": cid, "symbol": meta["symbol"], "name": meta.get("name"),
        "dataset": {"n_molecules": int(len(feat)), "n_scaffolds": int(feat["scaffold"].nunique()),
                    "positive_rate": float(y.mean()), "decoy_fraction": float(feat["decoy"].mean()),
                    "n_train": int(len(tr)), "n_test": int(len(te))},
        "metrics": res, "metrics_measured_only": res_measured,
        "enrichment_factor_top1pct": ef, "reliability_vs_similarity": reliability,
        "base_models": base_names, "threshold": thr,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{cid}_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": stack, "lgbm": stack.bases[0], "preprocessor": prep,
                 "feature_columns": names, "descriptor_columns": list(prep.cols),
                 "threshold": thr, "chembl_id": cid, "symbol": meta["symbol"],
                 "name": meta.get("name"), "decoy_fraction": float(feat["decoy"].mean()),
                 "active_fingerprints": np.packbits(act_fp.astype(np.uint8), axis=1),
                 "active_smiles": feat["smiles"].to_numpy()[tr][ytr == 1],
                 "method": "Stacked ensemble (descriptors + Morgan fingerprint)",
                 "test_metrics": {k: res[k] for k in ("ROC-AUC", "PR-AUC", "F1-Score",
                                                      "Balanced_Accuracy", "Precision", "Recall")}},
                MODELS / f"target_{cid}.pkl", compress=("xz", 3))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--background", type=int, default=20000,
                    help="random ZINC molecules used as the screening background")
    args = ap.parse_args()
    kb = json.loads((PROC / "diseases.json").read_text(encoding="utf-8"))
    todo = [t for t in kb["targets"] if (TARGETS / f"{t['chembl_id']}.parquet").exists()
            and (args.targets is None or t["chembl_id"] in args.targets)]
    log(f"{len(todo)} targets with downloaded activity data")

    # a fixed background of random ZINC molecules, shared by every target's enrichment test
    bg_df = pd.read_parquet(PROC / "molecules_unlabeled_feat.parquet").sample(
        args.background, random_state=7)
    background = feature_matrix(bg_df)
    log(f"screening background: {len(bg_df):,} random ZINC molecules")

    done = []
    for t in todo:
        try:
            r = train_one(t["chembl_id"], t, args, background)
        except Exception as exc:  # noqa: BLE001 - one bad target must not stop the run
            log(f"{t['symbol']:<9} failed: {type(exc).__name__}: {exc}")
            continue
        if r:
            done.append(r)
    if done:
        a = [d["metrics"]["ROC-AUC"] for d in done]
        m = [d["metrics_measured_only"]["ROC-AUC"] for d in done if d["metrics_measured_only"]]
        log(f"DONE {len(done)} targets | mean ROC-AUC {np.mean(a):.3f} "
            f"| measured molecules only {np.mean(m):.3f}")


if __name__ == "__main__":
    main()
