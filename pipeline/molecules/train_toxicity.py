"""Train one toxicity classifier per endpoint (12 Tox21 assays + ClinTox).

Same protocol as the materials high-accuracy track (pipeline/train_high_accuracy.py), but on
molecules:
  * train/test split **grouped by Bemis-Murcko scaffold**, so no scaffold is on both sides --
    the molecular equivalent of grouping materials by reduced formula;
  * ablation: descriptors -> + Morgan fingerprint -> + Optuna tuning -> + stacking -> + threshold;
  * stacked LightGBM / XGBoost / HistGB with a logistic meta-learner on out-of-fold probabilities;
  * decision threshold chosen on out-of-fold training predictions only.

The endpoints are strongly imbalanced (3-16 % positives), so the headline metric is ROC-AUC and
PR-AUC, and the threshold maximises F1 rather than accuracy.

Usage:  uv run python pipeline/molecules/train_toxicity.py [--endpoints SR-MMP NR-AhR] [--trials 10]
Output: app/models/mol_{endpoint}_model.pkl, data/predictions/molecules/{endpoint}_summary.json
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
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "pipeline" / "molecules"))
from featurize_mol import DESC_NAMES, feature_matrix, feature_names  # noqa: E402
from hiacc_model import StackedEnsemble, _logit  # noqa: E402
from run_ssl_experiments import Preprocessor  # noqa: E402
from ssl_core import evaluate  # noqa: E402

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "predictions" / "molecules"
MODELS = ROOT / "app" / "models"
TOX21 = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
         "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
ENDPOINTS = [*TOX21, "CT_TOX", "FDA_APPROVED"]
TITLES = {
    "NR-AR": "Androgen receptor", "NR-AR-LBD": "Androgen receptor (LBD)",
    "NR-AhR": "Aryl hydrocarbon receptor", "NR-Aromatase": "Aromatase",
    "NR-ER": "Estrogen receptor", "NR-ER-LBD": "Estrogen receptor (LBD)",
    "NR-PPAR-gamma": "PPAR-gamma", "SR-ARE": "Oxidative stress (ARE)",
    "SR-ATAD5": "DNA damage (ATAD5)", "SR-HSE": "Heat-shock response",
    "SR-MMP": "Mitochondrial toxicity", "SR-p53": "p53 stress response",
    "CT_TOX": "Clinical trial toxicity", "FDA_APPROVED": "FDA approved",
}
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def safe(name: str) -> str:
    return name.replace("-", "_").lower()


def lgbm(p: dict | None = None, nj: int = -1):
    base = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.4, min_child_samples=10,
                n_jobs=nj, verbose=-1, random_state=42)
    base.update(p or {})
    return lgb.LGBMClassifier(**base)


def xgb(p: dict | None = None, nj: int = -1):
    base = dict(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                colsample_bytree=0.4, tree_method="hist", n_jobs=nj, random_state=42,
                eval_metric="logloss")
    base.update(p or {})
    return XGBClassifier(**base)


def metrics(y, p, t, name: str) -> dict:
    pred = (p >= t).astype(int)
    r = evaluate(y, pred, p, name)
    r["Balanced_Accuracy"] = balanced_accuracy_score(y, pred)
    r["PR-AUC"] = average_precision_score(y, p)
    r["Threshold"] = float(t)
    return r


def tune(X, y, groups, n_trials: int, nj: int) -> tuple[dict, float]:
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=0).split(X, y, groups))

    def objective(trial):
        p = dict(n_estimators=trial.suggest_int("n_estimators", 200, 900, step=100),
                 learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                 num_leaves=trial.suggest_int("num_leaves", 15, 127, log=True),
                 min_child_samples=trial.suggest_int("min_child_samples", 5, 60, log=True),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 0.8),
                 subsample=trial.suggest_float("subsample", 0.5, 1.0),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True))
        return float(np.mean([log_loss(y[va], lgbm(p, nj).fit(X[tr], y[tr])
                                       .predict_proba(X[va])[:, 1], labels=[0, 1])
                              for tr, va in folds]))

    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    st.optimize(objective, n_trials=n_trials)
    return st.best_params, st.best_value


def best_threshold(y, p) -> tuple[float, str]:
    """Decision threshold chosen on out-of-fold training predictions.

    F1 of the positive class for the usual case (3-16 % positives). When the *positive* class
    is the majority, as for FDA_APPROVED (95 % approved), maximising F1 degenerates to
    predicting "positive" for everything, so balanced accuracy is maximised instead.
    """
    if y.mean() > 0.5:
        grid = np.linspace(0.05, 0.95, 181)
        bal = [balanced_accuracy_score(y, (p >= t).astype(int)) for t in grid]
        return float(grid[int(np.argmax(bal))]), "balanced accuracy"
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return (float(thr[int(np.argmax(f1[:-1]))]) if len(thr) else 0.5), "F1"


def train_endpoint(endpoint: str, df: pd.DataFrame, args) -> dict | None:
    sub = df[df[endpoint].notna()].reset_index(drop=True)
    y = sub[endpoint].to_numpy(int)
    if len(sub) < 500 or y.sum() < 30:
        log(f"skip {endpoint}: {len(sub)} molecules / {int(y.sum())} positives")
        return None
    groups = sub["scaffold"].to_numpy()
    Xd = sub[DESC_NAMES].to_numpy(np.float32)
    Xall = feature_matrix(sub)
    nj = args.n_jobs

    tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(Xall, y, groups))
    assert not set(groups[tr]) & set(groups[te]), "scaffold leakage between train and test"
    ytr, yte = y[tr], y[te]
    log(f"{endpoint}: {len(sub):,} molecules, {sub['scaffold'].nunique():,} scaffolds, "
        f"positives {y.mean():.3f} | train {len(tr):,} test {len(te):,}")

    # the descriptor block is imputed and scaled; fingerprint bits are already 0/1.
    # Preprocessor drops constant descriptors, so the real column list comes from prep.cols.
    prep = Preprocessor().fit(pd.DataFrame(Xd[tr], columns=DESC_NAMES))
    names = list(prep.cols) + [n for n in feature_names() if n.startswith("bit")]

    def desc_only(idx):
        return prep.transform(pd.DataFrame(Xd[idx], columns=DESC_NAMES))

    def full(idx):
        return np.hstack([desc_only(idx), Xall[idx][:, len(DESC_NAMES):]])

    ablation = []

    def record(step, p, t=0.5):
        r = metrics(yte, p, t, step)
        ablation.append(r)
        log(f"   {step:<42} AUC={r['ROC-AUC']:.4f} PR={r['PR-AUC']:.4f} "
            f"F1={r['F1-Score']:.4f} bal={r['Balanced_Accuracy']:.4f}")
        return r

    m1 = lgbm(nj=nj).fit(desc_only(tr), ytr)
    record("1. RDKit descriptors (LightGBM)", m1.predict_proba(desc_only(te))[:, 1])
    Xtr, Xte = full(tr), full(te)
    m2 = lgbm(nj=nj).fit(Xtr, ytr)
    record("2. + Morgan fingerprint", m2.predict_proba(Xte)[:, 1])

    params, cv = tune(Xtr, ytr, groups[tr], args.trials, nj)
    m3 = lgbm(params, nj).fit(Xtr, ytr)
    record(f"3. + Optuna-tuned LightGBM (CV logloss {cv:.4f})", m3.predict_proba(Xte)[:, 1])

    def bases():
        return [("LightGBM", lgbm(params, nj)), ("XGBoost", xgb(nj=nj)),
                ("HistGB", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08,
                                                          max_leaf_nodes=31, random_state=42))]

    base_names = [n for n, _ in bases()]
    oof = np.zeros((len(tr), len(base_names)))
    for a, b in StratifiedGroupKFold(3, shuffle=True, random_state=3).split(Xtr, ytr, groups[tr]):
        for j, (_, m) in enumerate(bases()):
            m.fit(Xtr[a], ytr[a])
            oof[b, j] = m.predict_proba(Xtr[b])[:, 1]
    meta = LogisticRegression(C=1.0, max_iter=1000).fit(_logit(oof), ytr)
    stack = StackedEnsemble([m.fit(Xtr, ytr) for _, m in bases()], meta, base_names)
    p4 = stack.predict_proba(Xte)[:, 1]
    record(f"4. + stacking ({' + '.join(base_names)})", p4)

    oof_meta = np.zeros(len(tr))
    for a, b in StratifiedGroupKFold(3, shuffle=True, random_state=4).split(oof, ytr, groups[tr]):
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(_logit(oof[a]), ytr[a])
        oof_meta[b] = lr.predict_proba(_logit(oof[b]))[:, 1]
    t_best, criterion = best_threshold(ytr, oof_meta)
    final = record(f"5. + {criterion}-optimal threshold (t = {t_best:.2f}, chosen on train)",
                   p4, t_best)

    fpr, tpr, _ = roc_curve(yte, p4)
    keep = np.unique(np.linspace(0, len(fpr) - 1, 100).astype(int))
    imp = pd.Series(stack.bases[0].booster_.feature_importance("gain"), index=names)
    summary = {
        "endpoint": endpoint, "title": TITLES.get(endpoint, endpoint),
        "dataset": {"n_molecules": int(len(sub)), "n_scaffolds": int(sub["scaffold"].nunique()),
                    "positive_rate": float(y.mean()), "n_train": int(len(tr)),
                    "n_test": int(len(te)), "n_features": int(Xall.shape[1])},
        "ablation": ablation, "final": final, "threshold": t_best,
        "base_models": base_names, "tuned_params": params, "tuning_cv_logloss": cv,
        "threshold_criterion": criterion,
        "confusion": confusion_matrix(yte, (p4 >= t_best).astype(int)).tolist(),
        "roc": [{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)} for i in keep],
        "top_features": (imp / max(imp.sum(), 1)).sort_values(ascending=False).head(12).round(4)
        .to_dict(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{safe(endpoint)}_summary.json").write_text(json.dumps(summary, indent=1,
                                                                   default=float))
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": stack, "lgbm": stack.bases[0], "preprocessor": prep,
                 "feature_columns": names, "descriptor_columns": list(prep.cols),
                 "threshold": t_best, "endpoint": endpoint,
                 "title": TITLES.get(endpoint, endpoint),
                 "method": "Stacked ensemble (descriptors + Morgan fingerprint)",
                 "config": " + ".join(base_names) + f", t={t_best:.2f}",
                 "test_metrics": {k: final[k] for k in ("Accuracy", "Balanced_Accuracy",
                                                        "Precision", "Recall", "F1-Score",
                                                        "ROC-AUC", "PR-AUC")}},
                MODELS / f"mol_{safe(endpoint)}_model.pkl", compress=("xz", 3))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", nargs="+", default=ENDPOINTS)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    df = pd.read_parquet(PROC / "molecules_labeled_feat.parquet")
    done = []
    for e in args.endpoints:
        s = train_endpoint(e, df, args)
        if s:
            done.append((e, s["final"]))
    log("DONE " + " | ".join(f"{e} AUC={m['ROC-AUC']:.3f}" for e, m in done))


if __name__ == "__main__":
    main()
