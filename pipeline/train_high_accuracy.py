"""High-accuracy track: all labels + composition AND crystal-structure descriptors +
tuned gradient boosting + stacking + accuracy-optimal decision threshold.

Each step is evaluated on the same held-out test set (ablation), so the contribution of every
ingredient is visible. The split is grouped by reduced formula: polymorphs of one formula are
never on both sides (no leakage through near-identical compositions).

Usage:  uv run python pipeline/train_high_accuracy.py --task metallicity
Outputs: data/predictions/high_accuracy/{task}_summary.json, {task}_ablation.csv/.png
         app/models/{task}_hiacc_model.pkl
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss, roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from featurize import feature_names, featurize_formulas  # noqa: E402
from hiacc_model import StackedEnsemble, _logit  # noqa: E402
from ssl_core import evaluate  # noqa: E402
from structure_features import STRUCT_COLUMNS  # noqa: E402

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "predictions" / "high_accuracy"
MODELS = ROOT / "app" / "models"
TITLES = {"metallicity": "Metallicity", "magnetism": "Magnetism", "stability": "Stability",
          "band_gap": "PV band gap"}
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def stratified_take(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx_pos, idx_neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n_pos = max(2, int(round(n * len(idx_pos) / len(y))))
    return np.concatenate([rng.choice(idx_pos, n_pos, replace=False),
                           rng.choice(idx_neg, n - n_pos, replace=False)])


def metrics(y, p, t, name: str) -> dict:
    pred = (p >= t).astype(int)
    r = evaluate(y, pred, p, name)
    r["Balanced_Accuracy"] = balanced_accuracy_score(y, pred)
    r["Threshold"] = t
    return r


def lgbm(params: dict | None = None, n_jobs: int = -1):
    base = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, min_child_samples=20,
                n_jobs=n_jobs, verbose=-1, random_state=42)
    base.update(params or {})
    return lgb.LGBMClassifier(**base)


def xgb(params: dict | None = None, n_jobs: int = -1):
    base = dict(n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.8,
                colsample_bytree=0.8, tree_method="hist", n_jobs=n_jobs, random_state=42,
                eval_metric="logloss")
    base.update(params or {})
    return XGBClassifier(**base)


def tune(kind: str, X, y, groups, n_trials: int, n_jobs: int) -> tuple[dict, float]:
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=0).split(X, y, groups))

    def objective(trial):
        if kind == "lgbm":
            p = dict(n_estimators=trial.suggest_int("n_estimators", 300, 1500, step=100),
                     learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                     num_leaves=trial.suggest_int("num_leaves", 31, 255, log=True),
                     min_child_samples=trial.suggest_int("min_child_samples", 5, 80, log=True),
                     colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                     subsample=trial.suggest_float("subsample", 0.5, 1.0),
                     reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True))
            make = lambda: lgbm(p, n_jobs)  # noqa: E731
        else:
            p = dict(n_estimators=trial.suggest_int("n_estimators", 300, 1500, step=100),
                     learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                     max_depth=trial.suggest_int("max_depth", 4, 12),
                     min_child_weight=trial.suggest_float("min_child_weight", 0.5, 20, log=True),
                     colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                     subsample=trial.suggest_float("subsample", 0.5, 1.0),
                     reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True))
            make = lambda: xgb(p, n_jobs)  # noqa: E731
        losses = []
        for tr, va in folds:
            m = make().fit(X[tr], y[tr])
            losses.append(log_loss(y[va], m.predict_proba(X[va])[:, 1]))
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def best_accuracy_threshold(y, p) -> float:
    grid = np.linspace(0.15, 0.85, 141)
    acc = [np.mean((p >= t) == y) for t in grid]
    return float(grid[int(np.argmax(acc))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TITLES))
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--tune-sample", type=int, default=60000)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    task, nj = args.task, args.n_jobs
    OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    df = pd.read_parquet(PROC / f"{task}_struct.parquet")
    comp, ok = featurize_formulas(pd.Series(df["formula"].unique()))
    comp.index = pd.Series(df["formula"].unique())[ok].values
    df = df[df["formula"].isin(comp.index)].reset_index(drop=True)
    Xc = comp.reindex(df["formula"]).to_numpy(np.float32)
    Xs = df[STRUCT_COLUMNS].to_numpy(np.float32)
    X = np.hstack([Xc, Xs])
    names = feature_names() + STRUCT_COLUMNS
    y = df["target"].to_numpy(int)
    groups = df["formula"].to_numpy()
    log(f"{task}: {len(df):,} structures, {df['formula'].nunique():,} formulas, "
        f"{X.shape[1]} features, positive rate {y.mean():.3f}")

    tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, groups))
    assert not set(groups[tr]) & set(groups[te]), "formula leakage between train and test"
    if len(te) > 20000:
        te = te[stratified_take(y[te], 20000, np.random.default_rng(7))]
    log(f"train {len(tr):,} | test {len(te):,} (grouped by formula, no overlap)")
    ytr, yte = y[tr], y[te]
    ablation = []

    def record(name, p, t=0.5):
        r = metrics(yte, p, t, name)
        ablation.append(r)
        log(f"  {name:<45} acc={r['Accuracy']:.4f} bal={r['Balanced_Accuracy']:.4f} "
            f"F1={r['F1-Score']:.4f} AUC={r['ROC-AUC']:.4f}")
        return r

    # ------------------------------------------------------------------ 1-2: features
    m1 = lgbm(n_jobs=nj).fit(Xc[tr], ytr)
    record("1. Composition only (LightGBM, all labels)", m1.predict_proba(Xc[te])[:, 1])
    m2 = lgbm(n_jobs=nj).fit(X[tr], ytr)
    record("2. + crystal-structure descriptors", m2.predict_proba(X[te])[:, 1])

    # ------------------------------------------------------------------ 3: tuning
    rng = np.random.default_rng(1)
    sub = tr if len(tr) <= args.tune_sample else rng.choice(tr, args.tune_sample, replace=False)
    log(f"Optuna on {len(sub):,} training structures ({args.trials} trials per family)")
    p_lgb, cv_lgb = tune("lgbm", X[sub], y[sub], groups[sub], args.trials, nj)
    p_xgb, cv_xgb = tune("xgb", X[sub], y[sub], groups[sub], max(10, args.trials * 2 // 3), nj)
    log(f"  tuned LightGBM logloss={cv_lgb:.4f} {p_lgb}")
    log(f"  tuned XGBoost  logloss={cv_xgb:.4f} {p_xgb}")
    m3 = lgbm(p_lgb, nj).fit(X[tr], ytr)
    p3 = m3.predict_proba(X[te])[:, 1]
    record("3. + Optuna-tuned LightGBM", p3)

    # ------------------------------------------------------------------ 4: stacking
    def bases():
        b = [("LightGBM", lgbm(p_lgb, nj)), ("XGBoost", xgb(p_xgb, nj)),
             ("HistGB", HistGradientBoostingClassifier(max_iter=500, learning_rate=0.08,
                                                       max_leaf_nodes=63, l2_regularization=1.0,
                                                       random_state=42))]
        if len(tr) <= 150_000:
            b.append(("ExtraTrees", ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2,
                                                         max_features=0.3, n_jobs=nj,
                                                         random_state=42)))
        return b

    base_names = [n for n, _ in bases()]
    oof = np.zeros((len(tr), len(base_names)))
    for k, (a, b) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=3)
                               .split(X[tr], ytr, groups[tr])):
        for j, (_, m) in enumerate(bases()):
            m.fit(X[tr][a], ytr[a])
            oof[b, j] = m.predict_proba(X[tr][b])[:, 1]
        log(f"  stacking fold {k + 1}/3 done")
    meta = LogisticRegression(C=1.0, max_iter=1000).fit(_logit(oof), ytr)
    fitted = []
    for _, m in bases():
        fitted.append(m.fit(X[tr], ytr))
    stack = StackedEnsemble(fitted, meta, base_names)
    p4 = stack.predict_proba(X[te])[:, 1]
    record(f"4. + stacking ({' + '.join(base_names)})", p4)

    # ------------------------------------------------------------------ 5: threshold
    oof_meta = np.zeros(len(tr))
    for a, b in StratifiedGroupKFold(3, shuffle=True, random_state=4).split(oof, ytr, groups[tr]):
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(_logit(oof[a]), ytr[a])
        oof_meta[b] = lr.predict_proba(_logit(oof[b]))[:, 1]
    t_best = best_accuracy_threshold(ytr, oof_meta)
    final = record(f"5. + accuracy-optimal threshold (t = {t_best:.2f}, chosen on train)",
                   p4, t_best)

    # ------------------------------------------------------------------ outputs
    pred = (p4 >= t_best).astype(int)
    cm = confusion_matrix(yte, pred).tolist()
    fpr, tpr, _ = roc_curve(yte, p4)
    keep = np.unique(np.linspace(0, len(fpr) - 1, 120).astype(int))
    imp = pd.Series(stack.bases[0].booster_.feature_importance("gain"), index=names)
    top = (imp / imp.sum()).sort_values(ascending=False).head(15)
    abl = pd.DataFrame(ablation)
    abl.to_csv(OUT / f"{task}_ablation.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    steps = [s.split(" (")[0].split(". ", 1)[1] if ". " in s else s for s in abl["Model"]]
    xs = np.arange(len(abl))
    ax.bar(xs - 0.2, abl["Accuracy"], 0.4, label="Accuracy", color="#4f46e5")
    ax.bar(xs + 0.2, abl["Balanced_Accuracy"], 0.4, label="Balanced accuracy", color="#0891b2")
    for x, v in zip(xs, abl["Accuracy"], strict=True):
        ax.text(x - 0.2, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)
    lo = float(min(abl["Accuracy"].min(), abl["Balanced_Accuracy"].min()))
    ax.set_ylim(max(0, lo - 0.05), 1.0)
    ax.set_xticks(xs, [f"{i + 1}. {s}" for i, s in enumerate(steps)], fontsize=8, rotation=12)
    ax.set_title(f"{TITLES[task]}: what each ingredient adds (held-out test set)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"{task}_ablation.png", dpi=150)
    plt.close(fig)

    summary = {
        "task": task, "title": TITLES[task],
        "dataset": {"n_structures": int(len(df)), "n_formulas": int(df["formula"].nunique()),
                    "positive_rate": float(y.mean()), "n_train": int(len(tr)),
                    "n_test": int(len(te)), "n_features": int(X.shape[1]),
                    "n_structure_features": len(STRUCT_COLUMNS)},
        "ablation": abl.to_dict(orient="records"),
        "final": final, "threshold": t_best, "base_models": base_names,
        "tuned_params": {"LightGBM": p_lgb, "XGBoost": p_xgb},
        "tuning_cv_logloss": {"LightGBM": cv_lgb, "XGBoost": cv_xgb},
        "top_features": top.round(4).to_dict(),
        "confusion": cm,
        "roc": [{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)} for i in keep],
        "runtime_min": (time.time() - T0) / 60,
    }
    (OUT / f"{task}_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": stack, "lgbm": stack.bases[0], "feature_columns": names,
                 "structure_columns": STRUCT_COLUMNS, "threshold": t_best,
                 "method": "Stacked ensemble (composition + structure)",
                 "config": " + ".join(base_names) + f", t={t_best:.2f}",
                 "test_metrics": {k: final[k] for k in ("Accuracy", "Balanced_Accuracy",
                                                         "Precision", "Recall", "F1-Score",
                                                         "ROC-AUC", "Composite_Score")},
                 "task": task},
                MODELS / f"{task}_hiacc_model.pkl", compress=("xz", 3))
    log(f"DONE {task}: accuracy {final['Accuracy']:.4f} (composition-only "
        f"{ablation[0]['Accuracy']:.4f}), balanced {final['Balanced_Accuracy']:.4f}")


if __name__ == "__main__":
    main()
