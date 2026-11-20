"""Predict the superconducting critical temperature Tc from a chemical formula.

Data: the SuperCon-derived dataset of Hamidieh (UCI "Superconductivty Data", 21,263 measured
records). Only the formula and the measured Tc are used -- the 81 engineered features that ship
with it are ignored, so the model relies on the same 142 Magpie descriptors as the rest of the
project and can score any composition the app already knows how to featurise.

Protocol mirrors pipeline/train_high_accuracy.py: one row per reduced formula, a formula-grouped
train/test split, an ablation, and a stacked ensemble (LightGBM + XGBoost + HistGB with a ridge
meta-learner).

Usage:  uv run python pipeline/industries/train_superconductor.py
Output: app/models/superconductor_model.pkl,
        data/predictions/industries/superconductor_summary.json (+ parity plot)
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
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import KFold, train_test_split  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
from featurize import feature_names, featurize_formulas, reduced_formula  # noqa: E402
from hiacc_model import StackedRegressor  # noqa: E402
from run_ssl_experiments import Preprocessor  # noqa: E402

warnings.filterwarnings("ignore")
RAW = ROOT / "data" / "raw" / "superconductors" / "unique_m.csv"
OUT = ROOT / "data" / "predictions" / "industries"
MODELS = ROOT / "app" / "models"
LN2 = 77.0  # liquid-nitrogen boiling point: the practical "useful superconductor" threshold
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def models(nj: int) -> list[tuple[str, object]]:
    return [
        ("LightGBM", lgb.LGBMRegressor(n_estimators=900, learning_rate=0.05, num_leaves=63,
                                       colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
                                       min_child_samples=10, n_jobs=nj, verbose=-1,
                                       random_state=42)),
        ("XGBoost", XGBRegressor(n_estimators=900, learning_rate=0.05, max_depth=8,
                                 colsample_bytree=0.7, subsample=0.8, tree_method="hist",
                                 n_jobs=nj, random_state=42)),
        ("HistGB", HistGradientBoostingRegressor(max_iter=600, learning_rate=0.06,
                                                 max_leaf_nodes=63, random_state=42)),
    ]


def scores(y, p) -> dict:
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
            "R2": float(r2_score(y, p)),
            "AUC_above_77K": float(roc_auc_score(y > LN2, p)) if (y > LN2).any() else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--negatives", type=int, default=20000,
                    help="real materials assumed non-superconducting, added with Tc = 0 K")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    nj = args.n_jobs

    raw = pd.read_csv(RAW)[["material", "critical_temp"]].rename(
        columns={"material": "raw_formula", "critical_temp": "tc"})
    raw["formula"] = raw["raw_formula"].map(reduced_formula)
    raw = raw.dropna(subset=["formula"])
    # several measurements per composition: keep the median, and remember the spread
    g = raw.groupby("formula")["tc"]
    df = pd.DataFrame({"tc": g.median(), "tc_std": g.std().fillna(0.0),
                       "n_reports": g.size()}).reset_index()
    df["measured"] = True
    log(f"{len(raw):,} measurements -> {len(df):,} unique compositions "
        f"(Tc {df.tc.min():.1f}-{df.tc.max():.1f} K, {int((df.tc > LN2).sum()):,} above 77 K)")

    # SuperCon lists *only* superconductors, so a model trained on it alone assigns a Tc to
    # anything -- it answers "if this superconducts, how hot?" instead of "does it superconduct?".
    # Real materials from the Materials Project that SuperCon never reports are therefore added
    # as Tc = 0 K. The assumption is that superconductivity above a few kelvin is rare enough
    # for an unreported material to be treated as non-superconducting.
    n_neg = 0
    if args.negatives:
        mp = pd.read_parquet(ROOT / "data" / "processed" / "metallicity_features.parquet",
                             columns=["formula"])
        cand = mp[~mp["formula"].isin(set(df["formula"]))]["formula"].drop_duplicates()
        neg = cand.sample(min(args.negatives, len(cand)), random_state=0)
        n_neg = len(neg)
        df = pd.concat([df, pd.DataFrame({"formula": neg.to_numpy(), "tc": 0.0, "tc_std": 0.0,
                                          "n_reports": 0, "measured": False})], ignore_index=True)
        log(f"added {n_neg:,} Materials Project compositions never reported as superconducting "
            f"(Tc = 0 K) -> {len(df):,} rows")

    feats, ok = featurize_formulas(df["formula"])
    df = df[ok.to_numpy()].reset_index(drop=True)
    X_raw = feats.reset_index(drop=True)
    y = df["tc"].to_numpy(float)
    log(f"featurised {len(df):,} compositions x {X_raw.shape[1]} descriptors")

    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42,
                              stratify=pd.qcut(y, 5, labels=False, duplicates="drop"))
    prep = Preprocessor().fit(X_raw.iloc[tr])
    X = prep.transform(X_raw)
    ytr, yte = y[tr], y[te]
    log(f"train {len(tr):,} | test {len(te):,} unique compositions (no formula on both sides)")

    measured_te = df["measured"].to_numpy()[te]
    ablation = []

    def record(name, p):
        s = {"Model": name, **scores(yte, p)}
        # MAE restricted to the measured superconductors, where predicting 0 is not an option
        s["MAE_superconductors"] = float(mean_absolute_error(yte[measured_te], p[measured_te]))
        ablation.append(s)
        log(f"   {name:<34} MAE={s['MAE']:.2f} K  (superconductors only "
            f"{s['MAE_superconductors']:.2f} K)  R2={s['R2']:.4f}")
        return s

    for name, m in models(nj):
        record(name, m.fit(X[tr], ytr).predict(X[te]))

    base_names = [n for n, _ in models(nj)]
    oof = np.zeros((len(tr), len(base_names)))
    for a, b in KFold(3, shuffle=True, random_state=3).split(tr):
        for j, (_, m) in enumerate(models(nj)):
            m.fit(X[tr][a], ytr[a])
            oof[b, j] = m.predict(X[tr][b])
    meta = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(oof, ytr)
    stack = StackedRegressor([m.fit(X[tr], ytr) for _, m in models(nj)], meta, base_names)
    p = stack.predict(X[te])
    final = record(f"Stacked ({' + '.join(base_names)})", p)

    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.scatter(yte, p, s=6, alpha=0.35, color="#4f46e5", edgecolors="none")
    lim = [0, max(yte.max(), p.max()) * 1.02]
    ax.plot(lim, lim, "--", color="#1B1F3B", lw=1)
    ax.axhline(LN2, color="#059669", lw=0.8, ls=":")
    ax.axvline(LN2, color="#059669", lw=0.8, ls=":")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("measured $T_c$ (K)")
    ax.set_ylabel("predicted $T_c$ (K)")
    ax.set_title(f"Superconductor $T_c$: MAE {final['MAE']:.1f} K, R² {final['R2']:.3f}\n"
                 f"{len(te):,} held-out compositions (dotted line = 77 K)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "superconductor_parity.png", dpi=150)
    plt.close(fig)

    imp = pd.Series(stack.bases[0].booster_.feature_importance("gain"), index=list(prep.cols))
    summary = {
        "task": "superconductor", "title": "Superconducting critical temperature",
        "source": "SuperCon / Hamidieh (UCI), measured values",
        "dataset": {"n_measurements": int(len(raw)), "n_compositions": int(len(df)),
                    "n_superconductors": int(df["measured"].sum()), "n_negatives": int(n_neg),
                    "n_train": int(len(tr)), "n_test": int(len(te)),
                    "tc_max": float(df.tc.max()), "n_above_77K": int((df.tc > LN2).sum())},
        "ablation": ablation, "final": final, "base_models": base_names,
        "top_features": (imp / max(imp.sum(), 1)).sort_values(ascending=False).head(12).round(4)
        .to_dict(),
        "parity": [{"measured": round(float(a), 2), "predicted": round(float(b), 2)}
                   for a, b in zip(yte[:3000], p[:3000], strict=False)],
    }
    (OUT / "superconductor_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": stack, "lgbm": stack.bases[0], "preprocessor": prep,
                 "feature_columns": list(prep.cols), "all_feature_names": feature_names(),
                 "task": "superconductor", "target": "Tc (K)",
                 "method": "Stacked ensemble (composition descriptors)",
                 "config": " + ".join(base_names), "test_metrics": final,
                 "tc_threshold_K": LN2},
                MODELS / "superconductor_model.pkl", compress=("xz", 3))
    log(f"DONE superconductor: MAE {final['MAE']:.2f} K, R2 {final['R2']:.4f}, "
        f"AUC(>77K) {final['AUC_above_77K']:.4f}")


if __name__ == "__main__":
    main()
