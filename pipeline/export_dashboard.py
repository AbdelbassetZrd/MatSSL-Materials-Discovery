"""Export every experiment result into one JSON file for the web dashboard
(frontend/public/data/dashboard.json, also served by GET /materials/experiments).

Includes datasets, SSL summaries, pool-size and label-budget curves, deployed models with
test ROC curves and confusion matrices, the band-gap GA/AD study, association rules, a
clustering sample, the chemical-space map used by the Results page and the autoencoder
training history.

Usage:  uv run python pipeline/export_dashboard.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ssl_experiments import Preprocessor, stratified_take  # noqa: E402
from self_supervised import MaskedAutoencoder  # noqa: E402

import __main__  # noqa: E402

__main__.Preprocessor = Preprocessor
__main__.MaskedAutoencoder = MaskedAutoencoder
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC, PRED, MODELS = ROOT / "data" / "processed", ROOT / "data" / "predictions", ROOT / "app" / "models"
OUT = ROOT / "frontend" / "public" / "data" / "dashboard.json"
TASKS = {
    "metallicity": {"title": "Metallicity", "source": "Materials Project", "positive": "metal"},
    "magnetism": {"title": "Magnetism", "source": "JARVIS-DFT", "positive": "magnetic"},
    "stability": {"title": "Stability", "source": "Alexandria", "positive": "stable"},
}
KEY = ["MagpieData mean Electronegativity", "MagpieData range Electronegativity",
       "MagpieData mean CovalentRadius", "MagpieData mean NValence",
       "MagpieData mean GSbandgap", "MagpieData mean GSmagmom", "MagpieData mean MeltingT",
       "0-norm"]
rng = np.random.default_rng(0)


def r4(x):
    if isinstance(x, dict):
        return {k: r4(v) for k, v in x.items()}
    if isinstance(x, list):
        return [r4(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    return x


def feat_cols(df):
    return [c for c in df.columns if c.startswith(("MagpieData", "frac ")) or c.endswith("-norm")]


def roc_points(y, p, n=60):
    fpr, tpr, _ = roc_curve(y, p)
    idx = np.unique(np.linspace(0, len(fpr) - 1, n).astype(int))
    return [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in idx]


def datasets_block(gn):
    out = []
    for task, meta in [*TASKS.items(), ("band_gap", {"title": "PV band gap",
                                                     "source": "Materials Project"})]:
        df = pd.read_parquet(PROC / f"{task}_features.parquet", columns=["formula", "target"] + KEY)
        u = gn[~gn["formula"].isin(set(df["formula"]))].sample(20000, random_state=0)
        s = df.sample(min(20000, len(df)), random_state=0)
        ks = [{"feature": k.replace("MagpieData ", ""),
               "ks": float(ks_2samp(s[k].dropna(), u[k].dropna()).statistic)} for k in KEY]
        out.append({"task": task, "title": meta["title"], "source": meta["source"],
                    "n": int(len(df)), "positive_rate": float(df["target"].mean()),
                    "n_positive": int(df["target"].sum()), "ks_vs_gnome": ks})
    return out


def ssl_block(task, pool):
    d = PRED / task / pool
    sj = d / f"ssl_summary_{task}.json"
    if not sj.exists():
        return None
    S = json.loads(sj.read_text())
    res = pd.read_csv(d / f"ssl_results_{task}.csv")
    ssl = res[~res["Method"].isin(["Baseline", "Self-Supervised (AE)"])]
    curves = {}
    for method, g in ssl.groupby("Method"):
        per = g.groupby(["Config", "N_Unlabeled"])[["Composite_Score", "F1-Score", "ROC-AUC"]].mean()
        best_cfg = per["Composite_Score"].groupby(level=0).mean().idxmax()
        cfg = per.loc[best_cfg].reset_index()
        sd = g[g["Config"] == best_cfg].groupby("N_Unlabeled")["Composite_Score"].std()
        curves[method] = {"config": best_cfg, "points": [
            {"n": int(r.N_Unlabeled), "composite": float(r.Composite_Score),
             "f1": float(r["F1-Score"]), "auc": float(r["ROC-AUC"]),
             "std": float(sd.get(r.N_Unlabeled, np.nan))} for _, r in cfg.iterrows()]}
    zoo = (res[res["Method"] == "Baseline"].groupby("Config")[["Composite_Score", "F1-Score",
                                                               "ROC-AUC"]]
           .agg(["mean", "std"]))
    zoo_rows = [{"model": c, "composite": float(zoo.loc[c, ("Composite_Score", "mean")]),
                 "composite_std": float(zoo.loc[c, ("Composite_Score", "std")]),
                 "f1": float(zoo.loc[c, ("F1-Score", "mean")]),
                 "auc": float(zoo.loc[c, ("ROC-AUC", "mean")])} for c in zoo.index]
    zoo_rows.sort(key=lambda r: -r["composite"])
    bc = pd.read_csv(d / f"{task}_label_budget_curve.csv")
    budget = []
    for (b, curve), g in bc.groupby(["Budget", "Curve"]):
        budget.append({"budget": int(b), "curve": curve, "composite": float(g["Composite_Score"].mean()),
                       "std": float(g["Composite_Score"].std())})
    keep_cols = ["Method", "Model_Config", "Accuracy_mean", "Precision_mean", "Recall_mean",
                 "F1-Score_mean", "ROC-AUC_mean", "Composite_Score_mean", "Composite_Score_std",
                 "Training_Time_mean"]
    return {
        "setup": S["training_setup"], "model_zoo": zoo_rows, "tuned_params": S["tuned_params"],
        "optuna_cv": S["optuna_cv_scores"],
        "best_baseline": S["best_supervised_baseline"],
        "best_overall": {k: S["best_model_overall"].get(k) for k in keep_cols},
        "best_ssl": {k: S["best_ssl_model"].get(k) for k in keep_cols},
        "best_per_method": [{k: r.get(k) for k in keep_cols} for r in S["best_per_method"]],
        "improvement_pct": S["ssl_improvement_over_best_baseline_pct"],
        "fully_supervised": S["fully_supervised_upper_bound"],
        "deployment": S.get("deployment_model_test_metrics"),
        "pool_curves": curves, "budget_curve": budget, "n_runs": S["n_training_runs"],
        "runtime_min": S["runtime_minutes"],
    }


def deployed_block(gn_ae=None):
    reg_f = MODELS / "registry.json"
    if not reg_f.exists():
        return {}
    reg = json.loads(reg_f.read_text())
    ae = joblib.load(MODELS / "self_supervised_autoencoder.pkl")
    out = {}
    for task, entry in reg.items():
        if task == "band_gap":
            continue
        pkg = joblib.load(MODELS / entry["model_file"])
        prep = joblib.load(MODELS / entry["scaler_file"])
        lab = pd.read_parquet(PROC / f"{task}_features.parquet")
        fc = feat_cols(lab)
        _, te = train_test_split(np.arange(len(lab)), test_size=0.2, stratify=lab["target"],
                                 random_state=42)
        y = lab["target"].values[te]
        keep = (stratified_take(y, 20000, np.random.default_rng(7)) if len(y) > 20000
                else np.arange(len(y)))
        raw = lab.iloc[te[keep]][fc]
        X = prep.transform(raw)
        space = pkg.get("feature_space", "raw")
        if space != "raw":
            Z = ae.encode(raw)
            X = Z if space == "ae" else np.hstack([X, Z])
        p = pkg["model"].predict_proba(X)[:, 1]
        yk = y[keep]
        cm = confusion_matrix(yk, (p >= pkg.get("threshold", 0.5)).astype(int)).tolist()
        out[task] = {**entry, "roc": roc_points(yk, p), "confusion": cm, "n_test": int(len(yk))}
        print(f"  deployed {task}: test AUC check, n={len(yk):,}")
    return out


def high_accuracy_block():
    """Structure-aware full-data models (pipeline/train_high_accuracy.py)."""
    H = PRED / "high_accuracy"
    out = {}
    for task in (*TASKS, "band_gap"):
        f = H / f"{task}_summary.json"
        if f.exists():
            S = json.loads(f.read_text())
            keys = ("Model", "Accuracy", "Balanced_Accuracy", "Precision", "Recall",
                    "F1-Score", "ROC-AUC", "Threshold")
            S["ablation"] = [{k: r[k] for k in keys} for r in S["ablation"]]
            out[task] = S
    return out


def band_gap_block():
    bj = PRED / "band_gap" / "band_gap_summary.json"
    if not bj.exists():
        return None
    B = json.loads(bj.read_text())
    hist = pd.read_csv(PRED / "band_gap" / "band_gap_ga_history.csv").to_dict(orient="records")
    pkg = joblib.load(MODELS / "band_gap_model.pkl")
    prep = joblib.load(MODELS / "band_gap_scaler.pkl")
    df = pd.read_parquet(PROC / "band_gap_features.parquet")
    fc = feat_cols(df)
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=df["target"],
                              random_state=42)
    X_te = prep.transform(df.iloc[te][fc])[:, pkg["selected_idx"]]
    p = pkg["model"].predict_proba(X_te)[:, 1]
    y = df["target"].values[te]
    t = pkg.get("threshold", 0.5)
    sample = rng.choice(tr, 3000, replace=False)
    Xs = prep.transform(df.iloc[sample][fc])[:, pkg["selected_idx"]]
    P = pkg["pca"].transform(Xs)
    space_map = [{"x": float(a), "y": float(b), "pv": int(c), "formula": f}
                 for (a, b), c, f in zip(P, df["target"].values[sample],
                                         df["formula"].values[sample], strict=True)]
    return {"summary": B, "ga_history": hist, "roc": roc_points(y, p),
            "confusion": confusion_matrix(y, (p >= t).astype(int)).tolist(), "threshold": t,
            "chemical_space": space_map,
            "pca_variance": B["applicability_domain"]["pca_variance_2pc"]}


def unsupervised_block(gn):
    U = PRED / "unsupervised"
    rules = {}
    for t in ("stability", "magnetism", "metallicity", "band_gap"):
        f = U / f"association_rules_{t}.csv"
        if f.exists():
            rules[t] = pd.read_csv(f).head(40).to_dict(orient="records")
    gr = U / "association_rules_gnome_elements.csv"
    summ = json.loads((U / "unsupervised_summary.json").read_text())
    prof = pd.read_csv(U / "kmeans_cluster_profiles.csv").to_dict(orient="records")
    # clustering sample for the scatter plot (same recipe as unsupervised_analysis.py)
    frames = []
    for task in ("metallicity", "magnetism", "stability"):
        d = pd.read_parquet(PROC / f"{task}_features.parquet").sample(1500, random_state=1)
        frames.append(d.assign(source=task))
    frames.append(gn.sample(3000, random_state=1).assign(source="gnome", target=np.nan))
    data = pd.concat(frames, ignore_index=True)
    fc = feat_cols(data)
    X = data[fc].loc[:, data[fc].nunique() > 1]
    X = X.fillna(X.median()).clip(X.quantile(0.001), X.quantile(0.999), axis=1)
    Z = StandardScaler().fit_transform(X)
    P = PCA(n_components=10, random_state=0).fit_transform(Z)
    cl = KMeans(summ["clustering"]["best_k"], random_state=0, n_init=10).fit_predict(P)
    pts = [{"x": float(a), "y": float(b), "cluster": int(c), "source": s}
           for (a, b), c, s in zip(P[:, :2], cl, data["source"], strict=True)]
    return {"rules": rules,
            "gnome_rules": pd.read_csv(gr).head(20).to_dict(orient="records") if gr.exists() else [],
            "summary": summ, "cluster_profiles": prof, "cluster_points": pts}


def main() -> None:
    gn = pd.read_parquet(PROC / "unlabeled_gnome_features.parquet")
    print("datasets ...")
    out = {"datasets": datasets_block(gn), "ssl": {}}
    for task in TASKS:
        out["ssl"][task] = {pool: ssl_block(task, pool) for pool in ("indist", "gnome")}
    print("deployed models ...")
    out["deployed"] = deployed_block()
    out["high_accuracy"] = high_accuracy_block()
    print("band gap ...")
    out["band_gap"] = band_gap_block()
    print("unsupervised ...")
    out["unsupervised"] = unsupervised_block(gn)
    ae = PRED / "self_supervised"
    out["self_supervised"] = {
        "history": pd.read_csv(ae / "autoencoder_training_history.csv").to_dict(orient="records"),
        "summary": json.loads((ae / "autoencoder_summary.json").read_text())}
    total_runs = sum(b["n_runs"] for t in out["ssl"].values() for b in t.values() if b)
    out["kpis"] = {"compositions": int(sum(d["n"] for d in out["datasets"]) + len(gn)),
                   "labeled": int(sum(d["n"] for d in out["datasets"])),
                   "unlabeled": int(len(gn)), "databases": 4, "models_trained": int(total_runs),
                   "tasks": 4}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r4(out), separators=(",", ":"), default=str), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
