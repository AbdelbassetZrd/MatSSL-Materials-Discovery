"""Full semi-supervised experiment for one task (friend's protocol, extended).

Stages
  1-2. Load processed data, stratified 80/20 split, preprocessing (impute, clip, z-score)
  3.   Model zoo: supervised baselines on the scarce labeled set
       (RF, ExtraTrees, XGBoost, LightGBM, LogReg, kNN, MLP) x seeds
  4.   Optuna hyper-parameter search (5-fold CV, composite score) -> tuned base learner
  4b.  Self-supervised pre-training: classifiers on the masked-autoencoder embedding
  5-8. SSL sweep x unlabeled-pool sizes x hyper-parameters x seeds:
       Label Propagation, Label Spreading (class-mass normalised), Self-Training,
       Co-Training, Cluster-then-Label
  9.   Best configuration per method (mean composite over seeds) + figures
  10.  Label-budget curve (baseline vs best SSL) + fully-supervised upper bound
  11.  Deployment model = selected architecture re-trained on all training labels

Unlabeled pool (--pool):
  gnome   external unlabeled data: GNoME compositions not present in the labeled set
          (the friend's ZINC setting)
  indist  classic SSL setting: the rest of the training pool with its labels hidden

Usage:  uv run python pipeline/run_ssl_experiments.py magnetism --pool indist
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ssl_core as core  # noqa: E402
from featurize import view_columns  # noqa: E402
from self_supervised import MaskedAutoencoder  # noqa: E402,F401  (needed to unpickle)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PRED = ROOT / "data" / "predictions"
MODELS = ROOT / "app" / "models"

TASK_INFO = {
    "metallicity": ("Metallicity (Materials Project)", "is_metal"),
    "magnetism": ("Magnetism (JARVIS-DFT)", "is_magnetic"),
    "stability": ("Thermodynamic Stability (Alexandria)", "is_stable"),
    "band_gap": ("Photovoltaic Band Gap (Materials Project)", "in_pv_window"),
}


class Preprocessor:
    """Median imputation -> outlier clipping (mean +- 3 std) -> z-score."""

    def fit(self, X: pd.DataFrame):
        self.cols = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
        X = X[self.cols]
        self.median = X.median()
        X = X.fillna(self.median)
        self.lo, self.hi = X.mean() - 3 * X.std(), X.mean() + 3 * X.std()
        self.scaler = StandardScaler().fit(X.clip(self.lo, self.hi, axis=1))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.cols].fillna(self.median).clip(self.lo, self.hi, axis=1)
        return self.scaler.transform(X).astype(np.float32)


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def stratified_take(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx_pos, idx_neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n_pos = max(2, int(round(n * len(idx_pos) / len(y))))
    return np.concatenate([rng.choice(idx_pos, n_pos, replace=False),
                           rng.choice(idx_neg, n - n_pos, replace=False)])


def optuna_search(kind: str, X, y, n_trials: int, seed=0) -> tuple[dict, float]:
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)

    def objective(trial):
        if kind in ("RF", "ExtraTrees"):
            p = dict(n_estimators=trial.suggest_int("n_estimators", 100, 600, step=100),
                     max_depth=trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
                     min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                     max_features=trial.suggest_categorical("max_features",
                                                            ["sqrt", "log2", 0.3, 0.5]))
        elif kind == "XGBoost":
            p = dict(n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
                     max_depth=trial.suggest_int("max_depth", 3, 10),
                     learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                     subsample=trial.suggest_float("subsample", 0.5, 1.0),
                     colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                     min_child_weight=trial.suggest_float("min_child_weight", 1, 10, log=True))
        else:  # LightGBM
            p = dict(n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
                     num_leaves=trial.suggest_int("num_leaves", 8, 128, log=True),
                     learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                     subsample=trial.suggest_float("subsample", 0.5, 1.0),
                     colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                     min_child_samples=trial.suggest_int("min_child_samples", 5, 50))
        scores = []
        for tr, va in skf.split(X, y):
            m = core.make_model(kind, seed, **p).fit(X[tr], y[tr])
            pr = core.proba_pos(m, X[va])
            scores.append(core.evaluate(y[va], (pr >= 0.5).astype(int), pr, "cv")
                          ["Composite_Score"])
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


# ============================================================================ figures
def plot_learning_curves(df: pd.DataFrame, base: dict, title: str, path: Path,
                         pool_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ssl = df[~df["Method"].isin(["Baseline", "Self-Supervised (AE)"])]
    for ax, metric in zip(axes, ["Composite_Score", "F1-Score"], strict=True):
        for method, g in ssl.groupby("Method"):
            per_cfg = g.groupby(["N_Unlabeled", "Config"])[metric].mean().reset_index()
            curve = per_cfg.groupby("N_Unlabeled")[metric].max()
            ax.plot(curve.index, curve.values, marker="o", linewidth=2, label=method)
        ax.axhline(base[metric], color="black", linestyle="--", linewidth=2,
                   label="Best supervised baseline")
        ax.set_xscale("log")
        ax.set_xlabel(f"Number of unlabeled samples ({pool_name})")
        ax.set_ylabel(metric)
        ax.set_title(f"{title}: {metric} vs unlabeled data")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_methods_comparison(best_rows: pd.DataFrame, title: str, path: Path) -> None:
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Composite_Score"]
    fig, ax = plt.subplots(figsize=(15, 6))
    w = 0.85 / len(best_rows)
    x = np.arange(len(metrics))
    for i, (_, r) in enumerate(best_rows.iterrows()):
        vals = [r[f"{m}_mean"] for m in metrics]
        errs = [0 if np.isnan(r[f"{m}_std"]) else r[f"{m}_std"] for m in metrics]
        ax.bar(x + i * w, vals, w, yerr=errs, capsize=3, label=r["Method"])
    ax.set_xticks(x + w * (len(best_rows) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (mean over seeds)")
    ax.set_title(f"{title}: best configuration of each method (Composite = mean(F1, ROC-AUC))")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================================ main
def main() -> None:  # noqa: C901, PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(TASK_INFO))
    ap.add_argument("--pool", choices=["gnome", "indist"], default="indist")
    ap.add_argument("--budget", type=int, default=1000, help="labeled samples (scarcity)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sizes", default="1000,5000,25000,100000")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--max-test", type=int, default=20000,
                    help="cap on the evaluated test rows (stratified) for speed")
    ap.add_argument("--no-final", action="store_true", help="skip the deployment model")
    args = ap.parse_args()

    title, target_name = TASK_INFO[args.task]
    pool_name = {"gnome": "GNoME", "indist": "same database, labels hidden"}[args.pool]
    out_dir = PRED / args.task / args.pool
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ------------------------------------------------------------------ 1. data
    log(f"STEP 1: LOADING PROCESSED DATA ({title}) | unlabeled pool: {pool_name}")
    lab = pd.read_parquet(PROC / f"{args.task}_features.parquet")
    gn = pd.read_parquet(PROC / "unlabeled_gnome_features.parquet")
    gn = gn[~gn["formula"].isin(set(lab["formula"]))].reset_index(drop=True)
    feat_cols = [c for c in lab.columns if c.startswith(("MagpieData", "frac ")) or
                 c.endswith("-norm")]
    log(f"labeled compositions: {len(lab):,} | GNoME pool (deduplicated): {len(gn):,} | "
        f"features: {len(feat_cols)} | positive rate {lab['target'].mean():.3f}")

    # ------------------------------------------------------------------ 2. split
    log("STEP 2: STRATEGIC DATA SPLITTING (stratified 80/20, test never touched)")
    idx_train, idx_test = train_test_split(np.arange(len(lab)), test_size=0.2,
                                           stratify=lab["target"], random_state=42)
    prep = Preprocessor().fit(pd.concat([
        lab.iloc[idx_train][feat_cols],
        gn[feat_cols].sample(min(len(gn), 100000), random_state=0)]))
    raw_train = lab.iloc[idx_train][feat_cols]
    raw_test = lab.iloc[idx_test][feat_cols]
    X_train_all, y_train_all = prep.transform(raw_train), lab["target"].values[idx_train]
    X_test_full, y_test_full = prep.transform(raw_test), lab["target"].values[idx_test]
    X_gn = prep.transform(gn[feat_cols])

    ae_path = MODELS / "self_supervised_autoencoder.pkl"
    ae = joblib.load(ae_path) if ae_path.exists() else None
    if ae is not None:
        Z_train_all, Z_test_full, Z_gn = ae.encode(raw_train), ae.encode(raw_test), \
            ae.encode(gn[feat_cols])
        log(f"self-supervised embedding loaded: {Z_train_all.shape[1]} dims")

    keep = (stratified_take(y_test_full, args.max_test, np.random.default_rng(7))
            if len(y_test_full) > args.max_test else np.arange(len(y_test_full)))
    y_test = y_test_full[keep]
    cols = prep.cols
    v1_names, v2_names = view_columns()
    view2 = [i for i, c in enumerate(cols) if c in v2_names or
             (any(k in c for k in ("Valence", "Unfilled", "Column", "Row")) and
              (c.endswith(" mean") or c.endswith(" avg_dev")))]
    view1 = [i for i in range(len(cols)) if i not in view2]
    log(f"train pool {len(y_train_all):,} | test {len(y_test_full):,} (evaluated on "
        f"{len(y_test):,}) | co-training views: {len(view1)} + {len(view2)} features")

    # per-seed labeled budget + unlabeled ordering
    seeds = list(range(args.seeds))
    splits = {}
    for s in seeds:
        rng = np.random.default_rng(1000 + s)
        lab_idx = stratified_take(y_train_all, args.budget, rng)
        if args.pool == "indist":
            rest = np.setdiff1d(np.arange(len(y_train_all)), lab_idx)
            unl_idx = rng.permutation(rest)
        else:
            unl_idx = rng.permutation(len(X_gn))
        splits[s] = (lab_idx, unl_idx)
    max_pool = len(splits[0][1])
    sizes = sorted({min(int(s), max_pool) for s in args.sizes.split(",")})

    def data_for(seed: int, n_unl: int, space: str = "raw", lab_idx=None) -> core.Data:
        li, ui = splits[seed]
        li = li if lab_idx is None else lab_idx
        ui = ui[:n_unl]
        srcX = {"raw": (X_train_all, X_gn, X_test_full)}
        if ae is not None:
            srcX["ae"] = (Z_train_all, Z_gn, Z_test_full)
            srcX["raw+ae"] = (np.hstack([X_train_all, Z_train_all]),
                              np.hstack([X_gn, Z_gn]),
                              np.hstack([X_test_full, Z_test_full]))
        Xtr, Xg, Xte = srcX[space]
        Xu = Xtr[ui] if args.pool == "indist" else Xg[ui]
        return core.Data(Xtr[li], y_train_all[li], Xu, Xte[keep], y_test)

    results: list[dict] = []

    def record(res: dict, seed: int, config: str, space: str) -> None:
        res.update(Seed=seed, Config=config, Feature_Space=space)
        results.append(res)
        log(f"   {config:<44} n_u={res['N_Unlabeled']:<6} seed={seed} "
            f"F1={res['F1-Score']:.4f} AUC={res['ROC-AUC']:.4f} "
            f"Composite={res['Composite_Score']:.4f} ({res['Training_Time']:.1f}s)")

    # ------------------------------------------------------------------ 3. model zoo
    log(f"STEP 3: BASELINE MODEL ZOO (supervised, {args.budget} labeled)")
    for s in seeds:
        d = data_for(s, 0)
        for kind in ["RF", "ExtraTrees", "XGBoost", "LightGBM", "LogReg", "kNN", "MLP"]:
            res, _ = core.run_baseline(d, kind, seed=s)
            record(res, s, f"{kind} (default)", "raw")
    zoo = pd.DataFrame(results)
    zoo_rank = zoo.groupby("Base")["Composite_Score"].mean().sort_values(ascending=False)
    log("model zoo ranking:\n" + zoo_rank.to_string())

    # ------------------------------------------------------------------ 4. Optuna
    tree_fams = [k for k in zoo_rank.index if k in ("RF", "ExtraTrees", "XGBoost", "LightGBM")]
    log(f"STEP 4: OPTUNA HYPER-PARAMETER SEARCH ({args.trials} trials, 5-fold CV) for "
        f"{tree_fams[:2]}")
    tuned, cv_scores = {}, {}
    d0 = data_for(0, 0)
    for kind in tree_fams[:2]:
        params, cv = optuna_search(kind, d0.X_l, d0.y_l, args.trials)
        tuned[kind], cv_scores[kind] = params, cv
        log(f"   {kind}: CV composite {cv:.4f} with {params}")
    for s in seeds:
        d = data_for(s, 0)
        for kind, params in tuned.items():
            res, _ = core.run_baseline(d, kind, seed=s, params=params)
            record(res, s, f"{kind} (tuned)", "raw")
    sup = pd.DataFrame(results).groupby("Config")["Composite_Score"].mean()
    best_base_cfg = sup.idxmax()
    best_kind = best_base_cfg.split(" ")[0]
    best_params = tuned.get(best_kind, {}) if "tuned" in best_base_cfg else {}
    star = f"{best_kind}*"
    log(f"best supervised baseline: {best_base_cfg} composite={sup.max():.4f} "
        f"(used as tuned base learner '{star}')")

    # ------------------------------------------------------------------ 4b. self-supervised
    if ae is not None:
        log("STEP 4b: SELF-SUPERVISED PRE-TRAINING (masked autoencoder embedding)")
        for s in seeds:
            for space in ("ae", "raw+ae"):
                res, _ = core.run_baseline(data_for(s, 0, space), best_kind, seed=s,
                                           params=best_params, label="SelfSup")
                res["Method"] = "Self-Supervised (AE)"
                record(res, s, f"{star} on {space} features", space)

    # ------------------------------------------------------------------ 5-8. SSL
    def ssl_configs(n_unl: int):
        small = n_unl + args.budget <= 11000      # dense RBF graph still fits in memory
        large = n_unl > 10000                     # lighter settings for 25k-100k pools
        st_iter, ct_rounds = (3, 5) if large else (5, 8)
        C = [("LP knn k=10", "raw", lambda d, s: core.run_label_propagation(
                d, "knn", n_neighbors=10, seed=s)),
             ("LS knn k=10 alpha=0.2", "raw", lambda d, s: core.run_label_propagation(
                d, "knn", n_neighbors=10, spreading=True, alpha=0.2, seed=s)),
             ("LS knn k=10 alpha=0.8", "raw", lambda d, s: core.run_label_propagation(
                d, "knn", n_neighbors=10, spreading=True, alpha=0.8, seed=s))]
        if small:
            C += [("LP rbf", "raw", lambda d, s: core.run_label_propagation(d, "rbf", seed=s)),
                  ("LS rbf alpha=0.2", "raw", lambda d, s: core.run_label_propagation(
                      d, "rbf", spreading=True, alpha=0.2, seed=s))]
        if ae is not None:
            C += [("LS knn k=10 alpha=0.2 [AE space]", "ae",
                   lambda d, s: core.run_label_propagation(d, "knn", n_neighbors=10,
                                                           spreading=True, alpha=0.2, seed=s))]
        # Self-training: friend's reference (RF, tau=0.75) on small pools + tuned learner
        if not large:
            C += [("ST RF tau=0.75", "raw", lambda d, s: core.run_self_training(
                d, "RF", threshold=0.75, max_iter=st_iter, seed=s))]
        for tau in ((0.75, 0.9) if not large else (0.9,)):
            C += [(f"ST {star} tau={tau}", "raw", lambda d, s, t=tau: core.run_self_training(
                d, best_kind, threshold=t, max_iter=st_iter, seed=s, params=best_params))]
        if ae is not None and not large:
            C += [(f"ST {star} tau=0.9 [raw+AE]", "raw+ae",
                   lambda d, s: core.run_self_training(d, best_kind, threshold=0.9,
                                                       max_iter=st_iter, seed=s,
                                                       params=best_params))]
        # Co-training (two views)
        if not large:
            C += [("CT RF tau=0.85", "raw", lambda d, s: core.run_co_training(
                d, view1, view2, "RF", threshold=0.85, n_rounds=ct_rounds, seed=s))]
        C += [(f"CT {star} tau=0.85", "raw", lambda d, s: core.run_co_training(
                d, view1, view2, best_kind, threshold=0.85, n_rounds=ct_rounds, seed=s,
                params=best_params)),
              (f"CtL {star} k=100 purity=0.8", "raw", lambda d, s: core.run_cluster_then_label(
                d, best_kind, n_clusters=100, purity=0.8, seed=s, params=best_params))]
        if ae is not None:
            C += [(f"CtL {star} k=100 purity=0.8 [raw+AE]", "raw+ae",
                   lambda d, s: core.run_cluster_then_label(
                       d, best_kind, n_clusters=100, purity=0.8, seed=s, params=best_params))]
        return C

    log("STEP 5-8: SEMI-SUPERVISED SWEEP (LP, LS, Self-Training, Co-Training, "
        "Cluster-then-Label)")
    for n_unl in sizes:
        log(f"--- unlabeled pool size {n_unl:,}")
        for s in seeds:
            for name, space, fn in ssl_configs(n_unl):
                try:
                    res, _ = fn(data_for(s, n_unl, space), s)
                    record(res, s, name, space)
                except Exception as exc:  # keep the sweep alive, log the failure
                    log(f"   !! {name} failed at n={n_unl}: {exc}")
        pd.DataFrame(results).to_csv(out_dir / f"ssl_results_{args.task}.csv", index=False)

    # ------------------------------------------------------------------ 9. comparison
    all_df = pd.DataFrame(results)
    all_df["Model_Config"] = np.where(
        all_df["N_Unlabeled"] > 0,
        all_df["Config"] + " | n=" + all_df["N_Unlabeled"].astype(str), all_df["Config"])
    all_df.to_csv(out_dir / f"ssl_results_{args.task}.csv", index=False)
    summary = core.summarise(all_df)
    summary.to_csv(out_dir / f"ssl_summary_table_{args.task}.csv", index=False)
    best_per_method = (summary.groupby("Method", as_index=False).first()
                       .sort_values("Composite_Score_mean", ascending=False))
    best = best_per_method.iloc[0]
    log("STEP 9: COMPREHENSIVE RESULTS COMPARISON (best config per method)\n" +
        best_per_method[["Method", "Model_Config", "Composite_Score_mean",
                         "Composite_Score_std", "F1-Score_mean", "ROC-AUC_mean"]]
        .to_string(index=False))
    base_rows = all_df[all_df["Config"] == best_base_cfg]
    base_metrics = {m: base_rows[m].mean() for m in ("Composite_Score", "F1-Score")}
    plot_learning_curves(all_df, base_metrics, title,
                         out_dir / f"{args.task}_learning_curves.png", pool_name)
    plot_methods_comparison(best_per_method, title,
                            out_dir / f"{args.task}_methods_comparison_balanced.png")

    # ------------------------------------------------------------------ 10. budget curve
    log("STEP 10: LABEL-BUDGET CURVE (best baseline vs best SSL) + full-supervision bound")
    ssl_only = best_per_method[~best_per_method["Method"].isin(
        ["Baseline", "Self-Supervised (AE)"])]
    best_ssl = ssl_only.iloc[0]
    ssl_name, n_best_unl = best_ssl["Model_Config"].split(" | n=")
    n_best_unl = int(n_best_unl)
    lookup = {name: (space, fn) for name, space, fn in ssl_configs(n_best_unl)}
    ssl_space, ssl_fn = lookup[ssl_name]
    budget_rows = []
    for budget in sorted({250, 500, args.budget, 2500, 5000}):
        for s in seeds:
            li = stratified_take(y_train_all, budget, np.random.default_rng(5000 + s))
            rb, _ = core.run_baseline(data_for(s, 0, "raw", li), best_kind, seed=s,
                                      params=best_params)
            rs, _ = ssl_fn(data_for(s, n_best_unl, ssl_space, li), s)
            for r, curve in ((rb, "Best supervised baseline"), (rs, f"Best SSL: {ssl_name}")):
                budget_rows.append({**r, "Budget": budget, "Seed": s, "Curve": curve})
            log(f"   budget={budget:<5} seed={s} baseline={rb['Composite_Score']:.4f} "
                f"SSL={rs['Composite_Score']:.4f}")
    full_d = core.Data(X_train_all, y_train_all, X_gn[:0], X_test_full[keep], y_test)
    full, full_model = core.run_baseline(full_d, best_kind, seed=0, params=best_params,
                                         label="Fully supervised")
    log(f"   fully supervised ({len(y_train_all):,} labels): "
        f"composite={full['Composite_Score']:.4f}")
    budget_df = pd.DataFrame(budget_rows)
    budget_df.to_csv(out_dir / f"{args.task}_label_budget_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    for curve, g in budget_df.groupby("Curve"):
        m = g.groupby("Budget")["Composite_Score"].agg(["mean", "std"])
        ax.errorbar(m.index, m["mean"], yerr=m["std"], marker="o", capsize=4, label=curve)
    ax.axhline(full["Composite_Score"], color="green", linestyle=":",
               label=f"Fully supervised ({len(y_train_all):,} labels)")
    ax.set_xscale("log")
    ax.set_xlabel("Number of labeled samples")
    ax.set_ylabel("Composite score (mean of F1 and ROC-AUC)")
    ax.set_title(f"{title}: label-budget curve")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"{args.task}_label_budget_curve.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ 11. export
    deploy = {}
    if not args.no_final:
        log("STEP 11: DEPLOYMENT MODEL = selected SSL architecture, all training labels "
            "+ GNoME unlabeled pool")
        n_dep = min(n_best_unl, len(X_gn))
        Xsp = {"raw": (X_train_all, X_gn, X_test_full)}
        if ae is not None:
            Xsp["ae"] = (Z_train_all, Z_gn, Z_test_full)
            Xsp["raw+ae"] = (np.hstack([X_train_all, Z_train_all]), np.hstack([X_gn, Z_gn]),
                             np.hstack([X_test_full, Z_test_full]))
        Xtr, Xg, Xte = Xsp[ssl_space]
        dep_d = core.Data(Xtr, y_train_all, Xg[np.random.default_rng(0)
                                                .permutation(len(Xg))[:n_dep]],
                          Xte[keep], y_test)
        dep_res, dep_model = ssl_fn(dep_d, 0)
        log(f"   deployment model test composite={dep_res['Composite_Score']:.4f} "
            f"F1={dep_res['F1-Score']:.4f} AUC={dep_res['ROC-AUC']:.4f}")
        deploy = {k: dep_res[k] for k in ("Composite_Score", "F1-Score", "ROC-AUC",
                                          "Accuracy", "Precision", "Recall")}
        MODELS.mkdir(parents=True, exist_ok=True)
        tag = f"{args.task}_{args.pool}"
        joblib.dump({"model": dep_model, "config": ssl_name, "method": best_ssl["Method"],
                     "feature_space": ssl_space, "feature_columns": cols,
                     "view1": view1, "view2": view2, "target": target_name,
                     "task": args.task, "test_metrics": deploy},
                    MODELS / f"{tag}_model.pkl")
        joblib.dump(prep, MODELS / f"{tag}_scaler.pkl")

    imp = (best_ssl["Composite_Score_mean"] - sup.max()) / sup.max() * 100

    def clean(row):
        return {k: (float(v) if isinstance(v, (np.floating, float, np.integer)) else v)
                for k, v in row.items()}

    summ = {
        "experiment_date": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "experiment_type": f"{title} (Semi-Supervised)",
        "training_setup": {
            "n_labeled_compositions_total": int(len(lab)),
            "n_train_pool": int(len(y_train_all)), "n_test": int(len(y_test_full)),
            "n_test_evaluated": int(len(y_test)), "labeled_budget": args.budget,
            "unlabeled_pool": pool_name, "unlabeled_pool_size": int(max_pool),
            "unlabeled_sizes_tested": sizes, "seeds": len(seeds), "n_features": len(cols),
            "co_training_views": [len(view1), len(view2)], "stratified_sampling": True,
            "metric_used_for_selection": "Composite Score (mean of F1 & ROC-AUC), "
                                         "averaged over seeds",
        },
        "model_zoo_ranking": zoo_rank.to_dict(),
        "optuna_cv_scores": cv_scores, "tuned_params": tuned,
        "best_supervised_baseline": {"config": best_base_cfg, "composite_score": sup.max()},
        "best_model_overall": clean(best),
        "best_ssl_model": clean(best_ssl),
        "best_per_method": [clean(r) for _, r in best_per_method.iterrows()],
        "ssl_improvement_over_best_baseline_pct": float(imp),
        "fully_supervised_upper_bound": {k: full[k] for k in
                                         ("Composite_Score", "F1-Score", "ROC-AUC", "Accuracy")},
        "deployment_model_test_metrics": deploy,
        "methods_tested": sorted(all_df["Method"].unique().tolist()),
        "n_training_runs": int(len(all_df) + len(budget_df) + 1 + len(tuned) * args.trials * 5),
        "runtime_minutes": (time.time() - t0) / 60,
    }
    (out_dir / f"ssl_summary_{args.task}.json").write_text(json.dumps(summ, indent=2,
                                                                     default=str))
    log(f"DONE in {(time.time() - t0) / 60:.1f} min. Best overall: {best['Method']} "
        f"[{best['Model_Config']}] composite {best['Composite_Score_mean']:.4f} | "
        f"best SSL {best_ssl['Model_Config']} vs best baseline: {imp:+.2f}%")


if __name__ == "__main__":
    main()
