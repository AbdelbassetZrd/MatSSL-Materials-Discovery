"""Traditional materials-informatics pipeline for photovoltaic band-gap classification
(counterpart of the friend's AMES QSAR notebook):

  1. Model comparison on all 142 composition descriptors (RF, ExtraTrees, XGBoost,
     LightGBM, LogReg, kNN, SVM) -> best model by ROC-AUC
  2. Genetic-algorithm feature selection (DEAP): population 100, 20 generations,
     cx 0.8, mut 0.2, elite 5, fitness = ROC-AUC - lambda * n_selected / n_total
  3. Retraining on the GA-selected descriptors (full training set) vs all descriptors
  4. Applicability domain: PCA projection, K-Means (k=6) cluster assignment,
     distance-to-training (k-NN distance, 95th percentile threshold), with
     performance inside vs outside the domain
  5. Saves the predictor package for the web application

Usage:  uv run python pipeline/band_gap_ga_ad.py
"""

from __future__ import annotations

import json
import random
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from deap import base, creator, tools  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ssl_core as core  # noqa: E402
from run_ssl_experiments import Preprocessor  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PROC, OUT, MODELS = (ROOT / "data" / "processed", ROOT / "data" / "predictions" / "band_gap",
                     ROOT / "app" / "models")
POP, NGEN, CXPB, MUTPB, ELITE, LAMBDA = 100, 20, 0.8, 0.2, 5, 0.05
GA_SAMPLE = 6000
N_JOBS = 6


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:  # noqa: PLR0915
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROC / "band_gap_features.parquet")
    feat = [c for c in df.columns if c.startswith(("MagpieData", "frac ")) or c.endswith("-norm")]
    log(f"band-gap dataset: {len(df):,} compositions, positive (1.1-1.7 eV) rate "
        f"{df['target'].mean():.3f}")
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=df["target"],
                              random_state=42)
    prep = Preprocessor().fit(df.iloc[tr][feat])
    cols = prep.cols
    X_tr, X_te = prep.transform(df.iloc[tr][feat]), prep.transform(df.iloc[te][feat])
    y_tr, y_te = df["target"].values[tr], df["target"].values[te]
    log(f"train {len(y_tr):,} | test {len(y_te):,} | descriptors after cleaning {len(cols)}")

    # ------------------------------------------------------------- 1. model comparison
    log("1. MODEL COMPARISON (all descriptors)")
    comp = []
    rng = np.random.default_rng(0)
    for kind in ["RF", "ExtraTrees", "XGBoost", "LightGBM", "LogReg", "kNN", "SVM"]:
        t = time.time()
        if kind == "SVM":  # kernel SVM does not scale to 60k samples: 10k subsample
            sub = rng.choice(len(y_tr), 10000, replace=False)
            m = SVC(C=1.0, probability=True, class_weight="balanced", random_state=0)
            m.fit(X_tr[sub], y_tr[sub])
        else:
            m = core.make_model(kind, 0, **({"n_jobs": N_JOBS} if kind != "LogReg" else {}))
            m.fit(X_tr, y_tr)
        pr = core.proba_pos(m, X_te)
        r = core.evaluate(y_te, (pr >= 0.5).astype(int), pr, kind,
                          Training_Time=time.time() - t)
        comp.append(r)
        log(f"   {kind:<11} ROC-AUC={r['ROC-AUC']:.4f} F1={r['F1-Score']:.4f} "
            f"PR-AUC={r['PR-AUC']:.4f} ({r['Training_Time']:.1f}s)")
    comp = pd.DataFrame(comp).sort_values("ROC-AUC", ascending=False)
    comp.to_csv(OUT / "band_gap_model_comparison.csv", index=False)
    best_kind = comp.iloc[0]["Model"]
    if best_kind in ("SVM", "kNN", "LogReg"):
        best_kind = "LightGBM"
    log(f"best model family: {comp.iloc[0]['Model']} -> GA fitness model {best_kind}")

    # ------------------------------------------------------------- 2. genetic algorithm
    log(f"2. GENETIC ALGORITHM (pop={POP}, gen={NGEN}, cx={CXPB}, mut={MUTPB}, "
        f"elite={ELITE}, lambda={LAMBDA})")
    n_feat = len(cols)
    ga_idx = np.concatenate([
        rng.choice(np.where(y_tr == 1)[0], int(GA_SAMPLE * y_tr.mean()), replace=False),
        rng.choice(np.where(y_tr == 0)[0], GA_SAMPLE - int(GA_SAMPLE * y_tr.mean()),
                   replace=False)])
    Xg, yg = X_tr[ga_idx], y_tr[ga_idx]
    skf = list(StratifiedKFold(3, shuffle=True, random_state=0).split(Xg, yg))
    cache: dict[tuple, float] = {}

    def cv_auc(mask: tuple) -> float:
        if mask in cache:
            return cache[mask]
        sel = np.flatnonzero(mask)
        if len(sel) == 0:
            return 0.5
        aucs = []
        for a, b in skf:
            m = core.make_model("LightGBM", 0, n_estimators=150, n_jobs=N_JOBS)
            m.fit(Xg[a][:, sel], yg[a])
            aucs.append(roc_auc_score(yg[b], core.proba_pos(m, Xg[b][:, sel])))
        cache[mask] = float(np.mean(aucs))
        return cache[mask]

    def fitness(ind):
        mask = tuple(ind)
        return (cv_auc(mask) - LAMBDA * sum(mask) / n_feat,)

    random.seed(0)
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    tb = base.Toolbox()
    tb.register("attr", lambda: 1 if random.random() < 0.5 else 0)
    tb.register("individual", tools.initRepeat, creator.Individual, tb.attr, n_feat)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("evaluate", fitness)
    tb.register("mate", tools.cxTwoPoint)
    tb.register("mutate", tools.mutFlipBit, indpb=1.0 / n_feat * 3)
    tb.register("select", tools.selTournament, tournsize=3)

    pop = tb.population(POP)
    history = []
    all_auc = cv_auc(tuple([1] * n_feat))
    log(f"   reference: all {n_feat} descriptors CV ROC-AUC={all_auc:.4f}")
    for gen in range(NGEN + 1):
        for ind in pop:
            if not ind.fitness.valid:
                ind.fitness.values = tb.evaluate(ind)
        fits = np.array([i.fitness.values[0] for i in pop])
        nsel = np.array([sum(i) for i in pop])
        best = pop[int(np.argmax(fits))]
        history.append({"generation": gen, "max_fitness": fits.max(),
                        "avg_fitness": fits.mean(), "min_fitness": fits.min(),
                        "best_auc": cv_auc(tuple(best)), "best_n_features": int(sum(best)),
                        "avg_n_features": nsel.mean()})
        log(f"   gen {gen:2d} max={fits.max():.4f} avg={fits.mean():.4f} "
            f"best AUC={history[-1]['best_auc']:.4f} with {sum(best)} features "
            f"(avg {nsel.mean():.1f})")
        if gen == NGEN:
            break
        elite = tools.selBest(pop, ELITE)
        offspring = [tb.clone(i) for i in tb.select(pop, POP - ELITE)]
        for c1, c2 in zip(offspring[::2], offspring[1::2], strict=False):
            if random.random() < CXPB:
                tb.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
        for m in offspring:
            if random.random() < MUTPB:
                tb.mutate(m)
                del m.fitness.values
        pop = [tb.clone(e) for e in elite] + offspring
    hist = pd.DataFrame(history)
    hist.to_csv(OUT / "band_gap_ga_history.csv", index=False)
    best = tools.selBest(pop, 1)[0]
    sel = np.flatnonzero(best)
    sel_names = [cols[i] for i in sel]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(hist["generation"], hist["max_fitness"], "o-", label="max fitness")
    axes[0].plot(hist["generation"], hist["avg_fitness"], "s-", label="average fitness")
    axes[0].plot(hist["generation"], hist["best_auc"], "^-", label="ROC-AUC of best")
    axes[0].set(xlabel="Generation", ylabel="Fitness / ROC-AUC",
                title="(A) Fitness evolution")
    axes[1].plot(hist["generation"], hist["best_n_features"], "o-", label="best individual")
    axes[1].plot(hist["generation"], hist["avg_n_features"], "s-", label="population average")
    axes[1].set(xlabel="Generation", ylabel="Number of selected descriptors",
                title="(B) Feature count evolution")
    for a in axes:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(OUT / "band_gap_ga_evolution.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------- 3. retrain
    log("3. RETRAINING ON FULL TRAINING SET: all descriptors vs GA-selected descriptors")
    res = {}
    for name, idx in (("all", np.arange(n_feat)), ("ga", sel)):
        m = core.make_model(best_kind, 0, n_jobs=N_JOBS).fit(X_tr[:, idx], y_tr)
        pr = core.proba_pos(m, X_te[:, idx])
        res[name] = core.evaluate(y_te, (pr >= 0.5).astype(int), pr, f"{best_kind}-{name}",
                                  n_features=len(idx))
        log(f"   {name:<4} ({len(idx):3d} descriptors) ROC-AUC={res[name]['ROC-AUC']:.4f} "
            f"F1={res[name]['F1-Score']:.4f} PR-AUC={res[name]['PR-AUC']:.4f}")
        if name == "ga":
            final_model, final_proba = m, pr
    cats = {"electronegativity": "Electronegativity", "radius / volume": ("CovalentRadius",
            "GSvolume"), "valence electrons": ("Valence", "Unfilled"),
            "periodic position": ("Column", "Row", "Number", "Mendeleev"),
            "thermal / mass": ("MeltingT", "AtomicWeight"), "elemental band gap / magnetism":
            ("GSbandgap", "GSmagmom"), "structure (space group)": "SpaceGroup",
            "stoichiometry": ("-norm", "frac ")}
    cat_counts = {k: sum(any(s in n for s in (v if isinstance(v, tuple) else (v,)))
                         for n in sel_names) for k, v in cats.items()}
    log(f"   descriptor categories: {cat_counts}")

    # ------------------------------------------------------------- 4. applicability domain
    log("4. APPLICABILITY DOMAIN (PCA, K-Means k=6, distance 95th percentile)")
    Xs_tr, Xs_te = X_tr[:, sel], X_te[:, sel]
    pca = PCA(n_components=2, random_state=0).fit(Xs_tr)
    P_tr, P_te = pca.transform(Xs_tr), pca.transform(Xs_te)
    km = KMeans(6, random_state=0, n_init=10).fit(P_tr)
    cl_tr, cl_te = km.labels_, km.predict(P_te)
    nn = NearestNeighbors(n_neighbors=6).fit(Xs_tr)
    d_tr = nn.kneighbors(Xs_tr)[0][:, 1:].mean(axis=1)   # skip self
    d_te = nn.kneighbors(Xs_te, n_neighbors=5)[0].mean(axis=1)
    thr = float(np.percentile(d_tr, 95))
    inside = d_te <= thr
    ad_in = core.evaluate(y_te[inside], (final_proba[inside] >= 0.5).astype(int),
                          final_proba[inside], "inside AD")
    ad_out = core.evaluate(y_te[~inside], (final_proba[~inside] >= 0.5).astype(int),
                           final_proba[~inside], "outside AD")
    log(f"   PCA variance (2 PCs): {pca.explained_variance_ratio_.sum():.3f} "
        f"(PC1 {pca.explained_variance_ratio_[0]:.3f}, PC2 {pca.explained_variance_ratio_[1]:.3f})")
    log(f"   AD threshold {thr:.3f}: {inside.sum():,}/{len(inside):,} test compositions "
        f"inside ({inside.mean():.1%}); ROC-AUC inside={ad_in['ROC-AUC']:.4f} "
        f"outside={ad_out['ROC-AUC']:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    s_tr = rng.choice(len(P_tr), min(15000, len(P_tr)), replace=False)
    axes[0].scatter(P_tr[s_tr, 0], P_tr[s_tr, 1], c=y_tr[s_tr], cmap="coolwarm", s=3, alpha=.5)
    axes[0].set(title="(A) Training set in PCA space (red = PV window)",
                xlabel="PC1", ylabel="PC2")
    axes[1].scatter(P_tr[s_tr, 0], P_tr[s_tr, 1], c="lightgrey", s=3, alpha=.4, label="train")
    axes[1].scatter(P_te[:, 0], P_te[:, 1], c="tab:blue", s=3, alpha=.4, label="test")
    axes[1].set(title="(B) Test set projected onto the training PCA space",
                xlabel="PC1", ylabel="PC2")
    axes[1].legend(markerscale=5)
    fig.tight_layout()
    fig.savefig(OUT / "band_gap_ad_pca.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(P_tr[s_tr, 0], P_tr[s_tr, 1], c=cl_tr[s_tr], cmap="tab10", s=3, alpha=.5)
    axes[0].scatter(*km.cluster_centers_.T, c="black", marker="X", s=120)
    axes[0].set(title="(A) Training set: K-Means clusters (k=6)", xlabel="PC1", ylabel="PC2")
    axes[1].scatter(P_te[:, 0], P_te[:, 1], c=cl_te, cmap="tab10", s=3, alpha=.5)
    axes[1].scatter(*km.cluster_centers_.T, c="black", marker="X", s=120)
    axes[1].set(title="(B) Test set assigned to training clusters", xlabel="PC1", ylabel="PC2")
    fig.tight_layout()
    fig.savefig(OUT / "band_gap_ad_kmeans.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(d_tr, bins=80, alpha=.6, density=True, label="train (leave-self-out)")
    axes[0].hist(d_te, bins=80, alpha=.6, density=True, label="test")
    axes[0].axvline(thr, color="red", ls="--", label=f"95th percentile = {thr:.2f}")
    axes[0].set(title="(A) Mean distance to 5 nearest training neighbours",
                xlabel="distance", ylabel="density")
    axes[0].legend()
    axes[1].bar(["inside AD", "outside AD"], [inside.sum(), (~inside).sum()],
                color=["tab:green", "tab:red"])
    axes[1].set(title=f"(B) {inside.mean():.1%} of test compositions inside the AD",
                ylabel="compositions")
    fig.tight_layout()
    fig.savefig(OUT / "band_gap_ad_distance.png", dpi=150)
    plt.close(fig)

    # cluster share train vs test
    share = pd.DataFrame({"train": pd.Series(cl_tr).value_counts(normalize=True),
                          "test": pd.Series(cl_te).value_counts(normalize=True)}).sort_index()

    # ------------------------------------------------------------- 5. save
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "selected_features": sel_names,
                 "selected_idx": sel.tolist(), "feature_columns": cols,
                 "pca": pca, "kmeans": km, "nn": nn, "ad_threshold": thr,
                 "task": "band_gap", "target": "in_pv_window (1.1-1.7 eV)"},
                MODELS / "band_gap_model.pkl")
    joblib.dump(prep, MODELS / "band_gap_scaler.pkl")
    summary = {
        "dataset": {"n": len(df), "positive_rate": float(df["target"].mean()),
                    "n_train": len(y_tr), "n_test": len(y_te), "n_descriptors": n_feat},
        "model_comparison": comp[["Model", "ROC-AUC", "F1-Score", "PR-AUC", "Accuracy",
                                  "Training_Time"]].to_dict(orient="records"),
        "ga": {"population": POP, "generations": NGEN, "cx_prob": CXPB, "mut_prob": MUTPB,
               "elite": ELITE, "lambda": LAMBDA, "fitness_sample": GA_SAMPLE,
               "initial_best_auc": float(hist.iloc[0]["best_auc"]),
               "final_best_auc": float(hist.iloc[-1]["best_auc"]),
               "initial_max_fitness": float(hist.iloc[0]["max_fitness"]),
               "final_max_fitness": float(hist.iloc[-1]["max_fitness"]),
               "initial_best_n_features": int(hist.iloc[0]["best_n_features"]),
               "final_n_features": len(sel), "initial_avg_n_features":
               float(hist.iloc[0]["avg_n_features"]),
               "final_avg_n_features": float(hist.iloc[-1]["avg_n_features"]),
               "initial_min_fitness": float(hist.iloc[0]["min_fitness"]),
               "final_min_fitness": float(hist.iloc[-1]["min_fitness"]),
               "all_descriptors_cv_auc": all_auc, "descriptor_categories": cat_counts,
               "selected_descriptors": sel_names},
        "test_all_descriptors": res["all"], "test_ga_descriptors": res["ga"],
        "applicability_domain": {
            "pca_variance_2pc": float(pca.explained_variance_ratio_.sum()),
            "pc1": float(pca.explained_variance_ratio_[0]),
            "pc2": float(pca.explained_variance_ratio_[1]),
            "kmeans_k": 6, "cluster_share": share.to_dict(),
            "distance_threshold_95pct": thr, "n_test": int(len(inside)),
            "n_inside": int(inside.sum()), "n_outside": int((~inside).sum()),
            "fraction_inside": float(inside.mean()), "metrics_inside": ad_in,
            "metrics_outside": ad_out},
    }
    (OUT / "band_gap_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log("saved band-gap results and predictor package")


if __name__ == "__main__":
    main()
