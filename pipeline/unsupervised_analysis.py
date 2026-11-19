"""Unsupervised learning on the materials data (no labels used for learning).

1. Association-rule mining (Apriori): every composition is a "transaction" whose
   items are its chemical elements (+ the property item, used only as a rule
   consequent). Rules such as {Fe, Co} -> magnetic are scored by support,
   confidence and lift. Also mined on GNoME alone (element co-occurrence).
2. Clustering: PCA projection + K-Means on the composition features of the
   labeled and unlabeled (GNoME) materials; k chosen with elbow + silhouette;
   clusters profiled by dominant elements and by property rates.

Usage:  uv run python pipeline/unsupervised_analysis.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlxtend.frequent_patterns import apriori, association_rules  # noqa: E402
from pymatgen.core import Composition  # noqa: E402
from sklearn.cluster import KMeans, MiniBatchKMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from featurize import feature_names  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "predictions" / "unsupervised"

TASKS = {
    "stability": "STABLE",
    "magnetism": "MAGNETIC",
    "metallicity": "METAL",
    "band_gap": "PV_GAP",
}


def element_sets(formulas: pd.Series) -> list[list[str]]:
    cache = {}
    out = []
    for f in formulas:
        if f not in cache:
            try:
                cache[f] = sorted(el.symbol for el in Composition(f).elements)
            except Exception:
                cache[f] = []
        out.append(cache[f])
    return out


def onehot(sets: list[list[str]]) -> pd.DataFrame:
    from mlxtend.preprocessing import TransactionEncoder

    te = TransactionEncoder()
    arr = te.fit(sets).transform(sets, sparse=False)
    return pd.DataFrame(arr, columns=te.columns_)


def mine_property_rules(task: str, item: str, min_support: float, log) -> pd.DataFrame:
    df = pd.read_parquet(PROC / f"{task}_features.parquet", columns=["formula", "target"])
    X = onehot(element_sets(df["formula"]))
    X[item] = df["target"].astype(bool).values
    # keep elements that appear often enough to form rules
    X = X.loc[:, X.mean() >= min_support]
    freq = apriori(X, min_support=min_support, use_colnames=True, max_len=3, low_memory=True)
    rules = association_rules(freq, metric="lift", min_threshold=1.2,
                              num_itemsets=len(X))
    rules = rules[rules["consequents"].apply(lambda c: c == frozenset([item]))]
    base = float(df["target"].mean())
    # confidence threshold relative to the base rate (rare classes such as PV_GAP)
    rules = rules[rules["confidence"] >= min(0.5, 2 * base)].sort_values(
        ["lift", "confidence"], ascending=False)
    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["consequents"] = item
    base_rate = float(df["target"].mean())
    log(f"{task}: {len(df):,} transactions, base rate {base_rate:.3f}, "
        f"{len(freq):,} frequent itemsets, {len(rules):,} rules -> {item}")
    keep = ["antecedents", "consequents", "support", "confidence", "lift", "leverage",
            "conviction"]
    return rules[keep].reset_index(drop=True), base_rate, len(df)


def plot_rules(rules: pd.DataFrame, title: str, path: Path, top=15) -> None:
    if rules.empty:
        return
    r = rules.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(r) + 1.5))
    bars = ax.barh(r["antecedents"] + "  →  " + r["consequents"], r["lift"],
                   color=plt.cm.viridis(r["confidence"].values))
    for b, c, s in zip(bars, r["confidence"], r["support"], strict=True):
        ax.text(b.get_width() * 1.01, b.get_y() + b.get_height() / 2,
                f"conf={c:.2f}, supp={s:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Lift")
    ax.set_title(title)
    ax.set_xlim(0, r["lift"].max() * 1.35)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def clustering(log) -> dict:
    cols = feature_names()
    frames = []
    for task in TASKS:
        d = pd.read_parquet(PROC / f"{task}_features.parquet")
        frames.append(d.sample(min(len(d), 25000), random_state=0)
                      .assign(source=task, prop=d["target"]))
    gn = pd.read_parquet(PROC / "unlabeled_gnome_features.parquet")
    frames.append(gn.sample(100000, random_state=0).assign(source="gnome", target=np.nan))
    data = pd.concat(frames, ignore_index=True)
    X = data[cols]
    X = X.loc[:, X.nunique() > 1]
    X = X.fillna(X.median())
    X = X.clip(X.quantile(0.001), X.quantile(0.999), axis=1)
    Z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=10, random_state=0).fit(Z)
    P = pca.transform(Z)
    log(f"PCA: first 2 components explain {pca.explained_variance_ratio_[:2].sum():.3f}, "
        f"10 components {pca.explained_variance_ratio_.sum():.3f}")

    ks = list(range(3, 13))
    inertia, sil = [], []
    rng = np.random.default_rng(0)
    sub = rng.choice(len(P), 15000, replace=False)
    for k in ks:
        km = MiniBatchKMeans(k, random_state=0, n_init=5, batch_size=8192).fit(P)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(P[sub], km.predict(P[sub])))
        log(f"   k={k:2d} inertia={km.inertia_:.3e} silhouette={sil[-1]:.4f}")
    best_k = ks[int(np.argmax(sil))]
    km = KMeans(best_k, random_state=0, n_init=10).fit(P)
    data["cluster"] = km.labels_

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(ks, inertia, "o-")
    axes[0].set(xlabel="k", ylabel="Inertia (within-cluster SSE)", title="Elbow method")
    axes[1].plot(ks, sil, "o-", color="tab:orange")
    axes[1].axvline(best_k, ls="--", color="grey")
    axes[1].set(xlabel="k", ylabel="Silhouette score", title=f"Silhouette (best k={best_k})")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "kmeans_elbow_silhouette.png", dpi=150)
    plt.close(fig)

    show = rng.choice(len(P), 30000, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sc = axes[0].scatter(P[show, 0], P[show, 1], c=data["cluster"].values[show], s=2,
                         cmap="tab10", alpha=0.6)
    axes[0].set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
                title=f"K-Means clusters (k={best_k}) in PCA space")
    axes[0].legend(*sc.legend_elements(), title="cluster", fontsize=8, markerscale=1.5)
    for src, color in [("gnome", "lightgrey"), ("stability", "tab:blue"),
                       ("magnetism", "tab:red"), ("metallicity", "tab:green"),
                       ("band_gap", "tab:purple")]:
        m = data["source"].values[show] == src
        axes[1].scatter(P[show][m, 0], P[show][m, 1], s=2, alpha=0.5, c=color,
                        label="GNoME (unlabeled)" if src == "gnome" else src)
    axes[1].set(xlabel="PC1", ylabel="PC2", title="Labeled datasets vs unlabeled pool")
    axes[1].legend(markerscale=6, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "kmeans_pca_clusters.png", dpi=150)
    plt.close(fig)

    # profile clusters: property rate per cluster + dominant elements
    prof = []
    sets = element_sets(data["formula"])
    data["elements"] = sets
    for c in range(best_k):
        dc = data[data["cluster"] == c]
        row = {"cluster": c, "size": len(dc),
               "share_gnome": float((dc["source"] == "gnome").mean()),
               "mean_n_elements": float(dc["elements"].map(len).mean())}
        for task, item in TASKS.items():
            t = dc[dc["source"] == task]["target"]
            row[f"{item}_rate"] = float(t.mean()) if len(t) else np.nan
        top = pd.Series([e for s in dc["elements"] for e in s]).value_counts(normalize=True)
        row["top_elements"] = ", ".join(f"{e} ({v:.0%})" for e, v in top.head(5).items())
        prof.append(row)
    prof = pd.DataFrame(prof)
    prof.to_csv(OUT / "kmeans_cluster_profiles.csv", index=False)
    rate_cols = [f"{i}_rate" for i in TASKS.values()]
    fig, ax = plt.subplots(figsize=(12, 5))
    prof.set_index("cluster")[rate_cols].plot.bar(ax=ax)
    ax.set(ylabel="Fraction of positive labels", title="Property rates per K-Means cluster "
           "(clusters found without labels)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "kmeans_cluster_property_rates.png", dpi=150)
    plt.close(fig)
    log("cluster profiles:\n" + prof.to_string(index=False))
    return {"best_k": best_k, "silhouette": dict(zip(ks, map(float, sil), strict=True)),
            "pca_explained_2d": float(pca.explained_variance_ratio_[:2].sum()),
            "n_points": len(data)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    def log(m):
        print(m, flush=True)

    summary = {"association_rules": {}}
    log("=== ASSOCIATION RULES (Apriori) ===")
    for task, item in TASKS.items():
        ms = 0.002 if task in ("stability",) else 0.003
        rules, base, n = mine_property_rules(task, item, ms, log)
        rules.to_csv(OUT / f"association_rules_{task}.csv", index=False)
        plot_rules(rules, f"Top association rules → {item} (base rate {base:.1%})",
                   OUT / f"association_rules_{task}.png")
        log(rules.head(10).to_string(index=False))
        summary["association_rules"][task] = {
            "n_transactions": n, "base_rate": base, "n_rules": len(rules),
            "top_rules": rules.head(10).to_dict(orient="records")}

    log("=== ELEMENT CO-OCCURRENCE RULES ON GNoME (fully unsupervised) ===")
    gn = pd.read_parquet(PROC / "unlabeled_gnome_features.parquet", columns=["formula"])
    G = onehot(element_sets(gn["formula"]))
    G = G.loc[:, G.mean() >= 0.01]
    freq = apriori(G, min_support=0.01, use_colnames=True, max_len=3, low_memory=True)
    rules = association_rules(freq, metric="lift", min_threshold=1.5, num_itemsets=len(G))
    rules = rules[(rules["confidence"] >= 0.2) & (rules["lift"] >= 2)].sort_values("lift", ascending=False)
    for c in ("antecedents", "consequents"):
        rules[c] = rules[c].apply(lambda s: ", ".join(sorted(s)))
    rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]]
    rules.to_csv(OUT / "association_rules_gnome_elements.csv", index=False)
    plot_rules(rules.reset_index(drop=True), "GNoME element co-occurrence rules",
               OUT / "association_rules_gnome_elements.png")
    log(rules.head(10).to_string(index=False))
    summary["gnome_element_rules"] = {"n_transactions": len(G), "n_rules": len(rules),
                                      "top_rules": rules.head(10).to_dict(orient="records")}

    log("=== CLUSTERING (PCA + K-Means) ===")
    summary["clustering"] = clustering(log)
    (OUT / "unsupervised_summary.json").write_text(json.dumps(summary, indent=2,
                                                              default=str))
    log("saved to " + str(OUT))


if __name__ == "__main__":
    main()
