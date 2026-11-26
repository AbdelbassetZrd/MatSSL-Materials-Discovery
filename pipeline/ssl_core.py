"""Shared training / evaluation code for the semi-supervised experiments.

Mirrors the friend's QSAR-SSL protocol (Baseline RF, Label Propagation, Label
Spreading, Self-Training, Co-Training; Composite Score = mean(F1, ROC-AUC)) and
extends it with a model zoo (RF, ExtraTrees, XGBoost, LightGBM, LogReg, kNN, MLP),
Optuna hyper-parameter search, several random seeds and much larger unlabeled pools.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier

warnings.filterwarnings("ignore")
N_JOBS = -1


# ----------------------------------------------------------------------------- metrics
def evaluate(y_true, y_pred, y_proba, name: str, **extra) -> dict:
    y_proba = np.nan_to_num(np.asarray(y_proba, dtype=float), nan=0.5)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba)
        prauc = average_precision_score(y_true, y_proba)
    except ValueError:
        auc, prauc = 0.5, float(np.mean(y_true))
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1,
        "ROC-AUC": auc,
        "PR-AUC": prauc,
        "Composite_Score": (f1 + auc) / 2.0,
        **extra,
    }


# ----------------------------------------------------------------------------- model zoo
def make_model(kind: str, seed: int = 42, **params):
    if kind == "RF":
        p = dict(n_estimators=300, n_jobs=N_JOBS, random_state=seed,
                 class_weight="balanced_subsample")
        p.update(params)
        return RandomForestClassifier(**p)
    if kind == "ExtraTrees":
        p = dict(n_estimators=300, n_jobs=N_JOBS, random_state=seed,
                 class_weight="balanced_subsample")
        p.update(params)
        return ExtraTreesClassifier(**p)
    if kind == "XGBoost":
        from xgboost import XGBClassifier

        p = dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                 colsample_bytree=0.6, tree_method="hist", n_jobs=N_JOBS,
                 random_state=seed, eval_metric="logloss", verbosity=0)
        p.update(params)
        return XGBClassifier(**p)
    if kind == "LightGBM":
        from lightgbm import LGBMClassifier

        p = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.6, n_jobs=N_JOBS,
                 random_state=seed, verbose=-1, class_weight="balanced")
        p.update(params)
        return LGBMClassifier(**p)
    if kind == "LogReg":
        p = dict(max_iter=2000, C=1.0, class_weight="balanced")
        p.update(params)
        return LogisticRegression(**p)
    if kind == "kNN":
        p = dict(n_neighbors=15, weights="distance", n_jobs=N_JOBS)
        p.update(params)
        return KNeighborsClassifier(**p)
    if kind == "MLP":
        p = dict(hidden_layer_sizes=(256, 128), alpha=1e-3, max_iter=300,
                 early_stopping=True, random_state=seed)
        p.update(params)
        return MLPClassifier(**p)
    raise ValueError(kind)


def proba_pos(model, X) -> np.ndarray:
    p = model.predict_proba(X)
    return p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else np.zeros(len(X))


def best_threshold_free_predict(proba: np.ndarray) -> np.ndarray:
    return (proba >= 0.5).astype(int)


# ----------------------------------------------------------------------------- SSL methods
@dataclass
class Data:
    X_l: np.ndarray
    y_l: np.ndarray
    X_u: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def run_baseline(d: Data, kind="RF", seed=42, params=None, label="Baseline") -> dict:
    t = time.time()
    m = make_model(kind, seed, **(params or {}))
    m.fit(d.X_l, d.y_l)
    pr = proba_pos(m, d.X_test)
    return evaluate(d.y_test, best_threshold_free_predict(pr), pr,
                    f"{label} {kind} ({len(d.y_l)} labeled)", Method="Baseline",
                    Base=kind, N_Unlabeled=0, N_Train=len(d.y_l),
                    Training_Time=time.time() - t), m


def _stack(d: Data):
    X = np.vstack([d.X_l, d.X_u])
    y = np.concatenate([d.y_l, -np.ones(len(d.X_u), dtype=int)])
    return X, y


class ClassMassNormalized:
    """Class Mass Normalization (Zhu, Ghahramani & Lafferty, 2003) for graph SSL.

    Graph methods tend to push every unlabeled node towards the majority class.
    CMN rescales the two class scores so that the total class mass over the
    unlabeled nodes matches the class prior estimated from the labeled nodes.
    """

    def __init__(self, graph_model, n_labeled: int, prior_pos: float, X_u=None):
        self.m = graph_model
        dist = np.nan_to_num(graph_model.label_distributions_[n_labeled:], nan=0.5)
        self.mass = dist.sum(axis=0) + 1e-12
        self.prior = np.array([1 - prior_pos, prior_pos])
        # When the graph over-smooths, every node is close to the prior and CMN alone
        # cannot separate the classes at 0.5. We then also shift the decision threshold
        # so that the fraction of unlabeled nodes predicted positive equals the prior
        # (monotone transform: ROC-AUC is unchanged).
        if X_u is not None and len(X_u):
            sample = X_u[np.random.default_rng(0).permutation(len(X_u))[:5000]]
            pu = self._cmn(np.nan_to_num(graph_model.predict_proba(sample), nan=0.5))[:, 1]
        else:
            pu = self._cmn(dist)[:, 1]
        self.t = float(np.clip(np.quantile(pu, 1 - prior_pos), 1e-6, 1 - 1e-6))

    def _cmn(self, p):
        s = p * self.prior / self.mass
        return s / (s.sum(axis=1, keepdims=True) + 1e-12)

    def predict_proba(self, X):
        p1 = np.clip(self._cmn(np.nan_to_num(self.m.predict_proba(X), nan=0.5))[:, 1],
                     1e-9, 1 - 1e-9)
        logit = np.log(p1 / (1 - p1)) - np.log(self.t / (1 - self.t))
        q = 1.0 / (1.0 + np.exp(-logit))
        return np.column_stack([1 - q, q])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def run_label_propagation(d: Data, kernel="knn", gamma=None, n_neighbors=10,
                          spreading=False, alpha=0.2, seed=42) -> dict:
    t = time.time()
    X, y = _stack(d)
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    common = dict(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors, max_iter=1000,
                  n_jobs=N_JOBS)
    g = LabelSpreading(alpha=alpha, **common) if spreading else LabelPropagation(**common)
    g.fit(X, y)
    m = ClassMassNormalized(g, len(d.y_l), float(np.mean(d.y_l)), d.X_u)
    pr = proba_pos(m, d.X_test)
    method = "Label Spreading" if spreading else "Label Propagation"
    hp = f"k={n_neighbors}" if kernel == "knn" else f"gamma={gamma:.3g}"
    if spreading:
        hp += f", alpha={alpha}"
    return evaluate(d.y_test, best_threshold_free_predict(pr), pr,
                    f"{method.replace(' ', '')} ({kernel}, {hp}, n={len(d.X_u)})",
                    Method=method, Base="graph", Kernel=kernel, Gamma=gamma,
                    N_Neighbors=n_neighbors, Alpha=alpha if spreading else None,
                    N_Unlabeled=len(d.X_u), N_Train=len(d.y_l),
                    Training_Time=time.time() - t), m


def run_self_training(d: Data, kind="RF", threshold=0.9, max_iter=10, seed=42,
                      params=None) -> dict:
    t = time.time()
    X, y = _stack(d)
    m = SelfTrainingClassifier(make_model(kind, seed, **(params or {})),
                               threshold=threshold, max_iter=max_iter)
    m.fit(X, y)
    pr = proba_pos(m, d.X_test)
    n_pseudo = int((m.labeled_iter_ > 0).sum())
    return evaluate(d.y_test, best_threshold_free_predict(pr), pr,
                    f"SelfTraining-{kind} (tau={threshold}, n={len(d.X_u)})",
                    Method="Self-Training", Base=kind, Threshold=threshold,
                    N_Unlabeled=len(d.X_u), N_Train=len(d.y_l),
                    N_Pseudo_Labeled=n_pseudo, Final_Train_Size=len(d.y_l) + n_pseudo,
                    Training_Time=time.time() - t), m


class CoTrainingClassifier:
    """Blum & Mitchell co-training with two feature views.

    Each round, each view's classifier labels the unlabeled samples it is most
    confident about (above ``threshold``, at most ``per_round`` per class,
    keeping the labeled class ratio) and adds them to the shared labeled set.
    Final prediction = average of the two views' probabilities.
    """

    def __init__(self, base, view1, view2, threshold=0.85, per_round=None, n_rounds=10):
        self.base, self.view1, self.view2 = base, np.asarray(view1), np.asarray(view2)
        self.threshold, self.per_round, self.n_rounds = threshold, per_round, n_rounds

    def fit(self, X_l, y_l, X_u):
        Xl, yl = X_l.copy(), y_l.copy()
        pool = np.arange(len(X_u))
        per_round = self.per_round or max(20, len(X_u) // (2 * self.n_rounds))
        pos_ratio = float(np.mean(y_l))
        self.n_pseudo_ = 0
        for _ in range(self.n_rounds):
            self.c1_ = clone(self.base).fit(Xl[:, self.view1], yl)
            self.c2_ = clone(self.base).fit(Xl[:, self.view2], yl)
            if len(pool) == 0:
                break
            add_idx, add_lab = [], []
            for clf, view in ((self.c1_, self.view1), (self.c2_, self.view2)):
                p = proba_pos(clf, X_u[pool][:, view])
                for cls, prob in ((1, p), (0, 1 - p)):
                    quota = int(round(per_round * (pos_ratio if cls == 1 else 1 - pos_ratio)))
                    cand = np.where(prob >= self.threshold)[0]
                    cand = cand[np.argsort(-prob[cand])][:max(quota, 1)]
                    add_idx.extend(pool[cand])
                    add_lab.extend([cls] * len(cand))
            if not add_idx:
                break
            add_idx = np.asarray(add_idx)
            add_lab = np.asarray(add_lab)
            _, first = np.unique(add_idx, return_index=True)  # drop duplicates
            add_idx, add_lab = add_idx[first], add_lab[first]
            Xl = np.vstack([Xl, X_u[add_idx]])
            yl = np.concatenate([yl, add_lab])
            self.n_pseudo_ += len(add_idx)
            pool = np.setdiff1d(pool, add_idx, assume_unique=True)
        self.c1_ = clone(self.base).fit(Xl[:, self.view1], yl)
        self.c2_ = clone(self.base).fit(Xl[:, self.view2], yl)
        return self

    def predict_proba(self, X):
        p = (proba_pos(self.c1_, X[:, self.view1]) + proba_pos(self.c2_, X[:, self.view2])) / 2
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def run_co_training(d: Data, view1, view2, kind="RF", threshold=0.85, n_rounds=10,
                    seed=42, params=None) -> dict:
    t = time.time()
    m = CoTrainingClassifier(make_model(kind, seed, **(params or {})), view1, view2,
                             threshold=threshold, n_rounds=n_rounds)
    m.fit(d.X_l, d.y_l, d.X_u)
    pr = m.predict_proba(d.X_test)[:, 1]
    return evaluate(d.y_test, best_threshold_free_predict(pr), pr,
                    f"CoTraining-{kind} (tau={threshold}, n={len(d.X_u)})",
                    Method="Co-Training", Base=kind, Threshold=threshold,
                    N_Unlabeled=len(d.X_u), N_Train=len(d.y_l),
                    N_Pseudo_Labeled=m.n_pseudo_, Final_Train_Size=len(d.y_l) + m.n_pseudo_,
                    Training_Time=time.time() - t), m


class ClusterThenLabel:
    """Cluster-then-Label SSL (unsupervised clustering + label transfer).

    1. K-Means (unsupervised) on labeled + unlabeled samples.
    2. In every cluster that contains >= ``min_labeled`` labeled samples whose
       labels are at least ``purity`` pure, the unlabeled members receive the
       cluster's majority label (cluster assumption of SSL).
    3. The base learner is trained on labeled + pseudo-labeled samples, the
       pseudo-labels weighted by ``pseudo_weight``.
    """

    def __init__(self, base, n_clusters=100, purity=0.8, min_labeled=3, pseudo_weight=0.5,
                 seed=42):
        self.base, self.n_clusters, self.purity = base, n_clusters, purity
        self.min_labeled, self.pseudo_weight, self.seed = min_labeled, pseudo_weight, seed

    def fit(self, X_l, y_l, X_u):
        from sklearn.cluster import MiniBatchKMeans

        X = np.vstack([X_l, X_u])
        km = MiniBatchKMeans(self.n_clusters, random_state=self.seed, batch_size=4096,
                             n_init=3).fit(X)
        cl_l, cl_u = km.labels_[: len(X_l)], km.labels_[len(X_l):]
        pseudo = np.full(len(X_u), -1)
        for c in range(self.n_clusters):
            in_c = cl_l == c
            if in_c.sum() < self.min_labeled:
                continue
            p = y_l[in_c].mean()
            if p >= self.purity:
                pseudo[cl_u == c] = 1
            elif p <= 1 - self.purity:
                pseudo[cl_u == c] = 0
        sel = pseudo >= 0
        self.n_pseudo_ = int(sel.sum())
        Xt = np.vstack([X_l, X_u[sel]])
        yt = np.concatenate([y_l, pseudo[sel]])
        w = np.concatenate([np.ones(len(y_l)), np.full(self.n_pseudo_, self.pseudo_weight)])
        self.model_ = clone(self.base).fit(Xt, yt, sample_weight=w)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)


def run_cluster_then_label(d: Data, kind="RF", n_clusters=100, purity=0.8, seed=42,
                           params=None) -> dict:
    t = time.time()
    m = ClusterThenLabel(make_model(kind, seed, **(params or {})), n_clusters=n_clusters,
                         purity=purity, seed=seed).fit(d.X_l, d.y_l, d.X_u)
    pr = proba_pos(m, d.X_test)
    return evaluate(d.y_test, best_threshold_free_predict(pr), pr,
                    f"ClusterThenLabel-{kind} (k={n_clusters}, purity={purity}, "
                    f"n={len(d.X_u)})", Method="Cluster-then-Label", Base=kind,
                    Threshold=purity, N_Unlabeled=len(d.X_u), N_Train=len(d.y_l),
                    N_Pseudo_Labeled=m.n_pseudo_, Final_Train_Size=len(d.y_l) + m.n_pseudo_,
                    Training_Time=time.time() - t), m


def summarise(df: pd.DataFrame, keys=("Method", "Model_Config")) -> pd.DataFrame:
    """Mean +- std over seeds for every configuration."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC",
               "Composite_Score", "Training_Time"]
    g = df.groupby(list(keys), dropna=False)[metrics]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std"))
    out["n_seeds"] = g.size()
    return out.sort_values("Composite_Score_mean", ascending=False).reset_index()
