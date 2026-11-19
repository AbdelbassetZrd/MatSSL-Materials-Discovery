"""Stacked ensemble used by the high-accuracy track (importable by the web application)."""

from __future__ import annotations

import numpy as np


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class StackedEnsemble:
    """Base classifiers -> logits of their probabilities -> logistic-regression meta-learner."""

    def __init__(self, bases: list, meta, names: list[str]):
        self.bases = bases
        self.meta = meta
        self.names = names

    def base_proba(self, X) -> np.ndarray:
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.bases])

    def predict_proba(self, X) -> np.ndarray:
        return self.meta.predict_proba(_logit(self.base_proba(X)))

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
