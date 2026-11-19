"""Self-supervised pre-training: masked-feature denoising autoencoder.

Pretext task (no labels): randomly mask 25 % of the composition features of each
GNoME material and train an MLP autoencoder (142 -> 128 -> 32 -> 128 -> 142) to
reconstruct the full, clean feature vector. The 32-d bottleneck is then used as
a learned "materials embedding" by the downstream (semi-)supervised models.

Usage:  uv run python pipeline/self_supervised.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from featurize import feature_names  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, MODELS = ROOT / "data" / "processed", ROOT / "app" / "models"
OUT = ROOT / "data" / "predictions" / "self_supervised"


class MaskedAutoencoder:
    """Small wrapper so the encoder can be reused by the pipeline and the API."""

    def __init__(self, hidden=(128, 32, 128), mask_rate=0.25, epochs=20, seed=0):
        self.hidden, self.mask_rate, self.epochs, self.seed = hidden, mask_rate, epochs, seed

    # ---- preprocessing identical for every task (fitted on GNoME only)
    def _prep_fit(self, X: pd.DataFrame) -> np.ndarray:
        self.cols = [c for c in feature_names() if c in X and X[c].nunique() > 1]
        Xc = X[self.cols]
        self.median = Xc.median()
        Xc = Xc.fillna(self.median)
        self.lo, self.hi = Xc.quantile(0.001), Xc.quantile(0.999)
        Xc = Xc.clip(self.lo, self.hi, axis=1)
        self.scaler = StandardScaler().fit(Xc)
        return self.scaler.transform(Xc).astype(np.float32)

    def prep(self, X: pd.DataFrame) -> np.ndarray:
        Xc = X[self.cols].fillna(self.median).clip(self.lo, self.hi, axis=1)
        return self.scaler.transform(Xc).astype(np.float32)

    def fit(self, X: pd.DataFrame, val_frac=0.05, log=print):
        Z = self._prep_fit(X)
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(Z))
        n_val = int(len(Z) * val_frac)
        Zva, Ztr = Z[idx[:n_val]], Z[idx[n_val:]]
        self.net = MLPRegressor(hidden_layer_sizes=self.hidden, activation="tanh",
                                learning_rate_init=1e-3, batch_size=512, alpha=1e-5,
                                random_state=self.seed)
        self.history = []
        for ep in range(1, self.epochs + 1):
            t = time.time()
            mask = rng.random(Ztr.shape) < self.mask_rate
            noisy = np.where(mask, 0.0, Ztr)  # masked features set to the mean (0)
            self.net.partial_fit(noisy, Ztr)
            mva = rng.random(Zva.shape) < self.mask_rate
            rec = self.net.predict(np.where(mva, 0.0, Zva))
            r2_masked = r2_score(Zva[mva], rec[mva])
            r2_all = r2_score(Zva, rec)
            self.history.append({"epoch": ep, "loss": float(self.net.loss_),
                                 "val_r2_all": float(r2_all),
                                 "val_r2_masked_features": float(r2_masked)})
            log(f"epoch {ep:2d} loss={self.net.loss_:.4f} val R2={r2_all:.4f} "
                f"(masked features R2={r2_masked:.4f}) [{time.time() - t:.1f}s]")
        return self

    def encode(self, X: pd.DataFrame) -> np.ndarray:
        """Forward pass up to the bottleneck layer."""
        h = self.prep(X)
        n_enc = len(self.hidden) // 2 + 1  # layers up to and including the bottleneck
        for W, b in zip(self.net.coefs_[:n_enc], self.net.intercepts_[:n_enc], strict=True):
            h = np.tanh(h @ W + b)
        return h.astype(np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    gn = pd.read_parquet(PROC / "unlabeled_gnome_features.parquet")
    print(f"pre-training on {len(gn):,} unlabeled GNoME compositions", flush=True)
    # import the class through its module name so the pickle is loadable from the API
    from self_supervised import MaskedAutoencoder as ImportableMAE

    ae = ImportableMAE().fit(gn, log=lambda m: print(m, flush=True))
    joblib.dump(ae, MODELS / "self_supervised_autoencoder.pkl")
    pd.DataFrame(ae.history).to_csv(OUT / "autoencoder_training_history.csv", index=False)
    (OUT / "autoencoder_summary.json").write_text(json.dumps({
        "pretext_task": "masked feature reconstruction (denoising autoencoder)",
        "n_unlabeled_samples": len(gn), "architecture": [len(ae.cols), *ae.hidden, len(ae.cols)],
        "embedding_dim": ae.hidden[len(ae.hidden) // 2], "mask_rate": ae.mask_rate,
        "epochs": ae.epochs, "final": ae.history[-1]}, indent=2))
    print("saved", MODELS / "self_supervised_autoencoder.pkl")


if __name__ == "__main__":
    main()
