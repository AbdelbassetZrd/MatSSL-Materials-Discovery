"""Build the 'nearest known materials' index used by the web application.

Every labeled composition of the Materials Project (metallicity, band gap) and JARVIS
(magnetism) datasets is embedded with the self-supervised masked autoencoder (32-d) and
stored with all its known labels (Alexandria stability added when the composition exists
there). Also stores, for the 'why this prediction' panel, per-class percentiles of key
descriptors for every task.

Output: app/models/neighbor_index.joblib  (~10 MB)
Usage:  uv run python pipeline/build_neighbor_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from self_supervised import MaskedAutoencoder  # noqa: E402

import __main__  # noqa: E402

__main__.MaskedAutoencoder = MaskedAutoencoder

ROOT = Path(__file__).resolve().parents[1]
PROC, MODELS = ROOT / "data" / "processed", ROOT / "app" / "models"
KEY = ["MagpieData mean Electronegativity", "MagpieData range Electronegativity",
       "MagpieData mean CovalentRadius", "MagpieData mean NValence",
       "MagpieData mean NdValence", "MagpieData mean NUnfilled", "MagpieData mean GSmagmom",
       "MagpieData mean GSbandgap", "MagpieData mean MeltingT", "MagpieData mean Column",
       "0-norm", "frac d valence electrons"]


def main() -> None:
    ae = joblib.load(MODELS / "self_supervised_autoencoder.pkl")
    met = pd.read_parquet(PROC / "metallicity_features.parquet")
    mag = pd.read_parquet(PROC / "magnetism_features.parquet")
    gap = pd.read_parquet(PROC / "band_gap_features.parquet")
    stab = pd.read_parquet(PROC / "stability_features.parquet",
                           columns=["formula", "target", "e_above_hull_min"])

    feat_cols = [c for c in met.columns if c.startswith(("MagpieData", "frac ")) or
                 c.endswith("-norm")]
    union = pd.concat([met[["formula"] + feat_cols], mag[["formula"] + feat_cols],
                       gap[["formula"] + feat_cols]]).drop_duplicates("formula")
    union = union.reset_index(drop=True)
    labels = pd.DataFrame({"formula": union["formula"]})
    labels = labels.merge(met[["formula", "target"]].rename(columns={"target": "is_metal"}),
                          how="left", on="formula")
    labels = labels.merge(mag[["formula", "target", "magmom"]].rename(
        columns={"target": "is_magnetic"}), how="left", on="formula")
    labels = labels.merge(gap[["formula", "gap_pbe"]], how="left", on="formula")
    labels = labels.merge(stab.rename(columns={"target": "is_stable",
                                               "e_above_hull_min": "e_above_hull"}),
                          how="left", on="formula")
    emb = ae.encode(union[feat_cols]).astype(np.float16)
    print(f"index: {len(union):,} labeled compositions, embedding {emb.shape[1]}-d")

    stats = {}
    for task, df in (("metallicity", met), ("magnetism", mag), ("band_gap", gap)):
        stats[task] = {}
        for c in KEY:
            q = {}
            for y in (0, 1):
                v = df.loc[df["target"] == y, c].dropna().to_numpy()
                q[y] = np.quantile(v, np.linspace(0, 1, 101)).astype(np.float32)
            stats[task][c] = q
    stab_full = pd.read_parquet(PROC / "stability_features.parquet", columns=["target"] + KEY)
    stats["stability"] = {c: {y: np.quantile(stab_full.loc[stab_full["target"] == y, c]
                                             .dropna(), np.linspace(0, 1, 101))
                              .astype(np.float32) for y in (0, 1)} for c in KEY}

    joblib.dump({"formula": union["formula"].to_numpy(), "embedding": emb,
                 "labels": labels.drop(columns="formula").to_dict(orient="list"),
                 "percentiles": stats, "key_descriptors": KEY},
                MODELS / "neighbor_index.joblib", compress=("xz", 3))
    size = (MODELS / "neighbor_index.joblib").stat().st_size / 1e6
    print(f"saved neighbor_index.joblib ({size:.1f} MB)")


if __name__ == "__main__":
    main()
