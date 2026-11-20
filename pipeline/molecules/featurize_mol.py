"""Molecular featuriser -- the molecular counterpart of pipeline/featurize.py.

Every molecule becomes
  * 217 RDKit physico-chemical descriptors (weight, logP, TPSA, ring counts, graph indices, ...)
  * a 2048-bit Morgan fingerprint (radius 2), stored packed as 256 uint8 columns
  * its Bemis-Murcko scaffold, used to group the train/test split exactly as the reduced formula
    groups the materials split -- two molecules sharing a scaffold are never on both sides.

The two feature blocks are also the two *views* used by Co-Training: descriptors describe the
whole molecule, the fingerprint describes its substructures.

Usage:  uv run python pipeline/molecules/featurize_mol.py [--unlabeled 150000]
Output: data/processed/molecules_labeled_feat.parquet, molecules_unlabeled_feat.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
N_BITS = 2048
RADIUS = 2
BATCH = 2000
DESC_NAMES = [n for n, _ in Descriptors._descList]
FP_COLS = [f"fp{i}" for i in range(N_BITS // 8)]


def descriptor_names() -> list[str]:
    return DESC_NAMES


def feature_names() -> list[str]:
    """Column order of the full feature matrix (descriptors first, then fingerprint bits)."""
    return DESC_NAMES + [f"bit{i}" for i in range(N_BITS)]


def view_columns() -> tuple[list[int], list[int]]:
    """Co-Training views: (descriptor indices, fingerprint indices)."""
    n = len(DESC_NAMES)
    return list(range(n)), list(range(n, n + N_BITS))


def scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold; falls back to the molecule itself for acyclic compounds."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:  # noqa: BLE001
        return smiles
    return s or smiles


def _batch(smiles: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    gen = GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    desc = np.full((len(smiles), len(DESC_NAMES)), np.nan, dtype=np.float64)
    fp = np.zeros((len(smiles), N_BITS // 8), dtype=np.uint8)
    scafs = []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            scafs.append("")
            continue
        try:
            d = Descriptors.CalcMolDescriptors(mol)
            desc[i] = [d.get(n, np.nan) for n in DESC_NAMES]
        except Exception:  # noqa: BLE001
            pass
        fp[i] = np.packbits(np.asarray(gen.GetFingerprint(mol), dtype=np.uint8))
        scafs.append(scaffold(s))
    return desc, fp, scafs


def featurize_smiles(smiles: list[str], n_jobs: int = 8, quiet: bool = False) -> pd.DataFrame:
    """Descriptors + packed fingerprint + scaffold for a list of SMILES (keeps the input order)."""
    t0 = time.time()
    chunks = [smiles[i:i + BATCH] for i in range(0, len(smiles), BATCH)]
    # RDKit objects cannot be pickled, so the worker function must be sent *by reference*:
    # that only works when this file is imported as a module, not run as __main__.
    import featurize_mol as _self

    out = Parallel(n_jobs=n_jobs, batch_size=1)(delayed(_self._batch)(c) for c in chunks)
    desc = np.vstack([o[0] for o in out])
    fp = np.vstack([o[1] for o in out])
    scafs = [s for o in out for s in o[2]]
    # Ipc and a few graph indices explode on large molecules: they overflow float64 outright,
    # and values that are merely huge become +inf once cast to float32. Both become NaN, which
    # the Preprocessor then imputes.
    desc[~np.isfinite(desc)] = np.nan
    desc = np.clip(desc, -3e38, 3e38)
    desc32 = desc.astype(np.float32)
    desc32[~np.isfinite(desc32)] = np.nan
    df = pd.concat([
        pd.DataFrame({"smiles": smiles, "scaffold": scafs}),
        pd.DataFrame(desc32, columns=DESC_NAMES),
        pd.DataFrame(fp, columns=FP_COLS),
    ], axis=1)
    if not quiet:
        print(f"   featurised {len(smiles):,} molecules in {time.time() - t0:.0f}s")
    return df


def unpack_fingerprints(df: pd.DataFrame) -> np.ndarray:
    """256 packed uint8 columns -> (n, 2048) float32 bit matrix."""
    return np.unpackbits(df[FP_COLS].to_numpy(np.uint8), axis=1).astype(np.float32)


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Full (n, 217 + 2048) matrix in the order given by feature_names()."""
    return np.hstack([df[DESC_NAMES].to_numpy(np.float32), unpack_fingerprints(df)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unlabeled", type=int, default=150_000,
                    help="size of the ZINC pool to featurise (0 = all)")
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()

    lab = pd.read_parquet(PROC / "molecules_labeled.parquet")
    print(f"labeled: {len(lab):,} molecules")
    feat = featurize_smiles(lab["smiles"].tolist(), args.n_jobs)
    label_cols = [c for c in lab.columns if c != "smiles"]
    feat = feat.join(lab.set_index("smiles")[label_cols], on="smiles")
    feat.to_parquet(PROC / "molecules_labeled_feat.parquet", index=False)
    print(f"   -> molecules_labeled_feat.parquet  {feat['scaffold'].nunique():,} scaffolds")

    unl = pd.read_parquet(PROC / "molecules_unlabeled.parquet")
    if args.unlabeled and len(unl) > args.unlabeled:
        unl = unl.sample(args.unlabeled, random_state=0).reset_index(drop=True)
    print(f"unlabeled pool: {len(unl):,} ZINC molecules")
    fu = featurize_smiles(unl["smiles"].tolist(), args.n_jobs)
    fu.to_parquet(PROC / "molecules_unlabeled_feat.parquet", index=False)
    size = (PROC / "molecules_unlabeled_feat.parquet").stat().st_size / 1e6
    print(f"   -> molecules_unlabeled_feat.parquet ({size:.0f} MB)")


if __name__ == "__main__":
    main()
