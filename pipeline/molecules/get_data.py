"""Prepare the molecular datasets (toxicity track).

Labeled data
  Tox21    : 12 nuclear-receptor / stress-response assays, ~7,800 molecules
  ClinTox  : CT_TOX (failed clinical trials for toxicity) and FDA_APPROVED, ~1,480 molecules
Unlabeled pool
  ZINC     : ~499,000 drug-like molecules -- the molecular counterpart of the GNoME pool
             used in the materials study.

Every SMILES is canonicalised with RDKit and molecules that do not parse are dropped, so the
same molecule written in two ways cannot end up in both the train and the test split.

Usage:  uv run python pipeline/molecules/get_data.py
Output: data/processed/molecules_labeled.parquet, molecules_unlabeled.parquet
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "molecules"
PROC = ROOT / "data" / "processed"
# the datasets ship with the QSAR project of the same course; copied in on first run
SOURCES = [
    ROOT.parent / "mini_projet-main" / "mini_projet-main" / "data" / "raw" / "original_data",
    ROOT / "data" / "raw" / "original_data",
]
TOX21_ASSAYS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
                "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
ENDPOINTS = [*TOX21_ASSAYS, "CT_TOX", "FDA_APPROVED"]


def fetch() -> None:
    """Copy the three CSVs into data/raw/molecules/ (they are not redistributed in git)."""
    RAW.mkdir(parents=True, exist_ok=True)
    for name in ("tox21.csv", "clintox.csv", "zinc_unlabeled.csv"):
        dst = RAW / name
        if dst.exists():
            continue
        src = next((d / name for d in SOURCES if (d / name).exists()), None)
        if src is None:
            sys.exit(f"missing {name}: put it in {RAW} (MoleculeNet Tox21/ClinTox, ZINC250k)")
        shutil.copy(src, dst)
        print(f"copied {name} ({dst.stat().st_size / 1e6:.1f} MB)")


def canonical(smiles: pd.Series) -> pd.Series:
    """Canonical SMILES, or NA when RDKit cannot parse the molecule."""
    def one(s):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        return Chem.MolToSmiles(m) if m is not None else None
    return smiles.map(one)


def main() -> None:
    fetch()
    PROC.mkdir(parents=True, exist_ok=True)

    tox = pd.read_csv(RAW / "tox21.csv")
    tox["smiles"] = canonical(tox["smiles"])
    tox = tox.dropna(subset=["smiles"])
    tox = tox.groupby("smiles", as_index=False)[TOX21_ASSAYS].max()  # duplicates: toxic wins
    print(f"Tox21    : {len(tox):,} molecules")

    ct = pd.read_csv(RAW / "clintox.csv")
    ct["smiles"] = canonical(ct["smiles"])
    ct = ct.dropna(subset=["smiles"]).groupby("smiles", as_index=False)[
        ["CT_TOX", "FDA_APPROVED"]].max()
    print(f"ClinTox  : {len(ct):,} molecules")

    lab = tox.merge(ct, on="smiles", how="outer")
    lab.to_parquet(PROC / "molecules_labeled.parquet", index=False)
    counts = {e: int(lab[e].notna().sum()) for e in ENDPOINTS}
    rates = {e: float(lab[e].mean()) for e in ENDPOINTS}
    print(f"labeled  : {len(lab):,} unique molecules, {len(ENDPOINTS)} endpoints")
    for e in ENDPOINTS:
        print(f"   {e:<14} {counts[e]:>6,} labels   positive rate {rates[e]:.3f}")

    zinc = pd.read_csv(RAW / "zinc_unlabeled.csv")
    zinc["smiles"] = canonical(zinc["smiles"].str.strip().str.strip('"'))
    zinc = zinc.dropna(subset=["smiles"]).drop_duplicates("smiles")
    zinc = zinc[~zinc["smiles"].isin(set(lab["smiles"]))]  # no labeled molecule in the pool
    zinc[["smiles"]].to_parquet(PROC / "molecules_unlabeled.parquet", index=False)
    print(f"unlabeled: {len(zinc):,} ZINC molecules (labeled ones removed)")


if __name__ == "__main__":
    main()
