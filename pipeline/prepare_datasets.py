"""Data-enhancing step: raw databases -> one labeled row per unique composition
-> 142 composition features (Magpie + stoichiometry + valence orbitals).

Tasks (binary classification):
  metallicity  Materials Project (matbench_mp_is_metal)  is_metal
  magnetism    JARVIS-DFT 3D                              |magmom| >= 0.5 muB (ground-state polymorph)
  stability    Alexandria PBE 3D                          min E_above_hull <= STABILITY_CUTOFF eV/atom
  band_gap     Materials Project (matbench_mp_gap)        1.1 <= Eg <= 1.7 eV (photovoltaic window)
Unlabeled pool: GNoME stable-materials summary (compositions, labels never used).

Usage:  uv run python pipeline/prepare_datasets.py [task ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from featurize import featurize_formulas, reduced_formula  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
STABILITY_CUTOFF = 0.05   # eV/atom: stable or near-hull (synthesizable) compositions
MAGNETIC_CUTOFF = 0.5     # Bohr magnetons per cell
PV_WINDOW = (1.1, 1.7)    # eV


def _reduce(series: pd.Series) -> pd.Series:
    uniq = pd.Series(series.dropna().unique())
    mapping = dict(zip(uniq, uniq.map(reduced_formula), strict=True))
    return series.map(mapping)


def labeled_metallicity() -> pd.DataFrame:
    df = pd.read_parquet(RAW / "matbench" / "matbench_mp_is_metal.parquet")
    df["formula"] = _reduce(df["formula"])
    g = df.groupby("formula")["is_metal"].agg(["mean", "size"]).reset_index()
    g = g[g["mean"] != 0.5]  # polymorphs disagree 50/50 -> ambiguous, drop
    return pd.DataFrame({"formula": g["formula"], "target": (g["mean"] > 0.5).astype(int),
                         "n_polymorphs": g["size"]})


def labeled_magnetism() -> pd.DataFrame:
    df = pd.read_parquet(RAW / "jarvis" / "jarvis_dft_3d.parquet")
    mag_col = "magmom_outcar" if "magmom_outcar" in df else "magmom_oszicar"
    df["mag"] = pd.to_numeric(df[mag_col], errors="coerce")
    df["ef"] = pd.to_numeric(df["formation_energy_peratom"], errors="coerce")
    df = df.dropna(subset=["mag", "ef"])
    df["formula"] = _reduce(df["formula"])
    df = df.sort_values("ef").groupby("formula", as_index=False).first()  # ground state
    return pd.DataFrame({"formula": df["formula"],
                         "target": (df["mag"].abs() >= MAGNETIC_CUTOFF).astype(int),
                         "magmom": df["mag"]})


def labeled_stability() -> pd.DataFrame:
    df = pd.read_parquet(RAW / "alexandria" / "alexandria_pbe3d.parquet")
    df = df.dropna(subset=["e_above_hull"])
    df["formula"] = _reduce(df["formula"])
    g = df.groupby("formula")["e_above_hull"].agg(["min", "size"]).reset_index()
    return pd.DataFrame({"formula": g["formula"],
                         "target": (g["min"] <= STABILITY_CUTOFF).astype(int),
                         "e_above_hull_min": g["min"], "n_polymorphs": g["size"]})


def labeled_band_gap() -> pd.DataFrame:
    df = pd.read_parquet(RAW / "matbench" / "matbench_mp_gap.parquet")
    df["formula"] = _reduce(df["formula"])
    g = df.groupby("formula")["gap pbe"].median().reset_index()
    lo, hi = PV_WINDOW
    return pd.DataFrame({"formula": g["formula"],
                         "target": g["gap pbe"].between(lo, hi).astype(int),
                         "gap_pbe": g["gap pbe"]})


def unlabeled_gnome() -> pd.DataFrame:
    df = pd.read_csv(RAW / "gnome" / "stable_materials_summary.csv",
                     usecols=["Reduced Formula", "NSites"])
    df["formula"] = _reduce(df["Reduced Formula"])
    return df.dropna(subset=["formula"]).drop_duplicates("formula")[["formula"]]


TASKS = {
    "metallicity": labeled_metallicity,
    "magnetism": labeled_magnetism,
    "stability": labeled_stability,
    "band_gap": labeled_band_gap,
    "unlabeled_gnome": unlabeled_gnome,
}


def build(task: str) -> dict:
    print(f"=== {task} ===")
    meta_df = TASKS[task]().dropna(subset=["formula"]).reset_index(drop=True)
    feats, ok = featurize_formulas(meta_df["formula"])
    meta_df = meta_df[ok.values].reset_index(drop=True)
    feats = feats.reset_index(drop=True)
    out = pd.concat([meta_df, feats], axis=1)
    PROC.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROC / f"{task}_features.parquet")
    info = {"task": task, "n_rows": len(out), "n_features": feats.shape[1]}
    if "target" in out:
        info["positive_rate"] = float(out["target"].mean())
        info["class_counts"] = out["target"].value_counts().to_dict()
    print(json.dumps(info, default=str))
    return info


if __name__ == "__main__":
    for t in (sys.argv[1:] or list(TASKS)):
        build(t)
