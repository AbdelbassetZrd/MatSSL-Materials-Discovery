"""Download JARVIS-DFT 3D and Matbench (Materials Project) datasets and keep the
columns we need (formula + labels) as compact Parquet files in data/raw/.

Usage:  uv run python pipeline/download_jarvis_matbench.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def jarvis() -> None:
    out = RAW / "jarvis" / "jarvis_dft_3d.parquet"
    if out.exists():
        print("[skip] JARVIS")
        return
    from jarvis.db.figshare import data

    store = RAW / "jarvis" / "download"
    store.mkdir(parents=True, exist_ok=True)
    d = data("dft_3d", store_dir=str(store))
    df = pd.DataFrame(d)
    print("JARVIS dft_3d entries:", len(df))
    print("JARVIS keys:", list(df.columns))
    keep = [c for c in ["jid", "formula", "formation_energy_peratom", "ehull",
                        "optb88vdw_bandgap", "mbj_bandgap", "magmom_oszicar",
                        "magmom_outcar", "spg_number", "crys"] if c in df.columns]
    df = df[keep].replace("na", pd.NA)
    df.to_parquet(out)
    print("[done] JARVIS ->", out)


def matbench(name: str, target: str) -> None:
    out = RAW / "matbench" / f"{name}.parquet"
    if out.exists():
        print(f"[skip] {name}")
        return
    from matminer.datasets import load_dataset

    home = RAW / "matbench" / "download"
    home.mkdir(parents=True, exist_ok=True)
    df = load_dataset(name, data_home=str(home))
    print(f"{name}: {len(df)} entries")
    comp = df["structure"].map(lambda s: s.composition)
    small = pd.DataFrame({
        "formula": comp.map(lambda c: c.reduced_formula),
        "nsites": df["structure"].map(len),
        "energy_proxy_volume_pa": df["structure"].map(lambda s: s.volume / len(s)),
        target: df[target].values,
    })
    small.to_parquet(out)
    print(f"[done] {name} ->", out)


if __name__ == "__main__":
    jarvis()
    matbench("matbench_mp_is_metal", "is_metal")
    matbench("matbench_mp_gap", "gap pbe")
