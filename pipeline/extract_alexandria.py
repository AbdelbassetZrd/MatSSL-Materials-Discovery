"""Extract the property table from the downloaded Alexandria PBE 3D chunks.

Keeps only (mat_id, formula, spg, nsites, e_above_hull, e_form, band_gap_ind,
total_mag) so that the multi-GB JSON is reduced to a small Parquet file.

Usage:  uv run python pipeline/extract_alexandria.py
"""

from __future__ import annotations

import bz2
import gc
import json
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "alexandria"
FIELDS = ["mat_id", "formula", "spg", "nsites", "e_above_hull", "e_form",
          "band_gap_ind", "total_mag"]


def main() -> None:
    parts = []
    for path in sorted(RAW.glob("alexandria_*.json.bz2")):
        out = path.with_suffix("").with_suffix(".parquet")
        if not out.exists():
            print("reading", path.name)
            with bz2.open(path, "rt") as f:
                entries = json.load(f)["entries"]
            rows = [{k: e["data"].get(k) for k in FIELDS} for e in entries]
            del entries
            gc.collect()
            pd.DataFrame(rows).to_parquet(out)
            print("  ->", out.name, len(rows), "entries")
        parts.append(pd.read_parquet(out))
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(RAW / "alexandria_pbe3d.parquet")
    print("total Alexandria entries:", len(df), "| unique formulas:", df["formula"].nunique())


if __name__ == "__main__":
    main()
