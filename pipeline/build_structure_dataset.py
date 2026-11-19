"""Per-structure datasets with crystal-structure descriptors (high-accuracy track).

One row per crystal structure (not per formula), so every label belongs to exactly one
structure:
  metallicity  Matbench mp_is_metal          is_metal
  band_gap     Matbench mp_gap               1.1 <= PBE gap <= 1.7 eV
  magnetism    JARVIS-DFT 3D                 |magmom_outcar| >= 0.5 mu_B
  stability    Alexandria PBE 3D (5 chunks)  e_above_hull <= 0.05 eV/atom

The raw JSON files are parsed as a stream (one entry at a time) to keep memory low, and the
structure descriptors (pipeline/structure_features.py) are computed in parallel. Energies,
moments and gaps are only used to build the label, never as features.

Usage:  uv run python pipeline/build_structure_dataset.py [--tasks metallicity band_gap ...]
Output: data/processed/{task}_struct.parquet  (formula, target, struct_* columns)
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from featurize import reduced_formula  # noqa: E402
from structure_features import (  # noqa: E402
    STRUCT_COLUMNS,
    arrays_from_jarvis_atoms,
    arrays_from_pymatgen_dict,
    default_tables,
    featurize_batch,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
BATCH = 2000
_WS = re.compile(r"[\s,]*")


def iter_json_array(text_stream, key: str | None, chunk: int = 1 << 20):
    """Yield the elements of a (possibly huge) JSON array one by one.

    key=None: the document itself is the array; otherwise the array stored under "key".
    """
    dec = json.JSONDecoder()
    buf = text_stream.read(chunk)
    if key is None:
        pos = buf.index("[") + 1
    else:
        while f'"{key}"' not in buf:
            more = text_stream.read(chunk)
            if not more:
                return
            buf = buf[-len(key) - 4:] + more
        pos = buf.index("[", buf.index(f'"{key}"')) + 1
    while True:
        pos = _WS.match(buf, pos).end()
        if pos >= len(buf) - 1:
            more = text_stream.read(chunk)
            if not more:
                return
            buf, pos = buf[pos:] + more, 0
            continue
        if buf[pos] == "]":
            return
        try:
            obj, end = dec.raw_decode(buf, pos)
        except json.JSONDecodeError:
            more = text_stream.read(chunk)
            if not more:
                return
            buf, pos = buf[pos:] + more, 0
            continue
        yield obj
        pos = end
        if pos > (8 << 20):
            buf, pos = buf[pos:], 0


# ----------------------------------------------------------------------------- sources
def matbench_items(name: str, label_fn, sym2z):
    path = RAW / "matbench" / "download" / f"{name}.json.gz"
    with gzip.open(path, "rt") as f:
        for i, (sdict, y) in enumerate(iter_json_array(f, "data")):
            lab = label_fn(y)
            if lab is None:
                continue
            lat, frac, z = arrays_from_pymatgen_dict(sdict, sym2z)
            yield (i, lab, lat, frac, z)


def jarvis_items(sym2z):
    path = RAW / "jarvis" / "download" / "jdft_3d-9-24-2025.json.zip"
    with zipfile.ZipFile(path) as zf:
        member = zf.namelist()[0]
        with zf.open(member) as raw:
            import io
            text = io.TextIOWrapper(raw, encoding="utf-8")
            for i, rec in enumerate(iter_json_array(text, None)):
                m = rec.get("magmom_outcar")
                if not isinstance(m, (int, float)):
                    continue
                lat, frac, z = arrays_from_jarvis_atoms(rec["atoms"], sym2z)
                yield (i, int(abs(m) >= 0.5), lat, frac, z)


def alexandria_items(sym2z):
    k = 0
    for path in sorted((RAW / "alexandria").glob("alexandria_*.json.bz2")):
        print("   reading", path.name, flush=True)
        with bz2.open(path, "rt") as f:
            for e in iter_json_array(f, "entries"):
                eh = e["data"].get("e_above_hull")
                if eh is None:
                    continue
                lat, frac, z = arrays_from_pymatgen_dict(e["structure"], sym2z)
                yield (k, int(eh <= 0.05), lat, frac, z)
                k += 1


SOURCES = {
    "metallicity": lambda s: matbench_items(
        "matbench_mp_is_metal", lambda y: int(bool(y)), s),
    "band_gap": lambda s: matbench_items(
        "matbench_mp_gap", lambda y: None if y is None else int(1.1 <= y <= 1.7), s),
    "magnetism": jarvis_items,
    "stability": alexandria_items,
}


# ----------------------------------------------------------------------------- build
def build(task: str, tables: dict, n_jobs: int) -> None:
    t0 = time.time()
    sym2z = tables["sym2z"]
    z2sym = {v: k for k, v in sym2z.items()}
    labels, formulas = {}, {}

    def batches():
        batch = []
        for key, lab, lat, frac, z in SOURCES[task](sym2z):
            labels[key] = lab
            counts: dict[str, int] = {}
            for zi in z:
                counts[z2sym.get(zi, "X")] = counts.get(z2sym.get(zi, "X"), 0) + 1
            formulas[key] = "".join(f"{el}{c}" for el, c in counts.items())
            batch.append((key, lat, frac, z))
            if len(batch) >= BATCH:
                yield batch
                batch = []
        if batch:
            yield batch

    rows = []
    gen = Parallel(n_jobs=n_jobs, return_as="generator", pre_dispatch="2*n_jobs")(
        delayed(featurize_batch)(b, tables) for b in batches())
    for res in gen:
        rows.extend((k, f) for k, f in res if f is not None)
        if len(rows) % 50000 < BATCH:
            print(f"   {task}: {len(rows):,} structures ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame([f for _, f in rows], columns=STRUCT_COLUMNS).astype("float32")
    keys = [k for k, _ in rows]
    raw_formula = pd.Series([formulas[k] for k in keys])
    uniq = {f: reduced_formula(f) for f in raw_formula.unique()}
    df.insert(0, "formula", raw_formula.map(uniq).values)
    df.insert(1, "target", [labels[k] for k in keys])
    df = df[df["formula"].notna()]
    n0 = len(df)
    # near-identical duplicate structures: same formula, symmetry, cell size and volume
    df = df.assign(_v=df["struct_vpa"].round(2)).drop_duplicates(
        ["formula", "struct_spg", "struct_nprim", "_v"]).drop(columns="_v")
    df = df.reset_index(drop=True)
    out = PROC / f"{task}_struct.parquet"
    df.to_parquet(out)
    print(f"[done] {task}: {len(df):,} structures ({n0 - len(df):,} duplicates removed), "
          f"{df['formula'].nunique():,} formulas, positive rate {df['target'].mean():.3f}, "
          f"{time.time() - t0:.0f}s -> {out.name}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["metallicity", "band_gap", "magnetism",
                                                    "stability"])
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()
    tables = default_tables()
    for task in args.tasks:
        print(f"== {task}", flush=True)
        build(task, tables, args.n_jobs)


if __name__ == "__main__":
    main()
