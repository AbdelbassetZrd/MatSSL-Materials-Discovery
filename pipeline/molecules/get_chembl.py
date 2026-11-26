"""Download measured bioactivities from ChEMBL for every target of the knowledge base.

Labels come from pChEMBL, the standardised -log10 of a potency measurement (IC50, Ki, EC50...):

    pChEMBL >= 6.5   active    (<= ~300 nM)
    pChEMBL <  5.0   inactive  (>= 10 uM)
    in between       dropped   (too ambiguous to label either way)

There is no agreed cutoff in the literature and the active/inactive balance moves a lot with
it, so the values used are written into every target file and can be changed from the command
line. Roughly 90 % of ChEMBL records would count as "active" at a cutoff of 5.

ChEMBL is biased towards published actives, so some targets have almost no measured inactives.
Where that happens, property-matched ZINC molecules (similar weight and logP, never a known
active) are added as assumed inactives, and the proportion is recorded per target so the
interface and the report can show it.

Usage:  uv run python pipeline/molecules/get_chembl.py [--max-activities 20000]
Output: data/processed/targets/{chembl_id}.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
PROC = ROOT / "data" / "processed"
OUT = PROC / "targets"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
PAGE = 1000


def canonical(s: str | None) -> str | None:
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToSmiles(m) if m is not None else None


def get_json(client: httpx.Client, url: str, params: dict, tries: int = 3) -> dict | None:
    """One GET with retries: pooled connections go stale on a flaky link and then hang."""
    for attempt in range(tries):
        try:
            r = client.get(url, params=params, timeout=45)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 502, 503, 504):
                time.sleep(2 * (attempt + 1))
                continue
            return None
        except Exception:  # noqa: BLE001 - timeouts and resets are both just retried
            time.sleep(2 * (attempt + 1))
    return None


def fetch_activities(client: httpx.Client, target: str, cap: int) -> pd.DataFrame:
    rows, offset = [], 0
    while offset < cap:
        j = get_json(client, f"{CHEMBL}/activity.json", {
            "target_chembl_id": target, "pchembl_value__isnull": "false",
            "limit": PAGE, "offset": offset})
        if j is None:
            print(f"      giving up on {target} at offset {offset}", flush=True)
            break
        acts = j.get("activities", [])
        if not acts:
            break
        rows += [{"molecule_chembl_id": a.get("molecule_chembl_id"),
                  "smiles": a.get("canonical_smiles"),
                  "pchembl": float(a["pchembl_value"]),
                  "assay_type": a.get("standard_type")}
                 for a in acts if a.get("pchembl_value") and a.get("canonical_smiles")]
        offset += PAGE
        if len(acts) < PAGE:
            break
    return pd.DataFrame(rows)


def decoys(pool: pd.DataFrame, actives: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """ZINC molecules whose weight and logP resemble the actives: assumed inactive."""
    if n <= 0 or pool.empty:
        return pd.DataFrame(columns=["smiles", "pchembl", "label", "decoy"])
    lo_w, hi_w = actives["mw"].quantile([0.05, 0.95])
    lo_p, hi_p = actives["logp"].quantile([0.05, 0.95])
    match = pool[(pool.MolWt.between(lo_w - 50, hi_w + 50))
                 & (pool.MolLogP.between(lo_p - 1, hi_p + 1))]
    if len(match) < n:
        match = pool
    take = match.sample(min(n, len(match)), random_state=seed)
    return pd.DataFrame({"smiles": take["smiles"].to_numpy(), "pchembl": pd.NA,
                         "label": 0, "decoy": True})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--active", type=float, default=6.5)
    ap.add_argument("--inactive", type=float, default=5.0)
    ap.add_argument("--max-activities", type=int, default=20000)
    ap.add_argument("--min-molecules", type=int, default=150)
    ap.add_argument("--min-inactive-ratio", type=float, default=0.3,
                    help="top up with ZINC decoys below this share of inactives")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    kb = json.loads((PROC / "diseases.json").read_text(encoding="utf-8"))
    pool = pd.read_parquet(PROC / "molecules_unlabeled_feat.parquet",
                           columns=["smiles", "MolWt", "MolLogP"])
    known = {s for d in kb["diseases"] for s in [x["smiles"] for x in d["known_drugs"]]}
    pool = pool[~pool["smiles"].isin(known)]
    t0, summary = time.time(), []

    with httpx.Client(headers={"accept": "application/json"}) as client:
        for i, tgt in enumerate(kb["targets"], 1):
            cid, sym = tgt["chembl_id"], tgt["symbol"]
            f = OUT / f"{cid}.parquet"
            if f.exists():
                print(f"[{i:>2}/{len(kb['targets'])}] {sym:<10} cached", flush=True)
                continue
            raw = fetch_activities(client, cid, args.max_activities)
            if raw.empty:
                print(f"[{i:>2}/{len(kb['targets'])}] {sym:<10} no activities", flush=True)
                continue
            raw["smiles"] = raw["smiles"].map(canonical)
            raw = raw.dropna(subset=["smiles"])
            # one row per molecule: the median of all its measurements against this target
            g = raw.groupby("smiles")["pchembl"].median().reset_index()
            g["label"] = pd.NA
            g.loc[g.pchembl >= args.active, "label"] = 1
            g.loc[g.pchembl < args.inactive, "label"] = 0
            g = g.dropna(subset=["label"])
            g["label"] = g["label"].astype(int)
            g["decoy"] = False

            n_act = int(g.label.sum())
            n_inact = int((g.label == 0).sum())
            if n_act < args.min_molecules:
                print(f"[{i:>2}/{len(kb['targets'])}] {sym:<10} only {n_act} actives - skipped", flush=True)
                continue
            need = int(n_act * args.min_inactive_ratio / (1 - args.min_inactive_ratio)) - n_inact
            if need > 0:
                props = raw.merge(g[["smiles"]], on="smiles").drop_duplicates("smiles")
                mols = props["smiles"].map(Chem.MolFromSmiles)
                from rdkit.Chem import Crippen, Descriptors
                act_props = pd.DataFrame({
                    "mw": [Descriptors.MolWt(m) for m in mols],
                    "logp": [Crippen.MolLogP(m) for m in mols]})
                extra = decoys(pool, act_props, need, seed=i)
                if not extra.empty:
                    g = pd.concat([g, extra], ignore_index=True)

            g = g.drop_duplicates("smiles").reset_index(drop=True)
            g.attrs = {}
            g.to_parquet(f, index=False)
            frac = float(g["decoy"].mean())
            summary.append({"chembl_id": cid, "symbol": sym, "n": len(g),
                            "positive_rate": float(g.label.mean()), "decoy_fraction": frac,
                            "n_measured_inactive": n_inact})
            print(f"[{i:>2}/{len(kb['targets'])}] {sym:<10} {len(g):>6,} molecules  "
                  f"{g.label.mean():.3f} active  {frac:.0%} decoys  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    meta = {"active_cutoff": args.active, "inactive_cutoff": args.inactive,
            "min_molecules": args.min_molecules, "targets": summary}
    (OUT / "_summary.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"\n{len(summary)} targets written to {OUT} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
