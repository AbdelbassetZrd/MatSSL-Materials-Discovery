"""Build the illness -> protein target -> measured activity knowledge base.

No model predicts "cures a disease". What a model can learn from measured data is whether a
molecule is **active against a protein target**. Illnesses enter through public evidence that
links them to proteins, so the chain is

    molecule --(QSAR model per target)--> target activities --(association scores)--> illnesses

This script builds the right-hand side of that chain:
  * Open Targets gives, per illness, the proteins associated with it and how strong the
    evidence is (0-1 association score);
  * ChEMBL says how much measured activity data each of those proteins actually has, so
    targets nobody has screened are dropped before any training is attempted;
  * the approved small-molecule drugs of each illness are kept as a validation set - a working
    pipeline should rank an illness highly for the drugs already used against it.

Usage:  uv run python pipeline/molecules/build_disease_graph.py
Output: data/processed/diseases.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "diseases.json"
OPEN_TARGETS = "https://api.platform.opentargets.org/api/v4/graphql"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"

# 15 well-studied illnesses with druggable protein targets and plenty of public data
DISEASES = [
    ("MONDO_0005148", "Type 2 diabetes"),
    ("MONDO_0004975", "Alzheimer's disease"),
    ("MONDO_0005180", "Parkinson's disease"),
    ("MONDO_0007254", "Breast cancer"),
    ("MONDO_0008903", "Lung cancer"),
    ("MONDO_0005044", "Hypertension"),
    ("MONDO_0004979", "Asthma"),
    ("MONDO_0018076", "Tuberculosis"),
    ("MONDO_0005136", "Malaria"),
    ("MONDO_0005109", "HIV infection"),
    ("MONDO_0002050", "Depression"),
    ("MONDO_0005027", "Epilepsy"),
    ("MONDO_0008383", "Rheumatoid arthritis"),
    ("MONDO_0011122", "Obesity"),
    ("MONDO_0100096", "COVID-19"),
]

DISEASE_QUERY = """query($efo:String!,$n:Int!){
  disease(efoId:$efo){ id name description
    associatedTargets(page:{index:0,size:$n}){
      rows{ score target{ id approvedSymbol approvedName } } }
    drugAndClinicalCandidates{
      rows{ maxClinicalStage drug{ id name drugType } } } } }"""


def graphql(client: httpx.Client, variables: dict) -> dict | None:
    r = client.post(OPEN_TARGETS, json={"query": DISEASE_QUERY, "variables": variables},
                    timeout=120)
    body = r.json()
    if "errors" in body:
        print("   open targets error:", body["errors"][0]["message"][:120])
        return None
    return body["data"]["disease"]


def chembl_target(client: httpx.Client, symbol: str, cache: dict) -> dict | None:
    """Gene symbol -> single human protein in ChEMBL, with its number of measured activities."""
    if symbol in cache:
        return cache[symbol]
    cache[symbol] = None
    try:
        r = client.get(f"{CHEMBL}/target/search.json", params={"q": symbol, "limit": 10},
                       timeout=60)
        hits = [t for t in r.json().get("targets", [])
                if t.get("organism") == "Homo sapiens"
                and t.get("target_type") == "SINGLE PROTEIN"
                and symbol.upper() in {c.get("component_description", "").upper()
                                       for c in t.get("target_components", [])}
                | {s.get("component_synonym", "").upper()
                   for c in t.get("target_components", [])
                   for s in c.get("target_component_synonyms", [])}]
        if not hits:  # fall back to the best-scoring human single protein
            hits = [t for t in r.json().get("targets", [])
                    if t.get("organism") == "Homo sapiens"
                    and t.get("target_type") == "SINGLE PROTEIN"]
        if not hits:
            return None
        tid = hits[0]["target_chembl_id"]
        r = client.get(f"{CHEMBL}/activity.json", params={
            "target_chembl_id": tid, "pchembl_value__isnull": "false", "limit": 1}, timeout=90)
        n = int(r.json()["page_meta"]["total_count"])
        cache[symbol] = {"chembl_id": tid, "pref_name": hits[0].get("pref_name"),
                         "n_activities": n}
    except Exception as exc:  # noqa: BLE001 - a missing target just gets skipped
        print(f"   chembl lookup failed for {symbol}: {type(exc).__name__}")
    return cache[symbol]


def drug_smiles(client: httpx.Client, chembl_ids: list[str]) -> dict[str, str]:
    """SMILES for approved drugs (Open Targets drug ids are ChEMBL ids)."""
    out = {}
    for i in range(0, len(chembl_ids), 20):
        chunk = chembl_ids[i:i + 20]
        try:
            r = client.get(f"{CHEMBL}/molecule.json", params={
                "molecule_chembl_id__in": ",".join(chunk), "limit": len(chunk)}, timeout=90)
            for m in r.json().get("molecules", []):
                smi = (m.get("molecule_structures") or {}).get("canonical_smiles")
                if smi:
                    out[m["molecule_chembl_id"]] = smi
        except Exception as exc:  # noqa: BLE001
            print(f"   smiles lookup failed: {type(exc).__name__}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets-per-disease", type=int, default=4)
    ap.add_argument("--min-activities", type=int, default=300,
                    help="skip targets with too little measured data to model")
    ap.add_argument("--candidates", type=int, default=25,
                    help="associated targets to consider before filtering")
    args = ap.parse_args()
    t0 = time.time()
    cache: dict[str, dict | None] = {}
    diseases = []

    with httpx.Client(headers={"accept": "application/json"}) as client:
        for efo, label in DISEASES:
            d = graphql(client, {"efo": efo, "n": args.candidates})
            if d is None:
                print(f"{label}: not found")
                continue
            targets = []
            for row in d["associatedTargets"]["rows"]:
                if len(targets) >= args.targets_per_disease:
                    break
                sym = row["target"]["approvedSymbol"]
                info = chembl_target(client, sym, cache)
                if not info or info["n_activities"] < args.min_activities:
                    continue
                targets.append({"symbol": sym, "name": row["target"]["approvedName"],
                                "ensembl_id": row["target"]["id"],
                                "chembl_id": info["chembl_id"],
                                "association_score": round(float(row["score"]), 4),
                                "n_activities": info["n_activities"]})
            approved = [r["drug"] for r in d["drugAndClinicalCandidates"]["rows"]
                        if r.get("maxClinicalStage") == "APPROVAL"
                        and r["drug"].get("drugType") == "Small molecule"]
            seen, uniq = set(), []
            for drug in approved:
                if drug["id"] not in seen:
                    seen.add(drug["id"])
                    uniq.append(drug)
            smiles = drug_smiles(client, [d_["id"] for d_ in uniq[:40]])
            drugs = [{"chembl_id": d_["id"], "name": d_["name"].title(),
                      "smiles": smiles[d_["id"]]} for d_ in uniq if d_["id"] in smiles]

            diseases.append({"id": efo, "label": label, "name": d["name"],
                             "description": (d.get("description") or "")[:600],
                             "targets": targets, "known_drugs": drugs[:25]})
            print(f"{label:<22} {len(targets)} targets "
                  f"[{', '.join(t['symbol'] for t in targets)}]  {len(drugs)} approved drugs")

    all_targets = {t["chembl_id"]: t for d in diseases for t in d["targets"]}
    payload = {
        "built": time.strftime("%Y-%m-%d"),
        "sources": {"associations": "Open Targets Platform", "activities": "ChEMBL"},
        "n_diseases": len(diseases), "n_targets": len(all_targets),
        "diseases": diseases,
        "targets": sorted(all_targets.values(), key=lambda t: -t["n_activities"]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\n{len(diseases)} illnesses, {len(all_targets)} unique targets, "
          f"{sum(len(d['known_drugs']) for d in diseases)} approved drugs "
          f"({time.time() - t0:.0f}s) -> {OUT}")


if __name__ == "__main__":
    main()
