"""Does the illness screening actually work? Two honest tests.

1. **Repurposing retrieval.** Every illness in the knowledge base comes with the small-molecule
   drugs already approved for it. Those drugs are pushed through the whole chain - featurise,
   run every protein model, weight by association score - and we check where their own illness
   lands in the ranking. If the pipeline carries any signal, a drug's own indication should
   come out near the top. The baseline is random ranking, 1/n_diseases.

   This is a weak-ish test: the same drugs appear in ChEMBL, so a model may simply remember
   them. The number is still worth reporting, because a pipeline that *fails* it is broken.
   The split-aware per-target ROC-AUC in train_targets.py is the strict measurement.

2. **Negative controls.** Molecules nobody would call drugs (water, ethanol, glucose, methane)
   should not light up every illness. If they do, the models are reading drug-likeness rather
   than target activity.

Usage:  uv run python pipeline/molecules/validate_indications.py
Output: data/predictions/targets/validation.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.services import indications as svc  # noqa: E402

OUT = ROOT / "data" / "predictions" / "targets" / "validation.json"
CONTROLS = [("water", "O"), ("ethanol", "CCO"), ("methane", "C"),
            ("glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"),
            ("acetic acid", "CC(=O)O"), ("benzene", "c1ccccc1")]


def main() -> None:
    reg = svc.get_registry()
    if not reg.targets:
        print("no target models yet - run pipeline/molecules/train_targets.py first")
        return
    t0 = time.time()
    n_dis = len(reg.diseases)
    print(f"{len(reg.targets)} protein models, {n_dis} illnesses "
          f"(random baseline = {1 / n_dis:.3f})\n")

    ranks, per_disease, tested = [], [], 0
    for d in reg.diseases:
        drugs = d["known_drugs"][:12]
        if not drugs:
            continue
        got = []
        for drug in drugs:
            try:
                res = svc.screen(drug["smiles"], with_toxicity=False)["results"]
            except Exception:  # noqa: BLE001 - a drug RDKit cannot parse is skipped
                continue
            scored = [r for r in res if r["score"] is not None]
            if not scored:
                continue
            order = [r["disease_id"] for r in scored]
            if d["id"] not in order:
                continue
            got.append(order.index(d["id"]) + 1)
            tested += 1
        if got:
            ranks += got
            per_disease.append({"illness": d["label"], "n_drugs": len(got),
                                "median_rank": float(np.median(got)),
                                "top3": float(np.mean([r <= 3 for r in got]))})
            print(f"{d['label']:<22} {len(got):>2} drugs  median rank "
                  f"{np.median(got):>4.1f}/{n_dis}  top-3 {np.mean([r <= 3 for r in got]):.0%}")

    ranks_arr = np.array(ranks)
    summary = {
        "n_diseases": n_dis, "n_targets": len(reg.targets), "n_drugs_tested": int(tested),
        "random_baseline_top1": round(1 / n_dis, 4),
        "top1": float(np.mean(ranks_arr == 1)) if len(ranks_arr) else None,
        "top3": float(np.mean(ranks_arr <= 3)) if len(ranks_arr) else None,
        "top5": float(np.mean(ranks_arr <= 5)) if len(ranks_arr) else None,
        "median_rank": float(np.median(ranks_arr)) if len(ranks_arr) else None,
        "per_disease": per_disease,
    }
    print(f"\nrepurposing retrieval over {tested} approved drugs:")
    for k in ("top1", "top3", "top5"):
        base = {"top1": 1, "top3": 3, "top5": 5}[k] / n_dis
        print(f"   {k}: {summary[k]:.3f}   (random {base:.3f})")
    print(f"   median rank {summary['median_rank']:.1f} of {n_dis}")

    print("\nnegative controls (should stay low):")
    controls = []
    for name, smi in CONTROLS:
        try:
            res = svc.screen(smi, with_toxicity=False)["results"]
        except Exception:  # noqa: BLE001
            continue
        scored = [r["score"] for r in res if r["score"] is not None]
        if not scored:
            continue
        top = max(scored)
        n_worth = sum(1 for r in res if r.get("verdict") == "worth testing")
        controls.append({"name": name, "smiles": smi, "max_score": round(float(top), 3),
                         "n_worth_testing": n_worth})
        print(f"   {name:<12} highest illness score {top:.3f}  "
              f"{n_worth} flagged 'worth testing'")
    summary["negative_controls"] = controls
    summary["runtime_s"] = round(time.time() - t0, 1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
