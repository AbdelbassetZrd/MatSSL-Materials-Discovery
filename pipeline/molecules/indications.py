"""Turn a molecule's predicted protein-target activities into an illness ranking.

What this does and does not claim
---------------------------------
A model trained on ChEMBL can estimate whether a molecule is **active against a protein
target**. It cannot estimate whether a molecule treats a disease: that depends on absorption,
dose, delivery, selectivity, toxicity and clinical effect, none of which are in the data.

Illnesses enter only as *context*. Open Targets records how strongly the public evidence links
each protein to each illness, and those association scores are used to weight the per-target
predictions:

    score(molecule, illness) = sum_t  w_t * P(active | molecule, target t)  /  sum_t  w_t

The result is a **screening score**: a way to rank which illnesses are worth a closer look for
a given molecule, in the same spirit as a reverse-screening tool. It is a hypothesis, not a
finding, and never a treatment recommendation.

These are pure functions so that the web service and the offline validation score molecules
identically (the pattern used by pipeline/industries/applications.py).
"""

from __future__ import annotations

# A prediction is only trustworthy if the molecule resembles something the model was trained
# on. The thresholds are read off the reliability-vs-similarity curves measured per target in
# train_targets.py, not chosen by hand.
SIMILAR_ENOUGH = 0.40  # Tanimoto to the nearest known active
CLOSE = 0.55

BANDS = [
    (0.55, "worth testing"),
    (0.30, "unclear"),
    (0.00, "unlikely"),
]


def confidence(similarity: float | None) -> tuple[str, str]:
    """How far the molecule is from the chemistry the target model actually learned."""
    if similarity is None:
        return "unknown", "no reference actives for this target"
    if similarity >= CLOSE:
        return "high", f"closely resembles a known active (Tanimoto {similarity:.2f})"
    if similarity >= SIMILAR_ENOUGH:
        return "medium", f"moderately similar to known actives (Tanimoto {similarity:.2f})"
    return "low", (f"unlike anything the model was trained on (Tanimoto {similarity:.2f}): "
                   "the prediction is an extrapolation")


def band(score: float, worst_confidence: str) -> str:
    verdict = next(label for cut, label in BANDS if score >= cut)
    # an extrapolated prediction is reported as unclear rather than as a finding
    if verdict == "worth testing" and worst_confidence == "low":
        return "unclear"
    return verdict


def score_disease(disease: dict, profile: dict) -> dict:
    """Weighted combination of the target predictions available for one illness.

    disease  entry from data/processed/diseases.json
    profile  {chembl_id: {"probability": float, "similarity": float|None, ...}}
    """
    rows, num, den = [], 0.0, 0.0
    for t in disease["targets"]:
        pred = profile.get(t["chembl_id"])
        w = float(t["association_score"])
        row = {"symbol": t["symbol"], "name": t.get("name"), "chembl_id": t["chembl_id"],
               "association_score": round(w, 3), "modelled": pred is not None}
        if pred is not None:
            p = float(pred["probability"])
            level, why = confidence(pred.get("similarity"))
            row |= {"probability": p, "similarity": pred.get("similarity"),
                    "nearest_active": pred.get("nearest_active"),
                    "confidence": level, "confidence_note": why,
                    "decoy_fraction": pred.get("decoy_fraction"),
                    "test_roc_auc": pred.get("test_roc_auc")}
            num += w * p
            den += w
        rows.append(row)

    rows.sort(key=lambda r: (-r.get("probability", -1), -r["association_score"]))
    modelled = [r for r in rows if r["modelled"]]
    if not modelled:
        return {"disease_id": disease["id"], "label": disease["label"], "score": None,
                "verdict": "no model", "coverage": 0.0, "targets": rows,
                "note": "none of the proteins linked to this illness has enough measured "
                        "activity data to train a model"}
    order = {"low": 0, "medium": 1, "high": 2, "unknown": 0}
    worst = min((r["confidence"] for r in modelled), key=lambda c: order[c])
    score = num / den
    return {
        "disease_id": disease["id"], "label": disease["label"],
        "score": round(score, 4),
        "verdict": band(score, worst),
        "coverage": round(len(modelled) / len(rows), 2),
        "confidence": worst,
        "best_target": modelled[0]["symbol"],
        "targets": rows,
        "note": (f"{len(modelled)} of {len(rows)} associated proteins have a model; "
                 f"strongest predicted activity on {modelled[0]['symbol']}"),
    }


def rank_diseases(diseases: list[dict], profile: dict) -> list[dict]:
    """Every illness scored for one molecule, best first - the 'what about illness C?' view."""
    scored = [score_disease(d, profile) for d in diseases]
    scored.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0)))
    return scored


def combine_with_toxicity(result: dict, n_toxicity_flags: int | None) -> dict:
    """A molecule flagged on many toxicity assays is not a promising starting point."""
    if n_toxicity_flags is None or result.get("score") is None:
        return result
    result["toxicity_flags"] = n_toxicity_flags
    if n_toxicity_flags >= 3 and result["verdict"] == "worth testing":
        result["verdict"] = "unclear"
        result["note"] += (f"; downgraded because {n_toxicity_flags} toxicity endpoints "
                           "are predicted active")
    return result


DISCLAIMER = ("In-silico screening for research and teaching. These are predicted activities "
              "against proteins associated with the illness in public databases - not evidence "
              "of a treatment effect, and not medical advice.")
