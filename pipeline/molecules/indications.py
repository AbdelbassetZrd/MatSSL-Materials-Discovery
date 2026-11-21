"""Turn a molecule's predicted protein-target activities into an illness ranking.

What this does and does not claim
---------------------------------
A model trained on ChEMBL can estimate whether a molecule is **active against a protein
target**. It cannot estimate whether a molecule treats a disease: that depends on absorption,
dose, delivery, selectivity, toxicity and clinical effect, none of which are in the data.

Illnesses enter only as *context*. Open Targets records how strongly the public evidence links
each protein to each illness, and those association scores are used to weight the per-target
predictions:

    score(molecule, illness) = max_t  w_t * P(active | molecule, target t)

that is, the single most promising protein: how likely the molecule is to be active against it,
discounted by how strongly the evidence links it to the illness. The maximum rather than a mean,
because a drug does not have to hit every protein associated with an illness - a selective
inhibitor of one of them is the normal case, and averaging punishes exactly the molecules that
work. See `score_disease` for why a noisy-OR was rejected too.

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
                    "test_roc_auc": pred.get("test_roc_auc"),
                    # the honest score and the denominator it rests on, so the interface can
                    # show how well measured this particular protein model really is
                    "roc_auc_measured": pred.get("roc_auc_measured"),
                    "n_measured_minority": pred.get("n_measured_minority"),
                    "well_supported": pred.get("well_supported")}
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

    # The score is the best evidence-weighted target, not the mean over all of them.
    #
    #     score = max_t  w_t * p_t
    #
    # Why the mean was wrong: a drug does not have to hit every protein linked to an illness, it
    # has to hit one that matters. Donepezil is a selective acetylcholinesterase inhibitor, and
    # that is exactly how it treats Alzheimer's - but averaging its 99 % on ACHE against its 1 %
    # on APP and 4 % on CDK5 gave Alzheimer's 25, ranking it below depression and labelling it
    # "unlikely". Selectivity is the normal case for a drug, and the mean penalises it.
    #
    # Why not a noisy-OR, 1 - prod(1 - w_t p_t), which is the other obvious choice: it assumes
    # the per-target predictions are independent. They are not - they are all computed from the
    # same descriptors and fingerprint of the same molecule, so they are strongly correlated, and
    # a noisy-OR turns four mediocre correlated predictions into one confident-looking score. The
    # maximum makes the weaker claim, and it is the claim the data supports.
    #
    # The evidence-weighted mean is still reported as `panel_mean`: "engages the whole panel" is
    # a real question, just not the one being asked here.
    driver = max(modelled, key=lambda r: r["association_score"] * r["probability"])
    score = driver["association_score"] * driver["probability"]
    panel_mean = num / den

    # Confidence follows the target that drives the score, not the least relevant one. Under the
    # mean every target contributed, so the weakest applicability domain mattered; now the claim
    # rests on one target, so that is the one whose domain decides whether to trust it.
    return {
        "disease_id": disease["id"], "label": disease["label"],
        "score": round(score, 4),
        "panel_mean": round(panel_mean, 4),
        "verdict": band(score, driver["confidence"]),
        "coverage": round(len(modelled) / len(rows), 2),
        "confidence": driver["confidence"],
        "best_target": driver["symbol"],
        "targets": rows,
        "note": (f"{len(modelled)} of {len(rows)} associated proteins have a model; the score is "
                 f"driven by {driver['symbol']} "
                 f"(predicted {100 * driver['probability']:.0f}% active, evidence "
                 f"{driver['association_score']:.2f})"),
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
