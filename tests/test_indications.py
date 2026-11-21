"""Illness screening: the scoring logic, and the API around it."""

import pytest
from fastapi.testclient import TestClient
from indications import DISCLAIMER, combine_with_toxicity, rank_diseases, score_disease

from app.main import app

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
DISEASE = {
    "id": "X", "label": "Test illness",
    "targets": [
        {"chembl_id": "T1", "symbol": "AAA", "name": "alpha", "association_score": 0.9},
        {"chembl_id": "T2", "symbol": "BBB", "name": "beta", "association_score": 0.4},
        {"chembl_id": "T3", "symbol": "CCC", "name": "gamma", "association_score": 0.2},
    ],
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def has_models(client):
    body = client.get("/indications/diseases").json()
    return any(d["n_modelled"] > 0 for d in body["diseases"])


# --------------------------------------------------------------------------- scoring logic
def test_score_is_the_best_evidence_weighted_target():
    profile = {"T1": {"probability": 0.9, "similarity": 0.7},
               "T2": {"probability": 0.2, "similarity": 0.6}}
    r = score_disease(DISEASE, profile)
    assert r["score"] == pytest.approx(max(0.9 * 0.9, 0.4 * 0.2), abs=1e-4)
    assert r["panel_mean"] == pytest.approx((0.9 * 0.9 + 0.4 * 0.2) / (0.9 + 0.4), abs=1e-4)
    assert r["coverage"] == pytest.approx(2 / 3, abs=0.01)
    assert r["best_target"] == "AAA"


def test_a_selective_molecule_is_not_punished_for_missing_the_other_targets():
    """The donepezil case: active on one well-evidenced protein, inactive on the rest.

    Under an evidence-weighted mean this scored 0.25 and was called "unlikely", which ranked
    Alzheimer's below depression for the standard Alzheimer's drug. Selectivity is how most
    drugs work, so the aggregation must not average it away.
    """
    # the real Alzheimer's panel: four proteins of comparable evidence, only one of them hit
    alzheimers = {
        "id": "MONDO_0004975", "label": "Alzheimer's disease",
        "targets": [{"chembl_id": "ACHE", "symbol": "ACHE", "name": "", "association_score": 0.75},
                    {"chembl_id": "ACE", "symbol": "ACE", "name": "", "association_score": 0.70},
                    {"chembl_id": "APP", "symbol": "APP", "name": "", "association_score": 0.68},
                    {"chembl_id": "CDK5", "symbol": "CDK5", "name": "", "association_score": 0.60}],
    }
    donepezil = {"ACHE": {"probability": 0.99, "similarity": 1.0},
                 "ACE": {"probability": 0.06, "similarity": 0.31},
                 "APP": {"probability": 0.01, "similarity": 0.28},
                 "CDK5": {"probability": 0.04, "similarity": 0.25}}
    r = score_disease(alzheimers, donepezil)
    assert r["score"] == pytest.approx(0.75 * 0.99, abs=1e-4)
    assert r["verdict"] == "worth testing"
    assert r["best_target"] == "ACHE"
    assert r["confidence"] == "high"      # the driving target is an exact match, so trust it
    assert r["panel_mean"] < 0.3          # what the old mean reported: "unlikely"
    assert r["score"] > 2 * r["panel_mean"]


def test_a_molecule_active_on_nothing_still_scores_low():
    r = score_disease(DISEASE, {"T1": {"probability": 0.05, "similarity": 0.9},
                                "T2": {"probability": 0.10, "similarity": 0.9}})
    assert r["score"] < 0.1
    assert r["verdict"] == "unlikely"


def test_targets_without_a_model_are_excluded_not_counted_as_zero():
    one = score_disease(DISEASE, {"T1": {"probability": 0.8, "similarity": 0.9}})
    assert one["score"] == pytest.approx(0.9 * 0.8, abs=1e-6)  # the one model, not averaged
    assert one["coverage"] == pytest.approx(1 / 3, abs=0.01)


def test_no_model_at_all_is_reported_as_such():
    r = score_disease(DISEASE, {})
    assert r["score"] is None
    assert r["verdict"] == "no model"


def test_an_extrapolated_prediction_is_never_reported_as_promising():
    close = score_disease(DISEASE, {"T1": {"probability": 0.95, "similarity": 0.8}})
    far = score_disease(DISEASE, {"T1": {"probability": 0.95, "similarity": 0.05}})
    assert close["verdict"] == "worth testing"
    assert far["score"] == close["score"]       # same number...
    assert far["verdict"] == "unclear"          # ...but not presented as a finding
    assert far["confidence"] == "low"


def test_many_toxicity_flags_downgrade_a_promising_result():
    r = score_disease(DISEASE, {"T1": {"probability": 0.95, "similarity": 0.8}})
    assert combine_with_toxicity(dict(r), 0)["verdict"] == "worth testing"
    assert combine_with_toxicity(dict(r), 5)["verdict"] == "unclear"


def test_ranking_is_sorted_and_keeps_unmodelled_last():
    other = {**DISEASE, "id": "Y", "label": "Other"}
    ranked = rank_diseases([DISEASE, other], {"T1": {"probability": 0.9, "similarity": 0.9}})
    assert [r["score"] is None for r in ranked] == sorted([r["score"] is None for r in ranked])


# --------------------------------------------------------------------------- API
def test_disease_list(client):
    body = client.get("/indications/diseases").json()
    assert len(body["diseases"]) >= 10
    assert body["disclaimer"] == DISCLAIMER
    for d in body["diseases"]:
        assert d["n_modelled"] <= d["n_targets"]


def test_disease_detail_and_unknown_id(client):
    first = client.get("/indications/diseases").json()["diseases"][0]["id"]
    d = client.get(f"/indications/diseases/{first}").json()
    assert d["targets"] and "known_drugs" in d
    assert all(0.0 <= t["association_score"] <= 1.0 for t in d["targets"])
    assert client.get("/indications/diseases/NOT_A_DISEASE").status_code == 404


def test_invalid_smiles_is_rejected(client, has_models):
    if not has_models:
        pytest.skip("no target models in app/models")
    assert client.post("/indications/screen",
                       json={"smiles": "not-a-molecule"}).status_code == 422


def test_screen_ranks_every_illness(client, has_models):
    if not has_models:
        pytest.skip("no target models in app/models")
    body = client.post("/indications/screen", json={"smiles": ASPIRIN}).json()
    scores = [r["score"] for r in body["results"] if r["score"] is not None]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert body["n_targets_modelled"] > 0
    assert body["disclaimer"] == DISCLAIMER


def test_screen_one_illness_matches_the_full_ranking(client, has_models):
    if not has_models:
        pytest.skip("no target models in app/models")
    full = client.post("/indications/screen", json={"smiles": ASPIRIN}).json()["results"]
    pick = next(r for r in full if r["score"] is not None)
    one = client.post("/indications/screen",
                      json={"smiles": ASPIRIN, "disease_id": pick["disease_id"]}).json()
    assert len(one["results"]) == 1
    assert one["results"][0]["score"] == pytest.approx(pick["score"], abs=1e-6)


def test_matrix_covers_every_molecule_and_illness(client, has_models):
    if not has_models:
        pytest.skip("no target models in app/models")
    body = client.post("/indications/matrix", json={"smiles": [ASPIRIN, "CCO"]}).json()
    assert len(body["molecules"]) == 2
    for m in body["molecules"]:
        assert len(m["scores"]) == len(body["diseases"])
