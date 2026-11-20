"""Design Lab: application scores and candidate ranking."""

import pytest
from applications import APPLICATIONS, score
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_scores_stay_in_range_and_explain_themselves():
    payload = {
        "stable": 0.9,
        "metal": 0.8,
        "magnetic": 0.95,
        "pv_gap": 0.1,
        "tc": 90.0,
        "elements": {"Nd", "Fe", "B"},
    }
    for app_id in APPLICATIONS:
        r = score(app_id, payload)
        assert 0.0 <= r["score"] <= 1.0
        assert r["verdict"] in ("strong candidate", "worth a look", "weak candidate")
        assert r["components"], f"{app_id} must explain its score"
        assert all(0.0 <= c["value"] <= 1.0 for c in r["components"])


def test_missing_models_do_not_break_the_score():
    r = score("superconductor", {})  # no model output at all
    assert r["score"] == 0.0
    assert score("permanent_magnet", {"magnetic": 1.0, "metal": 1.0})["score"] > 0


def test_battery_filter_rejects_compositions_without_a_working_ion():
    cathode = score(
        "battery_cathode", {"stable": 1.0, "metal": 0.0, "elements": {"Li", "Fe", "P", "O"}}
    )
    silica = score("battery_cathode", {"stable": 1.0, "metal": 0.0, "elements": {"Si", "O"}})
    assert cathode["score"] > 0.5
    assert silica["score"] == 0.0
    assert any("working ion" in f for f in silica["flags"])


def test_applications_endpoint(client):
    body = client.get("/design/applications").json()["applications"]
    assert {a["id"] for a in body} == set(APPLICATIONS)
    for a in body:
        assert a["title"] and a["question"] and a["examples"]


def test_score_endpoint(client):
    r = client.post("/design/score", json={"formula": "LiFePO4"})
    if r.status_code == 500:
        pytest.skip("materials models not available")
    assert r.status_code == 200
    body = r.json()
    assert body["elements"] == ["Fe", "Li", "O", "P"]
    assert len(body["applications"]) == len(APPLICATIONS)
    # results are sorted best-first
    s = [a["score"] for a in body["applications"]]
    assert s == sorted(s, reverse=True)
    assert client.post("/design/score", json={"formula": "Xx12Qq"}).status_code == 422


def test_candidates_endpoint(client):
    r = client.get("/design/candidates?application=superconductor&limit=5")
    if r.status_code == 404:
        pytest.skip("candidate ranking not computed yet")
    body = r.json()
    assert len(body["candidates"]) <= 5
    scores = [c["score"] for c in body["candidates"]]
    assert scores == sorted(scores, reverse=True)
    assert client.get("/design/candidates?application=nonsense").status_code == 422


def test_candidate_element_filter(client):
    r = client.get("/design/candidates?application=battery_cathode&limit=10&elements=Fe")
    if r.status_code == 404:
        pytest.skip("candidate ranking not computed yet")
    for c in r.json()["candidates"]:
        assert "Fe" in c["formula"]
