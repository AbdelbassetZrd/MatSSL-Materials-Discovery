"""API tests for the materials prediction service."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Materials" in r.json()["message"]


def test_models_info(client):
    r = client.get("/materials/models-info")
    assert r.status_code == 200
    tasks = {m["task"] for m in r.json()["models"]}
    assert {"metallicity", "magnetism", "stability", "band_gap"} <= tasks


def test_analyze_formula(client):
    r = client.post("/materials/analyze", json={"formula": "LiFePO4"})
    assert r.status_code == 200
    body = r.json()
    assert body["reduced_formula"] == "LiFePO4"
    assert body["n_elements"] == 4
    assert body["chemical_rules"]["charge_balanced"] is True


def test_invalid_formula(client):
    r = client.post("/materials/predict", json={"formula": "Xx12Qq"})
    assert r.status_code == 422


def test_predict_formula(client):
    loaded = [
        m for m in client.get("/materials/models-info").json()["models"] if m["status"] == "Loaded"
    ]
    if not loaded:
        pytest.skip("no trained models available in app/models")
    r = client.post("/materials/predict", json={"formula": "Fe2O3"})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == len(loaded)
    for p in preds:
        assert 0.0 <= p["probability"] <= 1.0
        assert p["prediction"] in (0, 1)


def test_similar_known_composition(client):
    r = client.get("/materials/similar", params={"formula": "LiFePO4", "k": 5})
    if r.status_code == 503:
        pytest.skip("neighbor index not built")
    assert r.status_code == 200
    body = r.json()
    assert len(body["neighbors"]) == 5
    assert body["neighbors"][0]["formula"] == "LiFePO4"
    assert body["known"] is True


def test_experiments_dashboard(client):
    r = client.get("/materials/experiments")
    if r.status_code == 404:
        pytest.skip("dashboard.json not exported")
    assert r.status_code == 200
    body = r.json()
    assert {"datasets", "ssl", "kpis"} <= set(body)


def test_prediction_has_explanation_and_threshold(client):
    loaded = [
        m for m in client.get("/materials/models-info").json()["models"] if m["status"] == "Loaded"
    ]
    if not loaded:
        pytest.skip("no trained models available in app/models")
    preds = client.post("/materials/predict", json={"formula": "Fe2O3"}).json()["predictions"]
    for p in preds:
        assert 0 < p["threshold"] < 1
        assert p["explanation"] is None or p["explanation"]["type"] in ("shap", "percentile")
