"""API tests for the molecular toxicity service."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def loaded(client):
    return [
        m for m in client.get("/molecules/models-info").json()["models"] if m["status"] == "Loaded"
    ]


def test_models_info_lists_every_endpoint(client):
    models = client.get("/molecules/models-info").json()["models"]
    endpoints = {m["endpoint"] for m in models}
    assert {"NR-AR", "SR-MMP", "CT_TOX", "FDA_APPROVED"} <= endpoints
    assert len(models) == 14


def test_properties_without_any_model(client):
    body = client.post("/molecules/properties", json={"smiles": ASPIRIN}).json()
    assert body["formula"] == "C9H8O4"
    assert 179 < body["descriptors"]["molecular_weight"] < 181
    assert body["lipinski"]["passes"] is True


def test_invalid_smiles_is_rejected(client):
    assert client.post("/molecules/predict", json={"smiles": "not-a-molecule"}).status_code == 422
    assert client.post("/molecules/properties", json={"smiles": "C((("}).status_code == 422
    assert client.post("/molecules/predict", json={"smiles": "   "}).status_code == 422


def test_predict_returns_one_entry_per_loaded_model(client, loaded):
    if not loaded:
        pytest.skip("no molecular models in app/models")
    body = client.post("/molecules/predict", json={"smiles": ASPIRIN}).json()
    assert len(body["predictions"]) == len(loaded)
    for p in body["predictions"]:
        assert 0.0 <= p["probability"] <= 1.0
        assert p["prediction"] in (0, 1)
        assert 0.0 < p["threshold"] < 1.0
        assert p["explanation"] is None or p["explanation"]["type"] == "shap"
    assert body["risk"]["level"] in ("Low", "Moderate", "High")
    assert body["risk"]["n_flagged"] <= body["risk"]["n_endpoints"]


def test_explanation_features_are_named(client, loaded):
    if not loaded:
        pytest.skip("no molecular models in app/models")
    preds = client.post("/molecules/predict", json={"smiles": ASPIRIN}).json()["predictions"]
    items = [i for p in preds if p["explanation"] for i in p["explanation"]["items"]]
    assert items, "expected at least one SHAP explanation"
    for i in items:
        assert i["kind"] in ("descriptor", "substructure")
        assert i["feature"] and not i["feature"].startswith("bit")


def test_batch_reports_errors_separately(client, loaded):
    if not loaded:
        pytest.skip("no molecular models in app/models")
    body = client.post("/molecules/batch-predict", json={"smiles": [ASPIRIN, "@@@"]}).json()
    assert len(body["results"]) == 1
    assert len(body["errors"]) == 1
