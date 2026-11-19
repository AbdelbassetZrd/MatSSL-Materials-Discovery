"""Structure descriptors and the structure-aware (high-accuracy) prediction track."""

import os

import pytest
from fastapi.testclient import TestClient

os.environ["MATSSL_OFFLINE"] = "1"  # never call the structure database in tests

from structure_features import (  # noqa: E402
    STRUCT_COLUMNS,
    arrays_from_pymatgen_dict,
    default_tables,
    features_from_arrays,
)

from app.main import app  # noqa: E402
from app.services import model as service  # noqa: E402

FORBIDDEN = ("energy", "hull", "e_form", "magmom", "mag", "gap", "target")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_structure_columns_do_not_leak_labels():
    for col in STRUCT_COLUMNS:
        assert not any(word in col.lower() for word in FORBIDDEN), col


def test_descriptors_are_cell_invariant():
    from pymatgen.core import Lattice, Structure

    conv = Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    prim = conv.get_primitive_structure()
    t = default_tables()
    f1 = features_from_arrays(*arrays_from_pymatgen_dict(conv.as_dict(), t["sym2z"]), t)
    f2 = features_from_arrays(*arrays_from_pymatgen_dict(prim.as_dict(), t["sym2z"]), t)
    assert f1["struct_spg"] == 225
    for k in ("struct_nprim", "struct_vpa", "struct_density", "struct_nn_min", "struct_cn_mean"):
        assert f1[k] == pytest.approx(f2[k], rel=1e-6), k
    assert f1["struct_density"] == pytest.approx(2.16, abs=0.05)


def test_structure_track_for_cached_material(client):
    if not service.get_registry().hiacc:
        pytest.skip("no structure-aware models in app/models")
    body = client.post("/materials/predict", json={"formula": "Fe2O3"}).json()
    if body["structure_status"] != "found":
        pytest.skip("Fe2O3 not in app/models/structure_cache.json")
    assert body["structure"]["id"].startswith("agm")
    tracks = {p["task"]: p["model_track"] for p in body["predictions"]}
    for task in service.get_registry().hiacc:
        assert tracks[task] == "structure"
    for p in body["predictions"]:
        assert 0 <= p["probability"] <= 1
        if p["model_track"] == "structure":
            assert p["ssl_probability"] is not None


def test_unknown_structure_falls_back_to_composition(client):
    body = client.post("/materials/predict", json={"formula": "Li7Ne3"}).json()
    assert body["structure"] is None
    assert all(p["model_track"] == "composition" for p in body["predictions"])
