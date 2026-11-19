"""Crystal-structure lookup (Alexandria OPTIMADE API) and structure descriptors.

The lowest-energy known structure of a composition is fetched once and cached in memory.
``app/models/structure_cache.json`` holds pre-fetched structures (example materials of the
web interface), so predictions and the 3D viewer also work offline during a demo. Set the
environment variable ``MATSSL_OFFLINE=1`` to never call the network (tests).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
CACHE_FILE = ROOT / "app" / "models" / "structure_cache.json"
OPTIMADE = "https://alexandria.icams.rub.de/pbe/v1/structures"
FIELDS = "lattice_vectors,cartesian_site_positions,species_at_sites,_alexandria_hull_distance"
sys.path.insert(0, str(ROOT / "pipeline"))

from structure_features import (  # noqa: E402
    arrays_from_optimade,
    default_tables,
    features_from_arrays,
)

logger = logging.getLogger(__name__)
_memory: dict[str, dict | None] = {}


class StructureServiceError(RuntimeError):
    """The structure database could not be reached."""


def names(formula: str) -> tuple[str, str]:
    """(reduced formula, OPTIMADE reduced formula: alphabetical, counts of 1 omitted)."""
    from pymatgen.core import Composition

    comp = Composition(formula)
    amounts = comp.reduced_composition.get_el_amt_dict()
    hill = "".join(el + ("" if n == 1 else f"{n:g}") for el, n in sorted(amounts.items()))
    return comp.reduced_formula, hill


@lru_cache(maxsize=1)
def _disk_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


@lru_cache(maxsize=1)
def _tables() -> dict:
    return default_tables()


def fetch(hill: str, timeout: float = 8.0) -> dict | None:
    """Query Alexandria; returns the lowest-hull entry or None if the composition is unknown."""
    params = {
        "filter": f'chemical_formula_reduced="{hill}"',
        "response_fields": FIELDS,
        "page_limit": 100,
    }
    try:
        r = httpx.get(OPTIMADE, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json().get("data", [])
    except Exception as exc:
        raise StructureServiceError(str(exc)) from exc
    if not data:
        return None

    def hull(d: dict) -> float:  # 0.0 is a valid (best) value, missing -> last
        v = d["attributes"].get("_alexandria_hull_distance")
        return 9.0 if v is None else float(v)

    best = min(data, key=hull)
    a = best["attributes"]
    return {
        "id": best["id"],
        "lattice_vectors": a["lattice_vectors"],
        "cartesian_site_positions": a["cartesian_site_positions"],
        "species_at_sites": a["species_at_sites"],
        "e_above_hull": a.get("_alexandria_hull_distance"),
    }


def ground_state(formula: str, allow_network: bool = True) -> dict | None:
    """Lowest-energy known structure (dict with OPTIMADE arrays) or None.

    Raises StructureServiceError only if the network is needed and unavailable.
    """
    reduced, hill = names(formula)
    if reduced in _memory:
        return _memory[reduced]
    if reduced in _disk_cache():
        _memory[reduced] = _disk_cache()[reduced]
        return _memory[reduced]
    if not allow_network or os.environ.get("MATSSL_OFFLINE") == "1":
        return None
    s = fetch(hill)
    if s is not None:
        s["formula"] = reduced
    _memory[reduced] = s
    return s


def descriptors(s: dict) -> dict | None:
    t = _tables()
    lat, frac, z = arrays_from_optimade(
        s["lattice_vectors"], s["cartesian_site_positions"], s["species_at_sites"], t["sym2z"]
    )
    return features_from_arrays(lat, frac, z, t)


def xyz(s: dict) -> str:
    lines = [str(len(s["species_at_sites"])), f"{s.get('formula', '')} {s['id']}"]
    lines += [
        f"{sp} {x:.5f} {y:.5f} {z:.5f}"
        for sp, (x, y, z) in zip(s["species_at_sites"], s["cartesian_site_positions"], strict=True)
    ]
    return "\n".join(lines)
