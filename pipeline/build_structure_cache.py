"""Pre-fetch the lowest-energy Alexandria structures of the demo materials into
app/models/structure_cache.json, so the web application works offline during a presentation.

Usage:  uv run python pipeline/build_structure_cache.py [extra formulas ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.services import structure as structures  # noqa: E402

DEMO = [
    # example library, demo tour and compare presets of the web interface
    "LiFePO4", "LiCoO2", "LiMn2O4", "LiNiO2", "LiMnPO4", "CsPbI3", "CdTe", "GaAs",
    "Cu2ZnSnS4", "Nd2Fe14B", "SmCo5", "Fe3O4", "MnBi", "Ni3Al", "TiAl", "SrTiO3", "NaCl",
    # common materials people type during a demo
    "Fe2O3", "Si", "Cu", "Fe", "Al", "Al2O3", "TiO2", "ZnO", "MgO", "GaN", "MoS2", "CoFe2O4",
    "BaTiO3", "Bi2Te3", "CuInSe2", "Sb2Se3", "FeS2", "Cs2AgBiBr6", "SnS", "Cu2O", "NiO",
    "CoO", "MnO", "CrO2", "LiF", "KCl", "CaTiO3", "WO3", "V2O5", "Mn3O4", "FeO", "Ge", "InP",
]


def main() -> None:
    names = DEMO + sys.argv[1:]
    cache = {}
    if structures.CACHE_FILE.exists():
        cache = json.loads(structures.CACHE_FILE.read_text(encoding="utf-8"))
    for f in names:
        reduced, hill = structures.names(f)
        if reduced in cache:
            continue
        try:
            s = structures.fetch(hill)
        except structures.StructureServiceError as exc:
            print(f"{f:<12} network error: {exc}")
            continue
        if s is None:
            print(f"{f:<12} not in Alexandria")
            continue
        s["formula"] = reduced
        cache[reduced] = s
        print(f"{f:<12} {s['id']}  {len(s['species_at_sites'])} atoms  "
              f"e_hull={s.get('e_above_hull')}")
    structures.CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")
    print(f"{len(cache)} structures in {structures.CACHE_FILE}")


if __name__ == "__main__":
    main()
