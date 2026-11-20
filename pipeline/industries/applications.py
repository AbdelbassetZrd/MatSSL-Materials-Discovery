"""Application scores for the Design Lab.

Each application turns the deployed property models into a single 0-1 score for a target use,
plus the breakdown that produced it -- so the interface can always show *why* a composition is
or is not a good candidate, never just a number.

Inputs (all optional; a missing model simply drops its term):
    stable   P(within 0.05 eV/atom of the convex hull)
    metal    P(metallic)
    magnetic P(magnetic)
    pv_gap   P(PBE band gap inside the 1.1-1.7 eV photovoltaic window)
    tc       predicted superconducting critical temperature, in kelvin
    elements {symbol: atomic fraction}
"""

from __future__ import annotations

LN2 = 77.0  # liquid-nitrogen boiling point: the line that makes a superconductor practical
TC_REFERENCE = LN2  # a Tc at or above it scores full marks on the temperature term

WORKING_IONS = {"Li", "Na", "K", "Mg", "Ca", "Zn", "Al"}
REDOX_METALS = {"Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Nb", "Mo", "W"}
CATHODE_ANIONS = {"O", "S", "F", "P", "Si", "Se"}


def _gate(value: float | None, default: float = 1.0) -> float:
    return default if value is None else float(value)


def superconductor(p: dict) -> dict:
    """High critical temperature, and stable enough to be made."""
    tc = _gate(p.get("tc"), 0.0)
    temp = min(max(tc, 0.0), TC_REFERENCE) / TC_REFERENCE
    stable = _gate(p.get("stable"))
    return {
        "score": temp * stable,
        "headline": f"{tc:.0f} K predicted Tc",
        "components": [
            {"label": "predicted Tc", "value": temp,
             "detail": f"{tc:.0f} K (full marks at {TC_REFERENCE:.0f} K, liquid nitrogen)"},
            {"label": "stability", "value": stable, "detail": "P(near convex hull)"},
        ],
        "flags": ["above liquid nitrogen (77 K)"] if tc > LN2 else [],
    }


def solar_absorber(p: dict) -> dict:
    """Band gap inside the photovoltaic window, not a metal, and stable."""
    gap, stable, metal = _gate(p.get("pv_gap"), 0.0), _gate(p.get("stable")), _gate(p.get("metal"), 0.0)
    return {
        "score": gap * stable * (1 - metal),
        "headline": f"{100 * gap:.0f}% chance of a 1.1-1.7 eV gap",
        "components": [
            {"label": "PV band gap", "value": gap, "detail": "P(gap in 1.1-1.7 eV)"},
            {"label": "stability", "value": stable, "detail": "P(near convex hull)"},
            {"label": "not metallic", "value": 1 - metal, "detail": "1 - P(metal)"},
        ],
        "flags": [],
    }


def permanent_magnet(p: dict) -> dict:
    """Magnetic, metallic and stable -- the profile of a hard magnet such as Nd2Fe14B."""
    mag, stable, metal = _gate(p.get("magnetic"), 0.0), _gate(p.get("stable")), _gate(p.get("metal"), 0.0)
    rare_earth = {"Nd", "Sm", "Pr", "Dy", "Tb", "Ce", "La", "Gd"} & set(p.get("elements") or {})
    return {
        "score": mag * stable * metal,
        "headline": f"{100 * mag:.0f}% chance of being magnetic",
        "components": [
            {"label": "magnetism", "value": mag, "detail": "P(magnetic)"},
            {"label": "metallic", "value": metal, "detail": "P(metal)"},
            {"label": "stability", "value": stable, "detail": "P(near convex hull)"},
        ],
        "flags": [f"contains rare earth: {', '.join(sorted(rare_earth))}"] if rare_earth else [],
    }


def battery_cathode(p: dict) -> dict:
    """Chemistry filter for an intercalation cathode, combined with the stability model.

    This is deliberately *not* a voltage prediction: it checks that the composition has a
    mobile working ion, a redox-active transition metal and a suitable anion, and that the
    model believes it can exist. Predicting voltage and capacity needs computed electrode
    pairs (Materials Project insertion-electrode data), which is the documented next step.
    """
    els = set(p.get("elements") or {})
    ion = els & WORKING_IONS
    redox = els & REDOX_METALS
    anion = els & CATHODE_ANIONS
    chem = (1.0 if ion else 0.0) * (1.0 if redox else 0.0) * (1.0 if anion else 0.0)
    stable, metal = _gate(p.get("stable")), _gate(p.get("metal"), 0.0)
    flags = []
    if not ion:
        flags.append("no mobile working ion (Li, Na, Mg, K, Ca, Zn, Al)")
    if not redox:
        flags.append("no redox-active transition metal")
    if not anion:
        flags.append("no suitable anion (O, S, F, P, Si, Se)")
    return {
        "score": chem * stable * (1 - 0.5 * metal),
        "headline": (f"{'/'.join(sorted(ion))} cathode chemistry" if chem else "chemistry does not fit"),
        "components": [
            {"label": "cathode chemistry", "value": chem,
             "detail": f"ion {', '.join(sorted(ion)) or '-'} · redox {', '.join(sorted(redox)) or '-'}"},
            {"label": "stability", "value": stable, "detail": "P(near convex hull)"},
            {"label": "not a pure metal", "value": 1 - 0.5 * metal, "detail": "1 - 0.5 P(metal)"},
        ],
        "flags": flags,
    }


APPLICATIONS = {
    "superconductor": {
        "title": "Superconductor",
        "subtitle": "High critical temperature",
        "question": "Could this composition superconduct at a usable temperature?",
        "needs": ["tc", "stable"],
        "fn": superconductor,
        "examples": ["YBa2Cu3O7", "MgB2", "Nb3Sn", "LaFeAsO"],
        "note": "Tc is predicted by a stacked ensemble trained on 15,542 measured compositions "
                "(SuperCon). 77 K is the liquid-nitrogen line.",
    },
    "solar_absorber": {
        "title": "Solar absorber",
        "subtitle": "Band gap in the photovoltaic window",
        "question": "Could this composition absorb sunlight efficiently?",
        "needs": ["pv_gap", "stable", "metal"],
        "fn": solar_absorber,
        "examples": ["CsPbI3", "CdTe", "Cu2ZnSnS4", "GaAs"],
        "note": "Uses the GA-selected band-gap model (PBE gap in 1.1-1.7 eV). PBE under-estimates "
                "gaps, so the window is a window on PBE values.",
    },
    "permanent_magnet": {
        "title": "Permanent magnet",
        "subtitle": "Magnetic, metallic and stable",
        "question": "Could this composition work as a hard magnet?",
        "needs": ["magnetic", "metal", "stable"],
        "fn": permanent_magnet,
        "examples": ["Nd2Fe14B", "SmCo5", "MnBi", "Fe3O4"],
        "note": "Magnetism and metallicity come from the deployed semi-supervised models; the "
                "score does not model coercivity or anisotropy.",
    },
    "battery_cathode": {
        "title": "Battery cathode",
        "subtitle": "Intercalation chemistry + stability",
        "question": "Does this composition have the makings of an intercalation cathode?",
        "needs": ["stable", "metal"],
        "fn": battery_cathode,
        "examples": ["LiFePO4", "LiCoO2", "LiMn2O4", "NaFePO4"],
        "note": "Chemistry filter (working ion + redox metal + anion) combined with the stability "
                "model. It is not a voltage or capacity prediction.",
    },
}


def score(app: str, p: dict) -> dict:
    cfg = APPLICATIONS[app]
    out = cfg["fn"](p)
    s = float(max(0.0, min(1.0, out["score"])))
    out["score"] = s
    out["application"] = app
    out["title"] = cfg["title"]
    out["verdict"] = ("strong candidate" if s >= 0.6 else
                      "worth a look" if s >= 0.3 else
                      "weak candidate")
    return out
