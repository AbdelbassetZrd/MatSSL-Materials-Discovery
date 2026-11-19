"""Cheap crystal-structure descriptors (numpy + spglib).

The same function is used to build the training sets (Alexandria, JARVIS, Matbench
structures) and inside the web application (structure fetched from the Alexandria
OPTIMADE API), so training and inference features are computed identically.

Only geometry and symmetry are used - never energies, magnetic moments or band gaps,
which would leak the labels.
"""

from __future__ import annotations

import os

os.environ.setdefault("SPGLIB_WARNING", "OFF")

import numpy as np  # noqa: E402
import spglib  # noqa: E402

CUTOFF = 5.0  # neighbour search radius (Angstrom)
CRYSTAL_SYSTEMS = ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal",
                   "hexagonal", "cubic"]
_CS_BOUNDS = [2, 15, 74, 142, 167, 194, 230]

STRUCT_COLUMNS = [
    "struct_nprim", "struct_nsym_sites", "struct_formula_units", "struct_vpa",
    "struct_density", "struct_packing", "struct_vol_ratio", "struct_spg",
    "struct_pg_order", "struct_b_a", "struct_c_a", "struct_angle_dev",
    *[f"struct_cs_{c}" for c in CRYSTAL_SYSTEMS],
    "struct_nn_min", "struct_nn_mean", "struct_nn_std", "struct_ratio_min",
    "struct_ratio_mean", "struct_cn_mean", "struct_cn_std", "struct_cn_min", "struct_cn_max",
]


def default_tables() -> dict:
    """Element tables (indexed by Z-1) taken from the Magpie data used by featurize.py."""
    from featurize import MAGPIE_PROPS, SYM2IDX, element_table

    t = element_table()
    cov = t[:, MAGPIE_PROPS.index("CovalentRadius")] / 100.0  # pm -> Angstrom
    cov = np.where(np.isfinite(cov), cov, np.nanmean(cov))
    wt = t[:, MAGPIE_PROPS.index("AtomicWeight")]
    wt = np.where(np.isfinite(wt), wt, np.nanmean(wt))
    gsv = t[:, MAGPIE_PROPS.index("GSvolume_pa")]
    gsv = np.where(np.isfinite(gsv), gsv, np.nanmean(gsv))
    return {"cov": cov, "wt": wt, "gsv": gsv, "sym2z": {s: i + 1 for s, i in SYM2IDX.items()}}


def _crystal_system(spg: int) -> int:
    for i, b in enumerate(_CS_BOUNDS):
        if spg <= b:
            return i
    return 0


def features_from_arrays(lattice, frac, z, tables: dict) -> dict | None:
    """Descriptors of one periodic structure.

    lattice: 3x3 row vectors (Angstrom); frac: N x 3 fractional coordinates; z: N atomic numbers.
    """
    lat = np.asarray(lattice, dtype=float)
    frac = np.mod(np.asarray(frac, dtype=float), 1.0)
    z = np.asarray(z, dtype=int)
    n = len(z)
    vol = abs(np.linalg.det(lat))
    if n == 0 or vol < 1e-6 or z.min() < 1 or z.max() > len(tables["cov"]):
        return None
    cov, wt, gsv = tables["cov"][z - 1], tables["wt"][z - 1], tables["gsv"][z - 1]

    # --- symmetry (cell-choice invariant quantities)
    try:
        ds = spglib.get_symmetry_dataset((lat, frac, z), symprec=0.1)
    except Exception:  # noqa: BLE001
        ds = None
    if ds is not None:
        spg = int(ds.number)
        rot = np.asarray(ds.rotations).reshape(len(ds.rotations), -1)
        pg_order = len(np.unique(rot, axis=0))
        nprim = len(set(np.asarray(ds.mapping_to_primitive).tolist()))
        nsym = len(set(np.asarray(ds.equivalent_atoms).tolist()))
        std = np.asarray(ds.std_lattice, dtype=float)
    else:
        spg, pg_order, nprim, nsym, std = 1, 1, n, n, lat
    lengths = np.linalg.norm(std, axis=1)
    a, b, c = np.sort(lengths)
    angles = []
    for i, j in ((1, 2), (0, 2), (0, 1)):
        cosang = np.dot(std[i], std[j]) / (lengths[i] * lengths[j])
        angles.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
    counts = np.bincount(z)
    counts = counts[counts > 0]
    gcd = np.gcd.reduce(counts)
    fu = (n / (counts.sum() / gcd)) * (nprim / n)  # formula units per primitive cell

    # --- neighbour statistics (periodic images within CUTOFF)
    cart = frac @ lat
    recip = np.linalg.inv(lat).T
    spacing = 1.0 / np.linalg.norm(recip, axis=1)
    reps = np.ceil(CUTOFF / spacing).astype(int)
    if np.prod(2 * reps + 1) * n > 400_000:  # pathological thin/huge cell: skip neighbours
        nn = np.full(n, CUTOFF)
        ratio = nn / (2 * cov)
        cn = np.zeros(n)
    else:
        shifts = np.stack(np.meshgrid(*[np.arange(-r, r + 1) for r in reps], indexing="ij"),
                          -1).reshape(-1, 3) @ lat
        images = (cart[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
        zimg = np.tile(np.arange(n), len(shifts))
        nn, ratio, cn = np.empty(n), np.empty(n), np.empty(n)
        for i in range(n):
            d = np.linalg.norm(images - cart[i], axis=1)
            mask = (d > 1e-4) & (d <= CUTOFF)
            if not mask.any():
                nn[i], ratio[i], cn[i] = CUTOFF, CUTOFF / (2 * cov[i]), 0
                continue
            dj, rsum = d[mask], cov[i] + cov[zimg[mask]]
            nn[i] = dj.min()
            ratio[i] = (dj / rsum).min()
            cn[i] = np.count_nonzero(dj <= 1.25 * rsum)

    cs = _crystal_system(spg)
    out = {
        "struct_nprim": nprim, "struct_nsym_sites": nsym, "struct_formula_units": fu,
        "struct_vpa": vol / n,
        "struct_density": wt.sum() * 1.66054 / vol,  # g/cm3
        "struct_packing": (4.0 / 3.0 * np.pi * cov ** 3).sum() / vol,
        "struct_vol_ratio": vol / gsv.sum(),
        "struct_spg": spg, "struct_pg_order": pg_order,
        "struct_b_a": b / a, "struct_c_a": c / a,
        "struct_angle_dev": float(np.mean(np.abs(np.array(angles) - 90.0))),
        **{f"struct_cs_{name}": float(i == cs) for i, name in enumerate(CRYSTAL_SYSTEMS)},
        "struct_nn_min": nn.min(), "struct_nn_mean": nn.mean(), "struct_nn_std": nn.std(),
        "struct_ratio_min": ratio.min(), "struct_ratio_mean": ratio.mean(),
        "struct_cn_mean": cn.mean(), "struct_cn_std": cn.std(),
        "struct_cn_min": cn.min(), "struct_cn_max": cn.max(),
    }
    return {k: float(v) for k, v in out.items()}


# ----------------------------------------------------------------------------- parsers
def arrays_from_pymatgen_dict(d: dict, sym2z: dict):
    lat = d["lattice"]["matrix"]
    frac = [s["abc"] for s in d["sites"]]
    z = [sym2z.get(max(s["species"], key=lambda x: x.get("occu", 1))["element"], 0)
         for s in d["sites"]]
    return lat, frac, z


def arrays_from_jarvis_atoms(a: dict, sym2z: dict):
    lat = np.asarray(a["lattice_mat"], dtype=float)
    coords = np.asarray(a["coords"], dtype=float)
    frac = coords @ np.linalg.inv(lat) if a.get("cartesian") else coords
    return lat, frac, [sym2z.get(e, 0) for e in a["elements"]]


def arrays_from_optimade(lattice_vectors, cartesian_positions, species, sym2z: dict):
    lat = np.asarray(lattice_vectors, dtype=float)
    frac = np.asarray(cartesian_positions, dtype=float) @ np.linalg.inv(lat)
    return lat, frac, [sym2z.get(s, 0) for s in species]


def featurize_batch(items: list, tables: dict) -> list:
    """items: list of (key, lattice, frac, z); returns list of (key, feature dict or None)."""
    out = []
    for key, lat, frac, z in items:
        try:
            out.append((key, features_from_arrays(lat, frac, z, tables)))
        except Exception:  # noqa: BLE001
            out.append((key, None))
    return out
