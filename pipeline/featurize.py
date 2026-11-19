"""Fast, vectorised composition featuriser (Magpie-equivalent).

Computes the same 132 Magpie statistics as
``matminer.featurizers.composition.ElementProperty.from_preset("magpie")``
(22 elemental properties x {minimum, maximum, range, mean, avg_dev, mode}),
plus the matminer ``Stoichiometry`` p-norms and ``ValenceOrbital`` fractions,
but on whole arrays at once so that hundreds of thousands of compositions are
featurised in seconds instead of hours.

Feature views (used by Co-Training):
  view 1  = elemental-property statistics (Magpie)
  view 2  = stoichiometry + valence-orbital + composition-complexity features
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element
from scipy import sparse

MAGPIE_PROPS = [
    "Number", "MendeleevNumber", "AtomicWeight", "MeltingT", "Column", "Row",
    "CovalentRadius", "Electronegativity", "NsValence", "NpValence", "NdValence",
    "NfValence", "NValence", "NsUnfilled", "NpUnfilled", "NdUnfilled", "NfUnfilled",
    "NUnfilled", "GSvolume_pa", "GSbandgap", "GSmagmom", "SpaceGroupNumber",
]
STATS = ["minimum", "maximum", "range", "mean", "avg_dev", "mode"]
STOICH_P = [0, 2, 3, 5, 7, 10]
MAX_Z = 103
SYMBOLS = [Element.from_Z(z).symbol for z in range(1, MAX_Z + 1)]
SYM2IDX = {s: i for i, s in enumerate(SYMBOLS)}


@lru_cache(maxsize=1)
def element_table() -> np.ndarray:
    """(n_elements x n_props) matrix of Magpie elemental properties."""
    from matminer.utils.data import MagpieData

    md = MagpieData()
    table = np.full((MAX_Z, len(MAGPIE_PROPS)), np.nan)
    for j, prop in enumerate(MAGPIE_PROPS):
        values = md.all_elemental_props[prop]
        for i, sym in enumerate(SYMBOLS):
            v = values.get(sym)
            if v is not None:
                table[i, j] = float(v)
    return table


def magpie_feature_names() -> list[str]:
    return [f"MagpieData {s} {p}" for p in MAGPIE_PROPS for s in STATS]


def view2_feature_names() -> list[str]:
    return (
        [f"{p}-norm" for p in STOICH_P]
        + ["frac s valence electrons", "frac p valence electrons",
           "frac d valence electrons", "frac f valence electrons"]
    )


def feature_names() -> list[str]:
    return magpie_feature_names() + view2_feature_names()


def parse_formula(formula: str) -> dict[str, float] | None:
    """Return {element: atomic fraction} or None if the formula is unusable."""
    try:
        comp = Composition(formula)
        if comp.num_atoms <= 0:
            return None
        frac = comp.fractional_composition.get_el_amt_dict()
        if any(el not in SYM2IDX for el in frac):
            return None
        return frac
    except Exception:
        return None


def reduced_formula(formula: str) -> str | None:
    try:
        return Composition(formula).reduced_formula
    except Exception:
        return None


def fraction_matrix(fracs: list[dict[str, float]]) -> sparse.csr_matrix:
    rows, cols, vals = [], [], []
    for i, d in enumerate(fracs):
        for el, f in d.items():
            rows.append(i)
            cols.append(SYM2IDX[el])
            vals.append(f)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(fracs), MAX_Z))


def featurize_fractions(F: sparse.csr_matrix) -> np.ndarray:
    """Vectorised Magpie + view-2 features from a (n x 103) fraction matrix."""
    P = element_table()  # (E x K)
    n, K = F.shape[0], P.shape[1]
    F = F.tocsr()
    F.sort_indices()
    coo = F.tocoo()
    r, c, f = coo.row, coo.col, coo.data
    pv = P[c]  # (nnz x K) property value of each present element

    # mean (NaN-safe: matminer would give NaN when a property is missing)
    mean = np.zeros((n, K))
    np.add.at(mean, r, f[:, None] * pv)
    absdev = np.zeros((n, K))
    np.add.at(absdev, r, f[:, None] * np.abs(pv - mean[r]))

    starts = F.indptr[:-1]
    mn = np.minimum.reduceat(pv, starts, axis=0)
    mx = np.maximum.reduceat(pv, starts, axis=0)

    # mode = property of the element with the largest fraction (ties -> smallest value)
    order = np.lexsort((-f, r))  # within each row, largest fraction first
    first = np.ones(len(order), dtype=bool)
    first[1:] = r[order][1:] != r[order][:-1]
    top_idx = order[first]  # index (into nnz) of max-fraction element per row
    max_f = f[top_idx]
    tie = f >= max_f[r] - 1e-12
    big = np.where(tie[:, None], pv, np.inf)
    mode = np.minimum.reduceat(big, starts, axis=0)

    stats = {"minimum": mn, "maximum": mx, "range": mx - mn,
             "mean": mean, "avg_dev": absdev, "mode": mode}
    magpie = np.stack([stats[s] for s in STATS], axis=2).reshape(n, K * len(STATS))

    # Stoichiometry p-norms
    ncomp = np.diff(F.indptr).astype(float)
    norms = [ncomp]
    for p in STOICH_P[1:]:
        norms.append(np.asarray(F.power(p).sum(axis=1)).ravel() ** (1.0 / p))
    # Valence-orbital fractions  (mean Ns / mean Nvalence, ...)
    idx = {p: MAGPIE_PROPS.index(p) for p in
           ["NsValence", "NpValence", "NdValence", "NfValence", "NValence"]}
    tot = mean[:, idx["NValence"]]
    with np.errstate(divide="ignore", invalid="ignore"):
        vfrac = [mean[:, idx[k]] / tot for k in
                 ["NsValence", "NpValence", "NdValence", "NfValence"]]
    view2 = np.column_stack(norms + vfrac)
    return np.hstack([magpie, view2])


def featurize_formulas(formulas: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Featurise a Series of formulas. Returns (features, mask_of_valid_rows)."""
    fracs = formulas.map(parse_formula)
    ok = fracs.notna()
    F = fraction_matrix(fracs[ok].tolist())
    X = featurize_fractions(F)
    df = pd.DataFrame(X, columns=feature_names(), index=formulas[ok].index)
    return df, ok


def view_columns() -> tuple[list[str], list[str]]:
    return magpie_feature_names(), view2_feature_names()
