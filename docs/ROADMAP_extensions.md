# Extending MatSSL: molecules, toxicity and a materials Design Lab

The project started as a materials platform (see [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)).
It is being extended in three directions. Two are **built and deployed**; the third is specified
and comes next.

| Module | What it does | Status |
|---|---|---|
| **A. Molecule Studio + toxicity** | build a molecule, get its predicted toxicity profile | **built** |
| **B. Protein-target activity** | test your molecules against the protein targets linked to an illness | next |
| **C. Design Lab** | pick an application (battery, superconductor…) and get candidate materials | **built** |

The guiding idea is that one method serves both domains. The semi-supervised protocol written for
materials — grouped splits, an ablation, stacking, a calibrated threshold — is reused unchanged on
molecules, with **ZINC playing the role GNoME plays for materials**: a large pool of unlabeled
candidates.

---

## Module A — Molecule Studio and toxicity (built)

### Data

| Dataset | Content | Size |
|---|---|---|
| Tox21 | 12 assays (7 nuclear-receptor, 5 stress-response) | 7,823 molecules |
| ClinTox | clinical-trial toxicity and FDA approval | 1,461 molecules |
| ZINC | drug-like molecules, no labels | 249,424 molecules |

All SMILES are canonicalised with RDKit, duplicates merged, and any molecule that appears in the
labelled set is removed from the unlabeled pool (`pipeline/molecules/get_data.py`).

### Features

`pipeline/molecules/featurize_mol.py` is the molecular twin of the Magpie featuriser:

- **217 RDKit descriptors** — weight, logP, TPSA, QED, ring counts, graph indices, fragment counts;
- **2048-bit Morgan fingerprint** (radius 2), stored packed as 256 bytes per molecule;
- **Bemis–Murcko scaffold**, which groups the train/test split exactly as the reduced formula does
  for materials.

The two blocks are also the two Co-Training views: descriptors describe the whole molecule, the
fingerprint describes its substructures.

### Models

`pipeline/molecules/train_toxicity.py` trains one classifier per endpoint with the materials
recipe: scaffold-grouped split, an ablation (descriptors → + fingerprint → + Optuna → + stacking
→ + threshold), a stacked LightGBM/XGBoost/HistGB ensemble, and a threshold that maximises F1 on
out-of-fold training predictions. F1 rather than accuracy, because the endpoints carry only
3–16 % positives.

**Results** (held-out molecules, scaffold-grouped):

| Endpoint | Molecules | Positives | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| SR-MMP (mitochondrial toxicity) | 5,804 | 15.8 % | **0.895** | 0.705 |
| NR-AhR (aryl hydrocarbon receptor) | 6,542 | 11.7 % | 0.842 | 0.452 |
| NR-Aromatase | 5,815 | 5.2 % | 0.826 | 0.299 |
| SR-ARE (oxidative stress) | 5,825 | 16.2 % | 0.791 | 0.446 |
| NR-PPAR-gamma | 6,443 | 2.9 % | 0.776 | 0.310 |
| NR-AR-LBD | 6,751 | 3.5 % | 0.770 | 0.397 |
| NR-ER-LBD | 6,948 | 5.0 % | 0.754 | 0.211 |
| SR-ATAD5 (DNA damage) | 7,065 | 3.7 % | 0.750 | 0.229 |
| NR-AR | 7,258 | 4.2 % | 0.651 | 0.351 |
| NR-ER | 6,186 | 12.8 % | 0.621 | 0.276 |

Mean ROC-AUC **0.768**. Published Tox21 numbers are usually higher because they use *random*
splits; with a scaffold split the test molecules are structurally new, which is the harder and
more realistic setting. The same gap was measured on materials when the split was grouped by
formula.

### Application

- `POST /molecules/predict` returns, per endpoint, the probability, the calibrated threshold, the
  verdict, and SHAP contributions. **Fingerprint bits are decoded back into the substructure they
  stand for**, so an explanation reads "substructure cOC(C)=O" rather than "bit 1427".
- An overall risk level counts how many toxicity endpoints fire (low / moderate / high). Aspirin
  and caffeine come out low; benzo[a]pyrene, a known carcinogen, comes out moderate.
- The **Molecule Studio** page offers three ways to make a molecule: a Ketcher drawing canvas, a
  fragment builder (ring system + substituent, always valid), and a SMILES box with live RDKit
  validation and 2D depiction. Molecules can be saved to a personal library in the browser.

---

## Module B — Protein-target activity with illness context (next)

**What it predicts, stated precisely:** whether a molecule is likely to be **active against a
specific protein target**, trained on measured ChEMBL bioactivities. An illness only supplies
*context* — which proteins public databases associate with it. The output is a screening ranking
for research and teaching, never a statement about treating anyone.

### Plan

1. **Knowledge base** (`pipeline/molecules/build_disease_graph.py`): the Open Targets GraphQL API
   gives, for ~15 well-studied illnesses, their most strongly associated protein targets and an
   association score. Targets without enough ChEMBL data are dropped. Stored as
   `data/processed/diseases.json`.
2. **Activity data** (`pipeline/molecules/get_chembl.py`): ChEMBL bioactivities per target through
   its REST API. Label from pChEMBL: ≥ 6 active, < 5 inactive, the uncertain band dropped.
   Property-matched ZINC molecules fill in where measured inactives are scarce.
3. **Models**: one classifier per target, same features and scaffold-grouped split as module A,
   Self-Training where labels are few. Reported per target with ROC-AUC and PR-AUC, plus an
   enrichment factor: how far known actives rank above 1,000 random ZINC molecules.
4. **Pages**: an illness explorer (description, associated targets with their scores, known drugs)
   and a screening table that runs your saved molecules against the targets of the selected
   illness, showing per-target probability, similarity to the nearest known active, an
   applicability-domain flag, and the module A toxicity flags.

Every page carries the same notice: *in-silico screening for research and teaching — not medical
advice.* A target-level prediction says nothing about absorption, dosage or safety in a person.

---

## Module C — Design Lab (built)

"Help me create a battery / a superconductor." Pick an application, score your own composition
for it, and let the models rank candidates nobody has measured.

### New model: superconducting critical temperature

`pipeline/industries/train_superconductor.py` predicts Tc from the composition alone, using the
same 142 Magpie descriptors as the rest of the project, on the SuperCon-derived dataset of
Hamidieh (21,263 measurements, 15,295 unique compositions).

**A trap worth recording.** SuperCon lists *only* superconductors. A model trained on it alone
answers "if this superconducts, how hot?" and will happily assign a critical temperature to table
salt — the first version predicted **12.7 K for NaCl**. Adding 20,000 Materials Project
compositions that SuperCon never reports, labelled Tc = 0 K, fixed it.

| | MAE | MAE (superconductors only) | R² | AUC above 77 K |
|---|---|---|---|---|
| Stacked ensemble | **2.94 K** | 5.40 K | **0.933** | **0.990** |

Sanity check after the fix: YBa₂Cu₃O₇ 91.6 K (measured 92), HgBa₂Ca₂Cu₃O₈ 127.5 K (134),
MgB₂ 34.6 K (39), Nb₃Sn 18.2 K (18), and NaCl, SiO₂, Al₂O₃, Fe₂O₃ all ≤ 0.5 K.

### Application scores

`pipeline/industries/applications.py` turns the property models into one score per application,
and always returns the breakdown that produced it:

| Application | Built from |
|---|---|
| Superconductor | predicted Tc (full marks at 77 K, liquid nitrogen) × stability |
| Solar absorber | P(gap in 1.1–1.7 eV) × stability × (1 − P(metal)) |
| Permanent magnet | P(magnetic) × P(metal) × stability |
| Battery cathode | cathode chemistry filter × stability × (1 − ½ P(metal)) |

The battery score is deliberately **not** a voltage prediction: it checks for a mobile working
ion, a redox-active transition metal and a suitable anion, then asks the stability model whether
the compound can exist. Predicting voltage and capacity needs computed electrode pairs from the
Materials Project insertion-electrode API, which is the documented next step.

### Candidate generation

`pipeline/industries/rank_candidates.py` scores all **554,054 unlabeled GNoME compositions** — the
same pool used as unlabeled data in the semi-supervised study — and stores the best per
application. These are compositions proposed by a generative search that nobody has measured, so
this is the real discovery setting.

What the ranking finds, with no chemistry rules telling it to:

- **superconductors** → cuprates: Ba₂La(CuO₂)₄ (69 K predicted), BaSr₅(CuO₃)₄, Ba₈TmSc(CuO₃)₆ —
  the family of every high-Tc superconductor discovered since 1986;
- **battery cathodes** → vanadium fluorophosphates such as RbNaV₂P₂(O₄F)₂ — the family of
  Na₃V₂(PO₄)₂F₃, a real sodium-ion cathode;
- **permanent magnets** → rare-earth pnictides and chalcogenides;
- **solar absorbers** → caesium and rubidium chalcogenides.

A "practical elements only" filter removes radioactive and strongly toxic elements, which the
models otherwise rank happily.

---

## How to reproduce

```bash
# molecules
uv run python pipeline/molecules/get_data.py
uv run python pipeline/molecules/featurize_mol.py --unlabeled 150000
uv run python pipeline/molecules/train_toxicity.py

# design lab
uv run python pipeline/industries/train_superconductor.py
uv run python pipeline/industries/rank_candidates.py

uv run pytest -q          # includes leakage and scaffold-split checks
cd frontend && npm run build
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000   # http://127.0.0.1:8000/ui/
```

## Files

| Path | Role |
|---|---|
| `pipeline/molecules/` | data preparation, RDKit featuriser, toxicity training |
| `pipeline/industries/` | superconductor model, application scores, candidate ranking |
| `app/services/molecules.py`, `app/api/routes/molecules.py` | toxicity API |
| `app/services/design.py`, `app/api/routes/design.py` | Design Lab API |
| `frontend/src/pages/MoleculeStudio.jsx`, `MoleculeResults.jsx`, `DesignLab.jsx` | the three new pages |
| `tests/test_molecules.py`, `tests/test_design.py` | API and scoring tests |
