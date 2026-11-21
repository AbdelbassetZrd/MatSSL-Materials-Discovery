# Extending MatSSL: molecules, toxicity and a materials Design Lab

The project started as a materials platform (see [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)).
It is being extended in three directions. All three are **built and deployed**.

| Module | What it does | Status |
|---|---|---|
| **A. Molecule Studio + toxicity** | build a molecule, get its predicted toxicity profile | **built** |
| **B. Protein-target activity** | test your molecules against the protein targets linked to an illness | **built** |
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

| Endpoint | Molecules | Positives | ROC-AUC | PR-AUC | Balanced acc. |
|---|---|---|---|---|---|
| SR-MMP (Mitochondrial toxicity) | 5,804 | 15.8 % | 0.895 | 0.705 | 0.762 |
| CT_TOX (Clinical trial toxicity) | 1,461 | 7.7 % | 0.916 | 0.490 | 0.751 |
| FDA_APPROVED (FDA approved) | 1,461 | 94.9 % | 0.902 | 0.994 | 0.679 |
| SR-p53 (p53 stress response) | 6,767 | 6.3 % | 0.863 | 0.377 | 0.644 |
| NR-AhR (Aryl hydrocarbon receptor) | 6,542 | 11.7 % | 0.842 | 0.452 | 0.685 |
| NR-Aromatase (Aromatase) | 5,815 | 5.2 % | 0.826 | 0.299 | 0.578 |
| SR-ARE (Oxidative stress (ARE)) | 5,825 | 16.2 % | 0.791 | 0.446 | 0.684 |
| NR-PPAR-gamma (PPAR-gamma) | 6,443 | 2.9 % | 0.776 | 0.310 | 0.657 |
| NR-AR-LBD (Androgen receptor (LBD)) | 6,751 | 3.5 % | 0.770 | 0.397 | 0.638 |
| NR-ER-LBD (Estrogen receptor (LBD)) | 6,948 | 5.0 % | 0.754 | 0.211 | 0.585 |
| SR-ATAD5 (DNA damage (ATAD5)) | 7,065 | 3.7 % | 0.750 | 0.229 | 0.586 |
| SR-HSE (Heat-shock response) | 6,460 | 5.8 % | 0.720 | 0.219 | 0.631 |
| NR-AR (Androgen receptor) | 7,258 | 4.2 % | 0.651 | 0.351 | 0.653 |
| NR-ER (Estrogen receptor) | 6,186 | 12.8 % | 0.621 | 0.276 | 0.587 |

Mean ROC-AUC **0.772** over the 12 Tox21 assays. Published Tox21 numbers are usually higher because they use *random*
splits; with a scaffold split the test molecules are structurally new, which is the harder and
more realistic setting. The same gap was measured on materials when the split was grouped by
formula.

`FDA_APPROVED` needed a different threshold rule. With 95 % of molecules approved, maximising F1
of the positive class degenerates to "approved for everything" (balanced accuracy exactly 0.500).
The threshold is therefore chosen on balanced accuracy whenever the positive class is the
majority, which lifts it to 0.679.

Sanity checks on molecules the models never saw as a group: aspirin and caffeine come out **low
risk**; **benzo[a]pyrene** comes out high risk and fires the **aryl hydrocarbon receptor**, the
receptor polycyclic aromatic hydrocarbons actually bind; **bisphenol A** comes out high risk and
fires the **estrogen receptor**, the mechanism it is known for.

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

## Module B — Protein-target activity with illness context (built)

**What it predicts, stated precisely:** whether a molecule is likely to be **active against a
specific protein target**, trained on measured ChEMBL bioactivities. An illness only supplies
*context* — which proteins public databases associate with it. The output is a screening ranking
for research and teaching, never a statement about treating anyone.

### The one design decision everything else follows from

A model is trained **per protein, not per illness**. There is no such thing as a labelled dataset
of "molecules that cure diabetes", but there are tens of thousands of measured
molecule-versus-protein affinities. So the chain is:

```
molecule --(one QSAR model per protein)--> activity profile over 45 proteins
         --(weighted by Open Targets evidence)--> a score for every illness
```

Three things fall out of that choice, and they are the reason it was made:

1. **"And what about illness C?" is free.** The molecule is featurised once and run through the
   45 protein models once. Every illness is then a different weighted sum over the *same* profile
   - a few multiplications, not another model. The whole 15-illness ranking returns in one call.
2. **Adding an illness costs no training at all.** It is one Open Targets query for its protein
   list; if those proteins already have models, the illness is immediately screenable.
3. **The intermediate result is the interpretable one.** "Predicted active against ACHE and
   BCHE" is a chemically checkable claim. "Might help Alzheimer's" is not.

### Data

| Source | What it gives | Size |
|---|---|---|
| Open Targets GraphQL | 15 illnesses to associated proteins + evidence score, and approved drugs | 15 illnesses, 52 proteins, 325 drugs |
| ChEMBL REST | measured activities (pChEMBL) per protein, with SMILES inline | 45 proteins with usable data |
| ZINC (reused from module A) | property-matched decoys where measured inactives are scarce | 12,188 of 60,058 rows (20.3 %) |

Labels come from pChEMBL: **>= 6.5 active, < 5 inactive, the uncertain band in between dropped.**
Research into ChEMBL benchmarks shows there is no standard cutoff and that the active/inactive
ratio swings strongly with it, so the cutoff is recorded in every per-target summary rather than
left implicit. Duplicate molecule-protein pairs are collapsed to their median pChEMBL, and SMILES
are canonicalised with the same `canonical()` helper used everywhere else in the project.

### Six honest-measurement problems, and what was done about each

These are the parts worth defending in a viva; each one was a real result that looked good and
was wrong.

**1. Decoys inflate the score.** ChEMBL mostly publishes *actives* - the literature rarely reports
what did not bind. Where a protein had too few measured inactives, property-matched ZINC molecules
(MW +/- 50, logP +/- 1) were added. But a ZINC molecule is *assumed* inactive, not measured
inactive, and it is easy to separate, so it flatters the model. The fix is not to hide the decoys
but to declare them: each model reports its **decoy fraction**, and ROC-AUC is reported twice -
over everything, and over **measured molecules only**.

**2. A measured-only score is only as good as its denominator.** Correcting for decoys raises a
second question that is easy to skip: how many measured molecules is that number computed against?
For NR3C1, fifteen. Of 1,112 measured molecules, fifteen are inactive - so its measured-only
ROC-AUC has almost nothing to rank against and comes out at 1.000. That is not a perfect model, it
is the absence of a measurement. So every measured-only score now carries the number of measured
molecules of its rarer class, and only targets with 60 or more enter any average:

| Protein | decoys | measured minority | AUC, all rows | AUC, measured only | |
|---|---|---|---|---|---|
| ATM | 0 % | 532 | 0.981 | **0.981** | trustworthy |
| ACHE (Alzheimer's) | 1 % | 417 | 0.919 | 0.914 | trustworthy |
| EGFR | 15 % | 216 | 0.994 | 0.987 | trustworthy |
| DRD2 | 27 % | 55 | 0.989 | **0.792** | thinly supported |
| TYK2 | 29 % | 22 | 0.991 | **0.836** | thinly supported |
| NR3C1 | 29 % | 15 | 1.000 | **1.000** | meaningless |

Read the *measured minority* column first. ATM is the trustworthy row: no decoys at all, 532 real
inactives, the same 0.981 either way. The rows that reach 1.000 are the ones to distrust.

**3. The model was right; the protein was wrong.** Gene symbols have to be resolved to ChEMBL
target ids before any activity can be downloaded, and the first resolver fell back to the
best-scoring name match when no target carried the symbol. For one gene of 52 that fallback fired:
**FLG** (filaggrin, linked to asthma) resolved to **CHEMBL3650, which is FGFR1** - a different
protein. ChEMBL returned 6,592 real activities, the pipeline trained a perfectly good model, and
reported it under the name *filaggrin* with a ROC-AUC of 1.000.

No training or evaluation code could have caught this: every number was correct for the data it
was given. It surfaced only because 1.000 on a protein with no known small-molecule chemistry was
implausible enough to check. All 52 resolutions were then audited against the gene symbols ChEMBL
itself records - exactly one was wrong. The resolver now requires an exact gene-symbol match and
**drops a target rather than guessing**; the bad target and its model were deleted, and asthma
keeps its three correct proteins (ADORA1, PDE4D and NR3C1, the glucocorticoid receptor that
inhaled corticosteroids act on). The lesson: a silent identifier error is invisible to every
validation metric, because the metrics never see the name.

**4. Enrichment factor against the test set is meaningless.** The first version measured "do known
actives rank above the other test molecules?" - but with ~70 % actives in the test set, the
arithmetic ceiling is 1.43x, so the metric could not say anything. Enrichment is now measured the
way a real screen would be: the actives against a **background of 20,000 random ZINC molecules**,
which is what the top 1 % of a virtual screen actually competes with. The numbers (44x to 98x)
mean something now.

**5. Scaffold leakage.** Splits are grouped by Bemis-Murcko scaffold, as in module A. Without it,
a protein's actives are often analogues of one another, so a random split puts near-copies on both
sides and the reported accuracy is a memory test.

**6. A fixed applicability-domain cutoff is arbitrary.** Rather than the usual hard Tanimoto 0.4,
held-out molecules are binned by similarity to their nearest known active and accuracy is measured
per bin, so the "unreliable" line sits where accuracy actually collapses. A molecule outside the
domain is reported as **unclear**, not as a confident low score - extrapolation is not a finding.

### Scoring

`pipeline/molecules/indications.py` is pure functions, so the API and the offline validation share
one implementation (the pattern that worked for `pipeline/industries/applications.py`):

```
illness_score = SUM_t  w_t * P(active | molecule, protein t)  /  SUM_t  w_t
                                              w = Open Targets association score
```

Proteins with no model are **excluded from the sum, not counted as zero** - a missing model is
missing information, not evidence of inactivity - and the illness instead reports its *coverage*
(how many of its proteins are modelled). The result carries its full breakdown: per-protein
probability, the evidence weight, the nearest known active with its Tanimoto similarity, and a
confidence flag. Toxicity flags from module A then downgrade an otherwise promising verdict. Three
bands: **worth testing / unclear / unlikely**.

### API and pages

- `GET /indications/diseases` - the list with protein and drug counts.
- `GET /indications/diseases/{id}` - description, proteins with evidence scores, approved drugs.
- `POST /indications/screen` - `{smiles, disease_id?}`. Without an id: **every illness ranked**.
- `POST /indications/matrix` - `{smiles: [...]}` gives a molecules x illnesses matrix.

`pages/Diseases.jsx` is the illness explorer: evidence bars per protein, model AUC where one
exists, and the already-approved drugs depicted with RDKit. `pages/Screen.jsx` is the screening
page: one molecule against all 15 illnesses as an expandable ranking, or all saved molecules as a
heat map. Both carry the same notice.

### Honest limits, stated in the UI as well as here

- A target-level prediction says nothing about absorption, dose, selectivity or safety in a person.
- Binding a protein associated with an illness is a **hypothesis**, not an effect.
- ChEMBL is biased toward published actives; decoy-based inactives are an assumption, declared
  per model.
- Many illnesses have no good druggable target, so a low score can mean "no useful protein model
  for this illness" rather than "this molecule does nothing". That is why coverage is reported.

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
