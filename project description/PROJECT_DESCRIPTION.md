# MatSSL — how this project was built

**ESI-SBA · Machine Learning mini-project**
Laref Ibrahim · Nemouchi Faycal Abdelhadi · Zerradi Abdelbasset · Medelci Djezzar Aymen

Repository: <https://github.com/Ibrahimlaref/ml_Mini-project>
Full report: `MatSSL_report.pdf` (69 pages) · Screenshots: `screenshots/`

---

## Contents

1. [What the project does](#1-what-the-project-does)
2. [The question it started from](#2-the-question-it-started-from)
3. [Where every piece of data came from](#3-where-every-piece-of-data-came-from)
4. [How it was built, stage by stage](#4-how-it-was-built-stage-by-stage)
5. [Results](#5-results)
6. [Every problem we hit, and what we did about it](#6-every-problem-we-hit-and-what-we-did-about-it)
7. [The web application](#7-the-web-application)
8. [How to run it](#8-how-to-run-it)
9. [What we would do next](#9-what-we-would-do-next)

---

## 1. What the project does

MatSSL predicts the properties of **materials** and **molecules** when almost none of the data is
labelled, and puts the result behind a web application anyone can use.

It has four parts, all built on one method:

| Part | What you give it | What it tells you |
|---|---|---|
| **Materials** | a chemical formula, e.g. `LiFePO4` | is it metallic, magnetic, stable, does it have a solar band gap |
| **Toxicity** | a molecule you draw or paste | predicted activity on 12 Tox21 toxicity assays + clinical toxicity |
| **Illness screening** | a molecule | which of 15 illnesses it is worth testing against, and why |
| **Design Lab** | an application (battery, superconductor…) | candidate materials ranked from 554,054 unmeasured compositions |

The common thread is **semi-supervised learning**: we have a few thousand labelled examples and
hundreds of thousands of unlabelled ones, and we ask whether the unlabelled pool can be made to
help.

---

## 2. The question it started from

Labelling a material means running a DFT calculation; labelling a molecule means running an
assay. Both are expensive. Unlabelled candidates, by contrast, are nearly free — Google DeepMind
released 554,054 new crystal compositions (GNoME), and ZINC lists hundreds of thousands of
purchasable drug-like molecules.

So the research question is simple to state:

> **Given 1,000 labels and 100,000 unlabelled examples, can semi-supervised learning beat
> ordinary supervised learning on the 1,000?**

We answer it on **three independent domains** — inorganic materials, molecular toxicity, and
protein-ligand activity — using the same protocol each time. Running one experiment three times
on different science is worth more than running it once, because it shows whether the answer is a
property of the method or an accident of the dataset.

---

## 3. Where every piece of data came from

Nothing here is synthetic. Every number traces back to a public database.

### Materials

| Dataset | Source | Size | What we take from it |
|---|---|---|---|
| **Materials Project** | [materialsproject.org](https://materialsproject.org) API | 154,718 entries | band gap, magnetisation, energy above hull, crystal structure |
| **GNoME** | Google DeepMind, released 2023 | **554,054** compositions | the unlabelled pool — predicted-stable crystals nobody has measured |
| **Alexandria** | OPTIMADE API | on demand | crystal structures for the 3D viewer, cached offline |
| **SuperCon** | Hamidieh (2018), UCI | 21,263 measurements, 15,295 unique | superconducting critical temperature |

Labels after cleaning:

| Task | Labelled examples | Positive rate |
|---|---|---|
| Metallicity | 77,432 | 47.7 % |
| Magnetism | 64,410 | 20.6 % |
| Stability (hull distance) | 436,791 | 29.5 % |
| PV band gap (1.1–1.7 eV) | 78,164 | 6.7 % |

### Molecules

| Dataset | Source | Size | What we take from it |
|---|---|---|---|
| **Tox21** | MoleculeNet / NIH | 7,823 molecules | 12 toxicity assays (7 nuclear-receptor, 5 stress-response) |
| **ClinTox** | MoleculeNet | 1,461 molecules | clinical trial toxicity, FDA approval |
| **ZINC250k** | ZINC database | **249,424** molecules | the unlabelled pool, and property-matched decoys |
| **ChEMBL** | EBI REST API | 58,167 activities | measured molecule-protein affinities (pChEMBL) |
| **Open Targets** | GraphQL API | 15 illnesses, 52 proteins, 325 drugs | which proteins the evidence links to which illness |

### How features are computed

- **Materials**: 142 **Magpie** composition descriptors (element statistics — electronegativity,
  radius, valence electrons…), plus 28 crystal-structure descriptors computed with `spglib`
  (symmetry, packing, cell geometry) for the high-accuracy track.
- **Molecules**: 217 **RDKit** descriptors + a 2048-bit **Morgan fingerprint** (radius 2).

---

## 4. How it was built, stage by stage

### Stage 1 — the semi-supervised study (the core experiment)

Five methods, compared under identical conditions:

| Method | Idea |
|---|---|
| Baseline | supervised learning on the labelled data only |
| Label Propagation | spread labels through a k-NN graph |
| Label Spreading | same, with class-mass normalisation |
| Self-Training | model labels its own confident predictions, retrains |
| Co-Training | two models on two feature views teach each other |
| Cluster-then-Label | cluster everything, label by cluster majority |

Run for every task × every unlabelled pool × multiple label budgets × multiple seeds — roughly
**2,700 benchmarked models**.

**The protocol matters more than the methods.** Two decisions keep the numbers honest:

- **Grouped splits.** Materials are split by *reduced formula*, molecules by *Bemis–Murcko
  scaffold*. Without this, `Fe2O3` and `Fe4O6` land on opposite sides of the split and the model
  is tested on what it memorised. This single change lowered our accuracies and made them real.
- **Thresholds calibrated out-of-fold.** We never pick a decision threshold on the test set.

### Stage 2 — the high-accuracy track

A second track asks a different question: *forget the label budget, what is the best we can do?*
All labels, crystal-structure descriptors, Optuna tuning, and a **stacked ensemble** (LightGBM +
XGBoost + HistGradientBoosting feeding a logistic meta-learner on out-of-fold predictions).

### Stage 3 — molecules and toxicity

The same pipeline, re-pointed at molecules. ZINC plays the role GNoME plays for materials.

### Stage 4 — illness screening

The part that needed the most thought. **No model here predicts that a molecule cures a disease.**
That label does not exist. What does exist, in quantity, is measured affinity between a molecule
and a single protein.

So the chain is:

```
molecule → one QSAR model per protein → activity profile over 45 proteins
        → weighted by Open Targets evidence → a score for every illness
```

**The key design decision is to train per protein, not per illness.** Three things follow:

1. Asking about a second and third illness is free — the molecule is scored once, and each
   illness is a different lookup over the *same* profile.
2. Adding an illness costs **no training at all** — one database query for its protein list.
3. The intermediate result is the interpretable one: "predicted active against ACHE" is a
   chemically checkable claim; "might help Alzheimer's" is not.

The score is the **best evidence-weighted protein**:

```
score(molecule, illness) = max over proteins of  evidence(protein) × P(active | molecule)
```

Section 6 explains why this is a maximum and not an average — it was the single biggest fix in
the project.

### Stage 5 — the Design Lab

Inverts the question: instead of scoring a composition you supply, it scores **all 554,054** GNoME
compositions for an application and ranks them.

---

## 5. Results

### The semi-supervised question, answered

| Domain | Pool | Best SSL method | Gain over supervised |
|---|---|---|---|
| Materials — stability | in-distribution | Co-Training | **+0.73 %** |
| Materials — magnetism | in-distribution | Self-Training | +0.30 % |
| Materials — metallicity | in-distribution | Self-Training | +0.01 % |
| Materials — magnetism | **GNoME** | — | **−1.25 %** (it hurt) |
| Protein activity (4 proteins) | ZINC | Self-/Co-Training | +0.00 % to +0.43 % |

**The honest summary: semi-supervised learning buys a fraction of a percent when it works, and
nothing when it does not.** On 2 of the 4 proteins no SSL variant beat plain supervised learning.

The negative result is the most instructive one. The GNoME pool *hurt* magnetism prediction by
1.25 %, because GNoME crystals are not drawn from the same chemistry as the labelled set. **An
unlabelled pool only helps when it matches the distribution of the labels** — and that is a
finding, not a failure.

### Deployed model accuracy (high-accuracy track)

| Task | Accuracy | Balanced acc. | ROC-AUC |
|---|---|---|---|
| Magnetism | **94.6 %** | 91.6 % | 0.983 |
| PV band gap | 93.5 % | 58.9 % | 0.906 |
| Metallicity | 91.5 % | 91.1 % | 0.969 |
| Stability | 90.3 % | 87.0 % | 0.961 |

Note the band gap row: 93.5 % accuracy looks excellent, but only 6.7 % of materials are positive,
so **balanced accuracy of 58.9 % is the honest number**. We report both everywhere.

### Toxicity — 14 endpoints, mean ROC-AUC 0.791

| Best | | Weakest | |
|---|---|---|---|
| Clinical toxicity | 0.916 | Oestrogen receptor | 0.621 |
| FDA approval | 0.902 | Androgen receptor | 0.651 |
| Mitochondrial toxicity | 0.895 | Heat-shock response | 0.720 |

Sanity checks pass: benzo[a]pyrene fires the aryl-hydrocarbon receptor, bisphenol A fires the
oestrogen receptor — both correct known mechanisms.

### Illness screening — 45 protein models

- 58,167 measured activities, 15 illnesses, 52 associated proteins
- Mean ROC-AUC **0.976** over all held-out molecules, **0.965** over measured molecules only
- Enrichment 11× to 100× against a 20,000-molecule ZINC background

**Validated on 163 approved drugs** by checking where each drug's own illness lands in its
predicted ranking:

| Group | Drugs | Median rank | Top-3 |
|---|---|---|---|
| Illnesses with human protein targets | 126 | 5.9 / 15 | **41 %** |
| — Depression | 12 | **1.0** / 15 | 83 % |
| — Alzheimer's disease | 12 | 3.0 / 15 | 83 % |
| Infectious disease | 37 | 12.1 / 15 | **0 %** |
| *Random baseline* | — | *8.0 / 15* | *20 %* |

The infectious-disease result is **0 % across all 37 drugs** — worse than random — and it is not
a modelling failure. Open Targets links an illness to its **human** proteins, but an
antiretroviral inhibits HIV protease, an antimalarial acts on the parasite, a TB drug on
mycobacterial enzymes. None of those is human, so none can be in the panel. The models correctly
answer that these drugs do not bind human proteins they were never designed to bind.

**This bounds the method by measurement rather than assertion: it applies to illnesses treated by
modulating a human protein, and not to infectious disease.**

### Superconductors

MAE **2.94 K**, R² **0.933**, ROC-AUC **0.990** for "superconducts above liquid nitrogen". Left to
rank 554,054 unmeasured compositions, it returns **cuprates** for superconductors and **vanadium
fluorophosphates** for battery cathodes — the families real high-Tc superconductors and sodium-ion
cathodes actually belong to. Nobody told it those families; it found them.

---

## 6. Every problem we hit, and what we did about it

This is the most useful section of the document. Most of these were results that **looked good and
were wrong**.

### A. Problems of honest measurement

**1. Data leakage through chemical families.**
A random train/test split put `Fe2O3` in training and `Fe4O6` in test — the same chemistry, so the
model scored well by memorising. *Fix:* split by reduced formula (materials) and Bemis–Murcko
scaffold (molecules) using `StratifiedGroupKFold`, so a whole chemical family falls on one side.
Our accuracies dropped, and became real.

**2. A model that "predicted" superconductivity in table salt.**
Our Tc model gave NaCl 12.7 K. The cause: SuperCon contains *only superconductors*, so the model
had never seen a non-superconductor and had no concept of one. *Fix:* add 20,000 Materials Project
compositions as Tc = 0 K negatives. NaCl dropped to 0.2 K and the MAE improved to 2.94 K. **A
dataset of only positives cannot teach a model what negative looks like.**

**3. An enrichment factor that could not say anything.**
We measured whether known actives rank above the other test molecules — but with ~70 % actives in
the test set, the arithmetic ceiling is 1/0.7 = **1.43×**. The metric was incapable of
distinguishing a good model from a bad one. *Fix:* measure against a **background of 20,000 random
ZINC molecules**, as a real virtual screen does. Values became 11×–100× and meaningful.

**4. Decoy-inflated ROC-AUC.**
ChEMBL publishes what *binds* far more often than what does not, so for many proteins we had to
add property-matched ZINC molecules as presumed inactives. But those are *assumed* inactive, not
measured, and they are easy to separate — so they flatter the model. *Fix:* report ROC-AUC
**twice**, over everything and over measured molecules only, and record each model's decoy
fraction. DRD2 fell from 0.989 to 0.792; SLC6A4 from 0.993 to 0.845.

**5. A perfect score that was really a missing measurement.**
NR3C1 then reported ROC-AUC **1.000** on measured molecules — the metric we had just added to be
honest. The reason: of its 1,112 measured molecules, **fifteen** are inactive. There was nothing
to rank against. *Fix:* every measured-only score now carries **the number of measured molecules
of its rarer class**. Below 15 it is withheld (4 proteins; EIF2AK4 has *none at all*), at or above
60 the protein is "well supported" and enters the average (26 proteins), and the 15 in between are
reported with a warning marker. **A ranking metric is only as good as its rarer class.**

**6. A degenerate threshold that predicted "approved" for everything.**
The FDA-approval dataset is 95 % positive. Optimising the decision threshold for F1 produced a
model that said *approved* for every molecule — F1 looked fine, and balanced accuracy was exactly
0.500. *Fix:* maximise **balanced accuracy** when the positive class is the majority. Result:
0.679, which is modest and true.

**7. 93.5 % accuracy that means almost nothing.**
Only 6.7 % of materials have a photovoltaic band gap, so predicting "no" always scores 93.5 %.
*Fix:* report balanced accuracy beside accuracy everywhere. The honest figure is 58.9 %.

### B. Problems of correctness

**8. The model was right; the protein was wrong.**
Our gene-symbol resolver searched ChEMBL by name and, when no target carried the symbol, fell back
to the best-scoring match. For one gene out of 52 that fallback fired: **FLG** (filaggrin, linked
to asthma) resolved to **CHEMBL3650 — which is FGFR1**, a completely different protein. ChEMBL
returned 6,592 genuine activities, a perfectly good model trained on them, and it was reported
under the name *filaggrin* with a ROC-AUC of 1.000.

No training or evaluation code could have caught this, **because the metrics never see the name**.
It surfaced only because 1.000 on a protein with no known small-molecule chemistry was implausible
enough to check. We then audited all 52 resolutions against the gene symbols ChEMBL itself
records — exactly one was wrong. *Fix:* require an exact gene-symbol match and **drop a target
rather than guess**. Asthma keeps its three correct proteins.

> **The lesson worth taking from this project: a silent identifier error is invisible to every
> validation metric. An implausibly good score is a reason to audit the inputs, not to celebrate.**

**9. A selective drug punished for being selective.**
The illness score began as the evidence-weighted *mean* over an illness's proteins. Screened
against our 15 illnesses, **donepezil** — the standard Alzheimer's drug — got Alzheimer's **0.25,
"unlikely"**, ranked below depression, even though the model identified its real target ACHE at
**99 %** with an exact structural match:

| Protein | evidence | P(active) | product |
|---|---|---|---|
| ACHE | 0.75 | 99 % | **0.74** |
| ACE | 0.70 | 6 % | 0.04 |
| APP | 0.68 | 1 % | 0.01 |
| CDK5 | 0.60 | 4 % | 0.02 |
| | | **mean** | 0.25 → *unlikely* |
| | | **max** | **0.74 → worth testing** |

The error is conceptual. A mean asks *does this molecule engage the whole panel?* Almost no drug
does, and the ones that come closest are promiscuous compounds that fail for that very reason.
**Selectivity is how a drug is supposed to work, and the mean penalises it hardest.** *Fix:* the
score is the best evidence-weighted protein. Donepezil's Alzheimer's: 0.74, *worth testing*, via
ACHE (see `screenshots/16_screen.png`).

We also considered a **noisy-OR**, `1 − ∏(1 − wₜpₜ)`, and rejected it: it assumes the per-protein
predictions are independent, but they are all computed from the same molecule's fingerprint, so
they are correlated by construction. Four proteins at p = 0.3 would combine to 0.61 — "worth
testing" on no good grounds.

**10. Six wrong disease identifiers.** Hypertension, tuberculosis, malaria, HIV, rheumatoid
arthritis and obesity all returned "not found" from Open Targets. *Fix:* resolve them through the
API's own `search` query to get correct MONDO identifiers.

**11. A stability bug hidden by falsy zero.** `value or 9` treated a hull distance of exactly
`0.0` — a perfectly stable material — as missing, and substituted 9. *Fix:* test for `None`
explicitly.

**12. Numeric overflow in molecular descriptors.** Some RDKit descriptors exceed the float32 range,
becoming `inf` on cast and breaking scikit-learn. *Fix:* replace non-finite values, clip to
±3 × 10³⁸, cast, then check again.

**13. An invalid LaTeX macro name.** `\TgtValTop1` contains a digit, which LaTeX does not allow in
command names — the report would not have compiled. *Fix:* rename to `\TgtValTopOne`, then scan all
264 generated macros to confirm none of the others had the problem.

**14. The knowledge base was never pushed.** `data/processed/diseases.json` was covered by a
blanket `data/processed/*` ignore rule, so anyone cloning the repository got an app with zero
illnesses. It is 79 KB of curated metadata, not a dataset. *Fix:* a targeted exception, with the
reasoning written in the `.gitignore` itself.

### C. Problems in the web application

**15. A blank Molecule Studio page.** Four bugs in sequence, each revealed by fixing the last:
`global is not defined` (the chemistry editor's bundled dependencies expect Node) → added a Vite
`define`; then `require('raphael')` → wrote a small shim; then the WebAssembly worker was not
emitted → imported from the package's `binaryWasm` entry point; then a race where the editor
initialising wiped the SMILES box → made the canvas draw-only with explicit "Use what I drew" and
"Load current molecule" buttons. The 8 MB engine now loads in the background without blocking
anything.

**16. An empty toxicity radar chart.** Two pairs of assay titles collapsed to identical axis
labels, so the chart silently drew nothing. *Fix:* use the unique endpoint codes.

**17. A feature-count mismatch (2265 vs 2249).** The ZINC background was not passed through the
same preprocessing as the training data. *Fix:* one shared `transform()` used by both.

**18. The screening page ignored its URL parameter.** `?smiles=...` worked on the toxicity page but
not the screening page, so a screening result could not be linked or shared. *Fix:* read the
parameter on both.

**19. The least trustworthy protein looked like the best one.**
Found by looking at the finished Illnesses page (`screenshots/15_illnesses.png`). GLP1R displayed
a confident green **AUC 1.00** with no warning — while GCK next to it showed 0.88 with a warning
marker. Exactly backwards: GLP1R's honest score had been *withheld* for having only 14 measured
molecules of its rarer class, so the page fell back to the decoy-inflated figure and, having no
honest number to qualify, displayed it unmarked.

*Fix:* three visual states instead of two — green where the score is well supported, amber where
it rests on too few molecules, and **red where it could not be computed at all and what is shown
includes presumed-inactive molecules**. The denominator `n` is now always displayed. **A fallback
value must never be allowed to look more confident than the real one.**

### D. Problems of process

**19. Ten concurrent download processes.** Repeated background launches of the ChEMBL downloader
left ten copies running against the same API. *Fix:* killed them all; the downloader caches
completed targets, so restarting is cheap and safe.

**20. Silent background jobs.** Long training runs wrote nothing to their log because the output
was buffered. *Fix:* run Python unbuffered (`-u`) and write directly to a log file rather than
through a pipe.

**21. Training that appeared to hang for six hours.** Timestamps in the log jumped from 19,174 s
to 40,060 s. The machine had gone to sleep. The job resumed cleanly and lost nothing — worth
recording so the gap in the logs is not mistaken for a stall.

---

## 7. The web application

**MatSSL Studio** — FastAPI backend, React + Vite frontend, one Docker image.

| Page | Screenshot | What it shows |
|---|---|---|
| Overview | `01_overview.png` | key figures, pipeline, deployed models |
| Composition Studio | `02_composition_studio.png` | periodic table, formula editor, atomic fractions |
| Crystal viewer | `03_crystal_viewer.png` | 3D structure from the Alexandria OPTIMADE API |
| Results | `04_results.png`, `05_results_magnet.png` | calibrated gauges, applicability domain, SHAP, nearest known materials |
| Compare | `06_compare.png` | up to four compositions side by side |
| Research Lab | `07`–`12_lab_*.png` | every experiment: datasets, SSL comparison, model zoo, band gap GA, unsupervised, accuracy |
| Molecule Studio | `13_molecule_studio.png` | drawing canvas, fragment builder, SMILES box, drug-likeness |
| Toxicity | `14_toxicity.png` | 12 Tox21 gauges, risk badge, SHAP naming the substructure behind each bit |
| Illnesses | `15_illnesses.png` | 15 illnesses, their proteins with evidence bars and model quality, approved drugs |
| Screen | `16_screen.png` | every illness ranked for your molecule, expandable to the per-protein breakdown |
| Design Lab | `17_design_lab.png` | application scores + ranked candidates from the GNoME pool |

Every page of the illness module carries the same notice: **in-silico screening for research and
teaching — not medical advice.**

---

## 8. How to run it

```bash
# install (uv manages Python 3.12 and every dependency)
uv sync

# start the application
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
# then open http://127.0.0.1:8000/ui/

# tests and linting
uv run pytest -q          # 43 tests
uv run ruff check app pipeline tests report
```

To rebuild everything from the raw data:

```bash
uv run python pipeline/get_data.py && uv run python pipeline/featurize.py
uv run python pipeline/run_ssl_experiments.py
uv run python pipeline/train_high_accuracy.py

uv run python pipeline/molecules/get_data.py && uv run python pipeline/molecules/featurize_mol.py
uv run python pipeline/molecules/train_toxicity.py

uv run python pipeline/molecules/build_disease_graph.py
uv run python pipeline/molecules/get_chembl.py
uv run python pipeline/molecules/train_targets.py
uv run python pipeline/molecules/guard_target_metrics.py
uv run python pipeline/molecules/validate_indications.py

uv run python pipeline/industries/train_superconductor.py
uv run python pipeline/industries/rank_candidates.py

uv run python report/make_results.py      # regenerates every number in the report
```

Every figure and number in the report is generated by `report/make_results.py` from the result
files — **no number is typed by hand.**

---

## 9. What we would do next

- **Graph neural networks** on the crystal structure and the molecular graph, instead of
  hand-engineered descriptors.
- **Active learning**: instead of asking whether unlabelled data helps passively, use the model to
  choose which 100 materials to label next.
- **Non-human protein targets**, which would bring infectious disease inside the method's reach —
  the one limitation the validation proved we have.
- **Battery-specific properties** (voltage, capacity) from the Materials Project insertion
  electrode API, to make the Design Lab's battery recommendation quantitative rather than
  heuristic.
