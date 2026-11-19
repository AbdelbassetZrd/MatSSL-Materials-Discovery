# How we raised the accuracy of the deployed models

**MatSSL – Materials Discovery with Semi-Supervised Learning** · ESI-SBA Machine Learning mini-project
Laref Ibrahim · Nemouchi Faycal Abdelhadi · Zerradi Abdelbasset · Medelci Djezzar Aymen

This document explains how we improved the accuracy of the four deployed classifiers (metallicity, magnetism, thermodynamic stability, photovoltaic band gap). For each method it says what we did, why it helps, how we implemented it, and how we checked that the gain is real and not caused by data leakage.

---

## 1. Two separate tracks

The project now has two experimental tracks. They answer two different questions, so we keep them apart.

| | **Track A – SSL study** (core of the assignment) | **Track B – high-accuracy track** (this document) |
|---|---|---|
| Question | Can unlabeled data replace labels? | What is the best accuracy we can deploy? |
| Labels | 1,000 per task | **all** labels |
| Input | chemical formula only (142 Magpie descriptors) | formula **+ crystal structure** (142 + 28 descriptors) |
| Rows | one per formula | one per **crystal structure** |
| Models | 5 SSL methods vs tuned baselines | tuned LightGBM/XGBoost + **stacked ensemble** |
| Decision threshold | 0.5 (0.22 for the band gap) | chosen to **maximise accuracy** on training data |

Track A is unchanged. Its conclusions (SSL gives small gains, up to +0.73 % on stability; distribution shift hurts) are still the main result of the report. Track B shows how far accuracy can go when the 1,000-label restriction is lifted and the crystal structure is used.

## 2. Starting point

Accuracy of the models deployed after Track A, on held-out test sets:

| Task | Model | Accuracy | ROC-AUC |
|---|---|---|---|
| Metallicity | Self-Training (LightGBM) | 88.8 % | 0.951 |
| Magnetism | Self-Training (LightGBM) | 91.7 % | 0.971 |
| Stability | Co-Training (LightGBM) | 74.5 % | 0.843 |
| PV band gap | GA descriptors + Random Forest | 91.4 % | 0.903 |

## 3. Why the accuracy was limited

1. **Formula only.** A formula such as TiO₂ describes several polymorphs (rutile, anatase, brookite). These polymorphs can have different band gaps and different stabilities, and the formula cannot tell them apart.
2. **Labels averaged over polymorphs.** To get one row per formula, Track A merged polymorphs: majority vote for metallicity, median gap for the band gap, minimum hull distance for stability. This creates label noise for every formula whose polymorphs disagree.
3. **Few labels.** Track A uses 1,000 labels on purpose. The label-budget curves of the report show that every model keeps improving up to 5,000 labels and beyond.
4. **A single model** with a default threshold of 0.5. That threshold is not the one that maximises accuracy when the classes are imbalanced (magnetism 21 %, stability 30 %, band gap 7 % positives).

Each of the methods below fixes one of these four limitations.

---

## 4. The methods

### Method 1 – One row per crystal structure (fixes limitation 2)

Instead of merging polymorphs, every crystal structure of each database is kept with its **own** label:

| Task | Source (raw files already downloaded) | Label of one structure | Structures |
|---|---|---|---|
| Metallicity | Materials Project (Matbench `mp_is_metal`) | `is_metal` | 104,209 |
| PV band gap | Materials Project (Matbench `mp_gap`) | 1.1 ≤ PBE gap ≤ 1.7 eV | 104,209 |
| Magnetism | JARVIS-DFT 3D | \|total magnetic moment\| ≥ 0.5 μB | 87,671 |
| Stability | Alexandria PBE 3D | energy above hull ≤ 0.05 eV/atom | 499,654 |

- The raw JSON files hold up to 1.5 GB of uncompressed text, and the computer has little free RAM. They are therefore **streamed**: entries are decoded one at a time with `json.JSONDecoder.raw_decode`, so memory stays at a few MB.
- Near-identical duplicates are removed. These are rows with the same formula, space group, cell size and volume.

Code: `pipeline/build_structure_dataset.py`.

### Method 2 – Crystal-structure descriptors (fixes limitation 1)

For every structure we compute **28 descriptors** that capture symmetry, geometry and bonding. We use only `spglib` (symmetry analysis) and `numpy` (a periodic neighbour search within 5 Å). This makes the computation fast: about 10 ms per structure, run in parallel on 8 cores.

| Group | Descriptors | Physical meaning |
|---|---|---|
| Symmetry | space-group number, point-group order, crystal system (7 one-hot), number of symmetry-distinct sites | high symmetry ↔ simple, often stable structures; crystal system ↔ electronic structure |
| Cell content | atoms in the primitive cell, formula units per cell | structural complexity |
| Density / packing | volume per atom, density (g/cm³), packing fraction (covalent spheres), cell volume ÷ sum of elemental volumes | compression or expansion relative to the elements ↔ bonding type and strength |
| Lattice shape | b/a, c/a (standardised cell), mean deviation of the angles from 90° | anisotropy, layered or chain structures |
| Bond lengths | shortest and mean nearest-neighbour distance, their spread, shortest and mean distance ÷ (sum of covalent radii) | short bonds relative to radii ↔ strong covalent bonding; long ones ↔ metallic or ionic character |
| Coordination | mean, spread, min and max coordination number (neighbours within 1.25 × the sum of covalent radii) | close-packed metals (CN 8–12) vs. tetrahedral or octahedral semiconductors (CN 4–6) |

**Design choices that matter:**

- **Cell-choice invariance.** The same crystal can be stored as a primitive or as a conventional cell. Every descriptor is computed from cell-independent quantities: the spglib standardised lattice, the primitive-cell mapping and the symmetry-equivalent atoms. A unit test checks that NaCl gives identical descriptors in both cells (`tests/test_structure.py`).
- **No label leakage.** Energies, formation energies, hull distances, magnetic moments and band gaps are **never** used as inputs. They are only used to build the labels. A unit test forbids any descriptor whose name refers to these quantities.
- **Same code for training and for the application.** The web application fetches the lowest-energy structure from the Alexandria OPTIMADE API and computes the descriptors with exactly the same function (`pipeline/structure_features.py`).

The final feature vector has **170 dimensions**: 142 composition descriptors (Magpie statistics, stoichiometry, valence orbitals) plus 28 structure descriptors.

### Method 3 – Use all labels (fixes limitation 3)

All training structures are used: about 83,000 for metallicity and the band gap, 70,000 for magnetism and about 400,000 for stability. That is 70 to 400 times more than in Track A.

### Method 4 – Hyper-parameter tuning with Optuna

- **Algorithm:** Bayesian optimisation with the Tree-structured Parzen Estimator (TPE) sampler.
- **Models tuned:** LightGBM (15–30 trials) and XGBoost (10–20 trials).
- **Validation:** 3-fold `StratifiedGroupKFold` on a training sample of at most 60,000 structures. The folds are also grouped by formula, so validation is as strict as the final test.
- **Objective:** mean validation **log-loss**. We tune the probabilities, not the accuracy, because accuracy at a fixed threshold is a noisy objective. The threshold is optimised separately (Method 6).
- **Search space (LightGBM):** number of trees 300–1500, learning rate 0.02–0.15 (log scale), leaves 31–255, minimum samples per leaf 5–80, feature fraction 0.3–1, row fraction 0.5–1, L2 regularisation 10⁻³–10.
- **Search space (XGBoost):** analogous, with depth 4–12 instead of the number of leaves.

### Method 5 – Stacked ensemble (fixes limitation 4)

Different learners make different errors, and combining them reduces the variance of the prediction. We use **stacking**:

1. **Base learners:**
   - tuned LightGBM;
   - tuned XGBoost;
   - histogram gradient boosting (scikit-learn).

   We also tried **Extra Trees** as a fourth base learner. We removed it: the fully grown trees took **447 MB of RAM** (95 % of the ensemble), the stacker gave it only a small weight (0.17), and it barely changed the accuracy (see section 6). The deployed models are 3–4 MB.
2. **Out-of-fold predictions.** The training set is split into 3 folds grouped by formula. Each base learner is trained on 2 folds and predicts the third. Every training structure thus gets a probability from each learner that was produced *without having seen it*.
3. **Meta-learner.** A logistic regression is trained on the **logits** of these out-of-fold probabilities:

   $$p_{\text{final}} = \sigma\Big(w_0 + \sum_k w_k \,\log\frac{p_k}{1-p_k}\Big)$$

   It learns how much to trust each base learner.
4. **Final fit.** The base learners are re-trained on the full training set. At prediction time, their probabilities go through the meta-learner.

Training the meta-learner on in-sample predictions would make it trust over-fitted learners. Using out-of-fold predictions avoids that.

Code: `pipeline/hiacc_model.py` (class `StackedEnsemble`) and `pipeline/train_high_accuracy.py`.

### Method 6 – Accuracy-optimal decision threshold (fixes limitation 4)

A classifier outputs a probability *p*. The class is 1 if *p* ≥ *t*. The value *t* = 0.5 maximises accuracy only if the probabilities are perfectly calibrated. We therefore choose

$$t^\* = \arg\max_{t \in [0.15,\,0.85]} \ \text{Accuracy}\big(y_{\text{train}},\ \mathbb{1}[p^{\text{OOF}}_{\text{train}} \ge t]\big)$$

on the **out-of-fold predictions of the training set**. The test set is never used to choose *t*. It is only used once, at the end, to report the result.

---

## 5. Evaluation protocol and how we avoided cheating

High accuracies are easy to obtain dishonestly. These are the rules we followed:

| Risk | What could inflate the accuracy | What we did |
|---|---|---|
| **Polymorph leakage** | Several structures of the same formula in train *and* test: the model "recognises" the composition | Train/test split **grouped by reduced formula** (`StratifiedGroupKFold`, 80/20, seed 42); the script checks that the intersection is empty |
| Tuning on the test set | Choosing hyper-parameters or the threshold with test data | Optuna and the threshold use **training data only** (grouped CV / out-of-fold) |
| Label leakage through features | Using energy, hull distance, magnetic moment or gap as an input | Excluded by design and checked by a unit test |
| Misleading metric | The band gap has 7.3 % positives: "always no" already scores 92.7 % accuracy | We also report **balanced accuracy**, F1 and ROC-AUC |
| Cherry-picking | Reporting only the best step | **Ablation**: every step is evaluated on the *same* held-out test set |

The test set holds 20,000 structures per task (stratified), or the whole held-out fold when it is smaller.

---

## 6. Results – ablation on the held-out test set

*The numbers come from `data/predictions/high_accuracy/{task}_summary.json` and `{task}_ablation.csv`.*

**Test accuracy after each step:**

| Task (test structures) | 1. Composition only | 2. + structure | 3. + tuning | 4. + stacking | 5. + threshold (= final) |
|---|---|---|---|---|---|
| Metallicity (20,000) | 90.3 % | 91.4 % | 91.3 % | 91.5 % | **91.5 %** |
| Magnetism (17,535) | 93.4 % | 94.5 % | 94.3 % | 94.6 % | **94.6 %** |
| Stability (20,000) | 85.4 % | 89.8 % | 88.7 % | 90.3 % | **90.3 %** |
| PV band gap (20,000) | 93.5 % | 93.8 % | 93.3 % | 93.6 % | **93.5 %** |

**Final models (all metrics):**

| Task | Accuracy | Balanced accuracy | F1 | ROC-AUC | Threshold | Gain vs. composition only |
|---|---|---|---|---|---|---|
| Metallicity | **91.5 %** | 91.1 % | 0.901 | 0.969 | 0.50 | +1.2 points |
| Magnetism | **94.6 %** | 91.6 % | 0.874 | 0.983 | 0.51 | +1.2 points |
| Stability | **90.3 %** | 87.0 % | 0.818 | 0.961 | 0.50 | **+4.9 points** |
| PV band gap | **93.5 %** | 58.9 % | 0.292 | 0.906 | 0.56 | 0 points |

### What the ablation teaches us

1. **The crystal structure is the main ingredient.**
   - Stability gains +4.3 points from the structure descriptors alone.
   - Its five most important descriptors are all structural: atoms in the primitive cell, symmetry-distinct sites, shortest bond relative to covalent radii, mean coordination number and spread of bond lengths.
   - Metallicity and magnetism gain about +1.1 points each.
2. **Tuning did not help accuracy.** The Optuna-tuned LightGBM is slightly *below* the default one on every task (−0.1 to −1.0 points). It was tuned on log-loss over a sub-sample of at most 60,000 structures, and that optimum did not carry over to the full training set. We report this as it is rather than hiding the step.
3. **Stacking helps a little.** It adds +0.1 to +0.5 points over the best single model.
4. **The accuracy-optimal threshold** ends up at about 0.50 for the three balanced tasks, so it changes nothing. For the band gap it moves to 0.56 and costs 0.1 points on the test set: the threshold is chosen on training data only, and that is the honest outcome.
5. **The photovoltaic band gap stays hard.**
   - Only 7.3 % of structures are positive. "Always no" already scores 92.7 %.
   - Our 93.5 % is therefore only slightly better than that baseline, as the balanced accuracy of 58.9 % shows.
   - The ROC-AUC of 0.906 shows the model *ranks* materials well. It is useful for screening, where the top-ranked candidates are inspected first, but it is not a reliable yes/no classifier.
6. **Extra Trees was removed.** With it as a fourth base learner, metallicity scored 91.7 % instead of 91.5 % and magnetism 94.5 % instead of 94.6 %, for 450 MB of memory. Not worth it.

Step 1 is a LightGBM with default settings (the model family that won Track A), using composition descriptors only and trained on all structures. The comparison between steps 1 and 5 is fair: same test set, same split, same labels.

Track A's accuracies (section 2) were measured per *formula*, while Track B measures per *structure* on a different test set. The fair before/after comparison is therefore **step 1 against step 5 within Track B**.
## 7. Why we do not reach 98 %

98 % accuracy on all four tasks is not reachable with an honest evaluation. The reasons:

1. **Label noise at the decision boundary.** The labels are thresholds on DFT numbers: hull distance ≤ 0.05 eV/atom, moment ≥ 0.5 μB, gap in [1.1, 1.7] eV. A structure at 0.049 eV/atom and one at 0.051 eV/atom are physically the same, yet they get opposite labels. DFT numerical noise is of the same order of magnitude.
2. **Stability is a property of the whole phase diagram,** not of one structure. Whether a compound is on the convex hull depends on the energies of all competing phases. No descriptor of a single structure contains that information. This is why stability stays the hardest task.
3. **Metallicity and band gap depend on the electronic structure.** Borderline cases (small gaps, semimetals) are hard to separate from geometry alone.
4. **Our test is deliberately strict.** With a random split, polymorphs of the same formula would sit on both sides and the accuracy would look several points higher. We refuse that shortcut, because it measures memorisation, not prediction.

5. **A structure-aware model needs a relaxed structure.** Our structure descriptors come from DFT-relaxed geometries, which already contain some information about the energy. This is fine for known materials and database screening. For a brand-new composition, however, a structure must first be predicted or relaxed (prototype decoration, a machine-learned potential, or DFT). This is why the application keeps the composition-only SSL models as a fallback.

The next real gain would come from **graph neural networks** that learn directly from atomic positions (CGCNN, MEGNet, ALIGNN). They need a GPU and much longer training, and are left as future work.

## 8. Use in the web application (MatSSL Studio)

1. **Structure lookup.** The user types a formula. The backend fetches the **lowest-energy known structure** from the Alexandria OPTIMADE API (`app/services/structure.py`), and the 28 descriptors are computed.
2. **Choice of model.**
   - If a structure is found, the **structure-aware ensemble** is used for every task. The Results page shows a "+ structure" badge, the structure used (Alexandria id, space group, density, DFT hull distance), and the Track A SSL probability for comparison.
   - If no structure is known (a hypothetical composition) or the database cannot be reached, the app **falls back** to the Track A composition-only SSL models.
3. **Explanations.** SHAP contributions now include structure descriptors, marked ◆ (for example "mean coordination number" or "volume per atom").
4. **Offline demo.** `app/models/structure_cache.json` stores the structures of 50 demo materials, so the presentation works without internet.
5. **Model vs. DFT.** When the structure comes from Alexandria, its DFT hull distance is shown next to the stability prediction, so the model can be compared with the database in front of the audience. The model is not always right. Two examples are Cu and Fe₂O₃, which it predicts unstable:
   - Alexandria's lowest-energy Cu is an unusual 7-atom rhombohedral cell, and plain fcc Cu sits 0.076 eV/atom above the hull in that PBE database;
   - Alexandria's lowest-energy Fe₂O₃ is a 40-atom cubic cell.

   Neither ground state is among the 5 Alexandria files (out of about 45) that we downloaded. Training on the complete Alexandria database (about 4.5 million entries) is the obvious next step for stability.

While building this we fixed a bug. The lookup selected the entry with the lowest hull distance using `value or 9`, which treats a distance of exactly **0.0** (the true ground state) as missing. For Cu and Fe the app was therefore showing a metastable structure. The fix compares against `None` explicitly.

## 9. How to reproduce

```bash
uv run python pipeline/build_structure_dataset.py            # per-structure datasets + 28 descriptors
uv run python pipeline/train_high_accuracy.py --task metallicity --trials 30
uv run python pipeline/train_high_accuracy.py --task magnetism   --trials 15
uv run python pipeline/train_high_accuracy.py --task band_gap    --trials 15
uv run python pipeline/train_high_accuracy.py --task stability   --trials 12 --tune-sample 40000
uv run python pipeline/build_structure_cache.py              # offline structures for the demo
uv run pytest -q                                             # includes leakage / invariance tests
```

## 10. Files

| File | Role |
|---|---|
| `pipeline/structure_features.py` | the 28 structure descriptors (shared by training and the app) |
| `pipeline/build_structure_dataset.py` | streaming reader of the raw databases → `data/processed/{task}_struct.parquet` |
| `pipeline/train_high_accuracy.py` | grouped split, ablation, Optuna, stacking, threshold, model export |
| `pipeline/hiacc_model.py` | `StackedEnsemble` class |
| `app/services/structure.py` | Alexandria lookup, offline cache, descriptors at prediction time |
| `app/services/model.py` | chooses the structure-aware or the composition model for each prediction |
| `data/predictions/high_accuracy/` | summaries, ablation tables and figures |
| `notebooks/high_accuracy/high_accuracy.ipynb` | notebook version of the analysis |
| `tests/test_structure.py` | cell-invariance, no-leakage and fallback tests |
