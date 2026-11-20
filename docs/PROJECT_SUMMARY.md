# MatSSL – Machine-learning platform for materials discovery with semi-supervised learning

**ESI-SBA · Machine Learning mini-project · 2026**
Team: Laref Ibrahim · Nemouchi Faycal Abdelhadi · Zerradi Abdelbasset · Medelci Djezzar Aymen
Code: https://github.com/Ibrahimlaref/ml_Mini-project

> An end-to-end ML platform that predicts properties of inorganic materials (metallicity, magnetism, thermodynamic stability, solar band gap) when labels are scarce. It goes from 1.2 million raw database entries to a deployed web application with explanations, applicability domain and a 3D crystal viewer.

---

## The problem

Finding a new battery cathode, magnet or solar absorber means computing each candidate's properties with quantum-mechanical simulations (DFT). That costs hours of computing per material. As a result, only a small share of known compositions are labelled, while millions of candidate compositions are not.

The project asks whether we can still predict well from only **1,000 labels** by learning from unlabeled data, and how good the best deployable model can get.

## What we built

| Area | Work |
|---|---|
| **Data engineering** | **1.2 M compositions** from 4 open databases: Materials Project (Matbench), JARVIS-DFT (NIST), Alexandria, and GNoME (Google DeepMind, 554k unlabeled). Raw JSON is parsed as a stream to keep memory low. One row per composition, so the same material is never in both train and test. A vectorised Magpie featurizer gives **142 descriptors** (about 65,000 compositions per second). |
| **Semi-supervised learning** | Label Propagation, Label Spreading (with class-mass normalisation), Self-Training, Co-Training on two descriptor views, and Cluster-then-Label. They were compared with 9 supervised baselines tuned with Optuna. In total **≈ 2,700 trained models**: 3 seeds, unlabeled pools from 1k to 100k, two pool types (same database vs. external GNoME), label-budget curves and a fully supervised upper bound. |
| **Unsupervised learning** | Apriori association rules, e.g. {Mn, O} ⇒ magnetic (95 % confidence, lift 4.6). K-Means and PCA find 3 material families without using any labels. |
| **Self-supervised learning** | A masked autoencoder pre-trained on 554k unlabeled compositions (R² = 0.91 on hidden descriptors). Its 32-dimensional embedding powers a similar-material search. |
| **Classical pipeline** | A genetic algorithm (DEAP) cuts the descriptors from 142 to 52 with the same ROC-AUC. Applicability domain built from PCA, K-Means and k-NN distance (95 % of test compositions fall inside it). The decision threshold is calibrated on the training set. |
| **High-accuracy track** | One row per crystal structure (≈ 690k structures) and 28 crystal-structure descriptors (spglib and numpy: symmetry, density, packing, bond lengths, coordination). Stacked ensemble of LightGBM, XGBoost and histogram gradient boosting with a logistic meta-learner, and a threshold chosen to maximise accuracy. The train/test split is grouped by formula, so no composition appears on both sides. |
| **Deployment** | **MatSSL Studio**: React, Vite, Tailwind and Recharts on a FastAPI backend, in one Docker image. Features: periodic-table composition editor, 3D crystal viewer (Alexandria OPTIMADE API), calibrated probability gauges, exact SHAP explanations, nearest known materials, applicability-domain check, comparison of up to 4 candidates, guided demo tour, and a "Research Lab" with every experiment, interactive and available offline. |
| **Quality** | 16 API and leakage tests (pytest), ruff, GitHub Actions CI, executed Jupyter notebooks, a 62-page LaTeX report whose numbers are generated from the result files, and a 23-slide deck. |

## Key results

**Semi-supervised study** (1,000 labels; composite score = (F1 + ROC-AUC)/2; mean of 3 seeds)

| Task | Best SSL method | Gain vs. tuned LightGBM |
|---|---|---|
| Thermodynamic stability (Alexandria) | Co-Training | **+0.73 %** (better on all 3 seeds) |
| Magnetism (JARVIS) | Self-Training | **+0.30 %** (better on all 3 seeds) |
| Metallicity (Materials Project) | Self-Training | +0.01 % (a tie) |

- **When SSL helps:** only when the unlabeled data comes from the **same distribution**.
- **When it hurts:** the external GNoME pool reduces scores by up to −1.25 % because of distribution shift (KS statistic up to 0.48).
- **What helps most:** labels. Going from 250 to 5,000 labels improves results far more than any SSL method.

**High-accuracy deployed models** (held-out crystal structures; split grouped by formula)

| Task | Composition only | Final accuracy | ROC-AUC |
|---|---|---|---|
| Metallicity | 90.3 % | **91.5 %** | 0.969 |
| Magnetism | 93.4 % | **94.6 %** | 0.983 |
| Thermodynamic stability | 85.4 % | **90.3 %** | 0.961 |
| Photovoltaic band gap | 93.5 % | **93.5 %** (balanced accuracy 58.9 %, rare class) | 0.906 |

## Tech stack

**ML and data:** Python 3.12, scikit-learn, LightGBM, XGBoost, Optuna, DEAP, mlxtend, pandas, NumPy, pymatgen, spglib, matminer, PyArrow, Matplotlib
**Chemistry:** RDKit (Python and WebAssembly), pymatgen, spglib, matminer, Ketcher
**Web:** FastAPI, Pydantic, httpx, React 18, Vite, Tailwind CSS, React Router, TanStack Query, Recharts, 3Dmol.js
**Engineering:** uv, Docker, GitHub Actions, pytest, ruff, Jupyter, LaTeX (Tectonic), pptxgenjs

## Skills demonstrated

- Semi-supervised, unsupervised and self-supervised learning on real, large and messy scientific data
- Feature engineering for chemistry and crystallography
- Rigorous, leakage-free evaluation: grouped splits, seeds, ablations, honest reporting of negative results
- Hyper-parameter optimisation, stacking, threshold calibration, applicability domain, explainability (SHAP)
- Full-stack deployment of ML models: REST API, interactive dashboard, Docker, CI
- Scientific communication: report, slides and an interactive results dashboard

---

## Beyond materials: molecules and a Design Lab

The platform now covers a second scientific domain and a discovery workflow. Details and results:
[ROADMAP_extensions.md](ROADMAP_extensions.md).

**Molecule Studio + toxicity (built).** Build a molecule (drawing canvas, fragment builder, or
SMILES), and get a predicted toxicity profile over the 12 Tox21 assays plus clinical toxicity.
It reuses the materials protocol unchanged: 217 RDKit descriptors and a 2048-bit Morgan
fingerprint, a train/test split grouped by **Bemis-Murcko scaffold** (the molecular equivalent of
grouping by formula), an ablation, a stacked ensemble and a calibrated threshold. 249,424
unlabeled ZINC molecules play the role GNoME plays for materials. Mean ROC-AUC **0.768** across
10 endpoints, up to 0.895 on mitochondrial toxicity. Explanations decode fingerprint bits back
into the substructure they stand for.

**Design Lab (built).** Pick a target application - superconductor, solar absorber, permanent
magnet or battery cathode - score your own composition for it, and let the models rank
candidates from the **554,054 unlabeled GNoME compositions** nobody has measured. It adds a new
model for the superconducting critical temperature (MAE **2.9 K**, R² **0.93**, ROC-AUC **0.99**
for "above liquid nitrogen"). Left to its own devices, the ranking returns **cuprates** for
superconductors and **vanadium fluorophosphates** for cathodes - the families real high-Tc
superconductors and sodium-ion cathodes belong to.

**Protein-target activity (next).** Test your molecules against the protein targets that public
databases associate with an illness, using activity models trained on measured ChEMBL data.
In-silico screening for research and teaching, not medical advice.

## Ready-to-paste CV lines

**MatSSL – ML platform for materials discovery** · Python, scikit-learn, LightGBM, FastAPI, React, Docker · [github.com/Ibrahimlaref/ml_Mini-project](https://github.com/Ibrahimlaref/ml_Mini-project)

- Built a data pipeline over **1.2 M compositions** from 4 open materials databases, with a vectorised featurizer (142 descriptors) and crystal-structure descriptors (spglib).
- Benchmarked **5 semi-supervised methods** against Optuna-tuned baselines (about 2,700 models, 3 seeds, label-budget curves). Co-Training gave a **+0.73 %** gain; I showed that distribution shift in the unlabeled data cancels the benefit.
- Added unsupervised (Apriori rules, K-Means) and self-supervised (masked autoencoder on 554k samples) learning, plus genetic-algorithm feature selection (142 → 52) and an applicability domain.
- Trained stacked ensembles with leakage-free grouped splits: **91–95 % accuracy** and **ROC-AUC 0.96–0.98** on metallicity, magnetism and stability.
- Extended the same protocol to **molecules**: RDKit descriptors + Morgan fingerprints, scaffold-grouped splits, 12 Tox21 toxicity endpoints (mean ROC-AUC 0.77), and a superconductor Tc model (MAE 2.9 K, R² 0.93) that ranks 554k unlabeled compositions and recovers the cuprate and vanadium-fluorophosphate families unaided.
- Deployed everything as **MatSSL Studio** (FastAPI + React + Docker), with SHAP explanations, a molecule editor, a 3D crystal viewer and an interactive results dashboard; CI with pytest and GitHub Actions.

**Short version (LinkedIn / portfolio):**
*Built MatSSL, an end-to-end ML platform that predicts properties of inorganic materials and molecules from scarce labels. It covers 1.2 M compositions and 260 k molecules, semi-supervised, unsupervised and self-supervised learning, about 2,700 benchmarked models, stacked ensembles up to 94.6 % accuracy, molecular toxicity models and a Design Lab that screens 554 k unmeasured compositions for batteries and superconductors - deployed as a FastAPI + React web app with explanations, a molecule editor and a 3D crystal viewer.*
