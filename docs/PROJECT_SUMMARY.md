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

## Planned extensions (roadmap)

The platform is being extended from materials to **molecules and medicine**, and to more **industries**. The full plan is in [ROADMAP_extensions.md](ROADMAP_extensions.md).

1. **Toxicity prediction.** Draw a molecule and get its predicted toxicity on the 12 Tox21 endpoints, clinical toxicity (ClinTox) and mutagenicity (AMES). It will use RDKit descriptors, Morgan fingerprints and the same semi-supervised methods, with ZINC as the unlabeled pool.

2. **Illness and therapeutic-efficacy simulator.**
   - The user **builds molecules** in a molecule editor.
   - The app links molecules to illnesses through a **knowledge graph**: illness → protein targets (Open Targets) → known active molecules (ChEMBL).
   - The user **selects an illness** and sees its description, targets and known drugs.
   - The user then **tests their own molecules** against it. Each molecule gets an efficacy score built from per-target activity models, combined with toxicity, drug-likeness and an applicability-domain check. The verdict is *likely effective / uncertain / unlikely*.
   - This is in-silico screening for research and education, not medical advice.

3. **Industry modules.** Specialised screeners built on the same materials pipeline:
   - **batteries:** electrode voltage and capacity;
   - **solar cells:** absorption efficiency;
   - **magnets:** Curie temperature;
   - **thermoelectrics:** zT;
   - **superconductors:** critical temperature.

---

## Ready-to-paste CV lines

**MatSSL – ML platform for materials discovery** · Python, scikit-learn, LightGBM, FastAPI, React, Docker · [github.com/Ibrahimlaref/ml_Mini-project](https://github.com/Ibrahimlaref/ml_Mini-project)

- Built a data pipeline over **1.2 M compositions** from 4 open materials databases, with a vectorised featurizer (142 descriptors) and crystal-structure descriptors (spglib).
- Benchmarked **5 semi-supervised methods** against Optuna-tuned baselines (about 2,700 models, 3 seeds, label-budget curves). Co-Training gave a **+0.73 %** gain; I showed that distribution shift in the unlabeled data cancels the benefit.
- Added unsupervised (Apriori rules, K-Means) and self-supervised (masked autoencoder on 554k samples) learning, plus genetic-algorithm feature selection (142 → 52) and an applicability domain.
- Trained stacked ensembles with leakage-free grouped splits: **91–95 % accuracy** and **ROC-AUC 0.96–0.98** on metallicity, magnetism and stability.
- Deployed everything as **MatSSL Studio** (FastAPI + React + Docker), with SHAP explanations, a 3D crystal viewer and an interactive results dashboard; CI with pytest and GitHub Actions.

**Short version (LinkedIn / portfolio):**
*Built MatSSL, an end-to-end ML platform that predicts properties of inorganic materials from scarce labels. It covers 1.2 M compositions, semi-supervised, unsupervised and self-supervised learning, about 2,700 benchmarked models, and stacked ensembles reaching up to 94.6 % accuracy, deployed as a FastAPI + React web app with explanations and a 3D crystal viewer. Next: toxicity prediction and an illness/drug-efficacy simulator.*
