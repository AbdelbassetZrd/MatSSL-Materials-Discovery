# MatSSL – Materials Discovery with Semi-Supervised Learning (MI–SSL)

ESI-SBA · Machine Learning mini-project (Dr. N. Dif)

**Group members:** Laref Ibrahim · Nemouchi Faycal Abdelhadi · Zerradi Abdelbasset · Medelci Djezzar Aymen

Predicting material properties from the chemical composition when labels are scarce,
using **semi-supervised**, **unsupervised** and **self-supervised** learning on large open
materials databases, deployed as a web application (FastAPI + React).

## Data

| Task | Labeled data | Unique compositions | Positive rate |
|---|---|---|---|
| Metallicity | Materials Project (Matbench `matbench_mp_is_metal`) | 77,432 | 47.7 % |
| Magnetism | JARVIS-DFT 3D (NIST) | 64,410 | 20.6 % |
| Thermodynamic stability | Alexandria PBE 3D (500,000 entries) | 436,791 | 29.5 % |
| Photovoltaic band gap (GA + applicability domain) | Materials Project (Matbench `matbench_mp_gap`) | 78,164 | 6.7 % |
| **Unlabeled pool** | GNoME stable materials (Google DeepMind) | 554,054 | – |

Every composition is described by **142 descriptors** (132 Magpie statistics + stoichiometry +
valence-orbital fractions), computed with a vectorised re-implementation of matminer's
featurisers (identical results, ~65,000 compositions/second).

## Methods

* **Semi-supervised:** Label Propagation, Label Spreading (both with class-mass normalisation),
  Self-Training, Co-Training (two descriptor views), Cluster-then-Label.
* **Supervised baselines:** RF, ExtraTrees, XGBoost, LightGBM, Logistic Regression, kNN, MLP,
  Optuna-tuned LightGBM/XGBoost.
* **Self-supervised:** masked-feature autoencoder pre-trained on the 554k GNoME compositions.
* **Unsupervised:** Apriori association rules (elements ⇒ property), PCA + K-Means clustering.
* **Traditional pipeline (band gap):** model comparison, genetic-algorithm descriptor selection
  (DEAP), applicability domain (PCA, K-Means, k-NN distance).
* Protocol: 1,000 labeled compositions, unlabeled pools of 1k–100k (GNoME or same database with
  labels hidden), 3 seeds, stratified 80/20 split, selection by Composite = (F1 + ROC-AUC)/2.

## Repository layout

```
app/            FastAPI backend (routes/materials.py, services/model.py, trained models in app/models)
frontend/       MatSSL Studio: React + Vite + Tailwind + Recharts web interface
pipeline/       data download, featurisation, experiments, GA/AD, unsupervised analysis
notebooks/      data_enhancing / eda / training notebooks per dataset, band_gap, unsupervised
data/predictions/  all result tables (CSV/JSON) and figures
report/         LaTeX report (compile on Overleaf or with tectonic)
presentation/   PPTX presentation
tests/          API tests (pytest)
```

## Reproduce

```bash
uv sync                                              # Python 3.12 environment
python pipeline/download_data.py --alexandria-files 5   # GNoME + Alexandria
uv run python pipeline/download_jarvis_matbench.py      # JARVIS + Matbench
uv run python pipeline/extract_alexandria.py
uv run python pipeline/prepare_datasets.py              # features for all tasks
uv run python pipeline/self_supervised.py               # masked autoencoder
uv run python pipeline/unsupervised_analysis.py         # association rules + clustering
bash pipeline/run_all.sh                                # SSL campaign (3 tasks x 2 pools)
uv run python pipeline/band_gap_ga_ad.py                # GA + applicability domain
uv run python pipeline/select_models.py                 # choose deployed models
uv run python pipeline/build_neighbor_index.py          # embedding index for /similar
uv run python pipeline/export_dashboard.py              # Research Lab data (dashboard.json)
uv run python pipeline/build_notebooks.py               # regenerate the notebooks
bash report/build.sh                                    # tables, macros and report PDF
cd presentation && node build_deck.js                   # PowerPoint deck
```

## Results (1,000 labels, composite = (F1 + ROC-AUC)/2, mean of 3 seeds)

See `report/main.pdf` (sections 6-9) and the Research Lab page of the app. In short:

* with unlabeled data from the same database, the best semi-supervised method beats a strong
  Optuna-tuned LightGBM baseline only slightly: +0.73 % on stability (Co-Training), +0.30 % on
  magnetism (Self-Training), a tie on metallicity;
* external unlabeled data (GNoME) suffers from distribution shift: SSL loses on metallicity
  (-0.30 %) and magnetism (-1.25 %). Larger unlabeled pools tend to hurt every method except
  confident Self-Training;
* labels matter more: the label-budget curve rises steeply from 250 to 5,000 labels.

The deployed models are listed in `app/models/registry.json`.

## High-accuracy track (all labels + crystal structure)

Separate from the 1,000-label SSL study. It uses one row per crystal structure, 28
crystal-structure descriptors (spglib + numpy), tuned LightGBM/XGBoost, a stacked ensemble and
an accuracy-optimal threshold. The test split is grouped by formula.
Full explanation: [`docs/high_accuracy_methods.md`](docs/high_accuracy_methods.md).

| Task | Composition only | Final accuracy | Balanced acc. | ROC-AUC |
|---|---|---|---|---|
| Metallicity | 90.3 % | **91.5 %** | 91.1 % | 0.969 |
| Magnetism | 93.4 % | **94.6 %** | 91.6 % | 0.983 |
| Stability | 85.4 % | **90.3 %** | 87.0 % | 0.961 |
| PV band gap | 93.5 % | **93.5 %** | 58.9 % | 0.906 |

```bash
uv run python pipeline/build_structure_dataset.py
uv run python pipeline/train_high_accuracy.py --task stability   # metallicity, magnetism, band_gap
uv run python pipeline/build_structure_cache.py                  # offline structures for the demo
```

The web app uses these models when the lowest-energy structure of the formula is known
(Alexandria), and the SSL composition models otherwise.

## Molecules and Design Lab

Two extensions on top of the materials core. Full write-up:
[`docs/ROADMAP_extensions.md`](docs/ROADMAP_extensions.md).

**Molecule Studio + toxicity.** Draw a molecule, build it from fragments, or paste a SMILES, and
get a predicted toxicity profile (12 Tox21 assays + clinical toxicity). Same protocol as the
materials track, with Bemis-Murcko **scaffold-grouped** splits and 249k unlabeled ZINC molecules
as the pool. Mean ROC-AUC **0.772** over the 12 Tox21 assays (0.895 mitochondrial toxicity, 0.916 clinical toxicity).

**Design Lab.** Pick an application and get candidate materials, ranked from the 554k unlabeled
GNoME compositions:

| Application | Score | Top hits found |
|---|---|---|
| Superconductor | predicted Tc x stability | cuprates (Ba2La(CuO2)4, 69 K) |
| Battery cathode | cathode chemistry x stability | vanadium fluorophosphates |
| Permanent magnet | magnetic x metallic x stable | rare-earth pnictides |
| Solar absorber | PV gap x stable x non-metal | Cs/Rb chalcogenides |

New model: superconducting critical temperature, MAE **2.9 K**, R² **0.93**, ROC-AUC **0.99**
for "above liquid nitrogen".

```bash
uv run python pipeline/molecules/get_data.py && uv run python pipeline/molecules/featurize_mol.py
uv run python pipeline/molecules/train_toxicity.py
uv run python pipeline/industries/train_superconductor.py
uv run python pipeline/industries/rank_candidates.py
```

## MatSSL Studio (web application)

| Page | What it shows |
|---|---|
| Overview | key figures, processing pipeline, deployed models with test scores |
| Composition Studio | periodic table (element cards, property heat maps), formula editor, atomic fractions, 3D crystal from the Alexandria OPTIMADE API |
| Results | probability gauges with calibrated thresholds, profile radar, applicability domain and chemical-space map, SHAP explanation, nearest known materials (self-supervised embedding), chemistry checks and association rules, CSV/JSON export |
| Research Lab | datasets and shift, SSL comparisons, learning and label-budget curves, model zoo, deployed models (ROC, confusion matrices), GA and applicability domain, association rules and clusters, autoencoder; works offline from `dashboard.json` |
| Compare | up to four compositions side by side |
| Molecule Studio | Ketcher drawing canvas, fragment builder, SMILES box, RDKit 2D depiction, drug-likeness |
| Toxicity | 12 Tox21 gauges, risk badge, SHAP explanations that name the substructure behind each bit |
| Design Lab | application scores for your composition + ranked candidates from the GNoME pool |
| Demo tour | scripted walk-through for the presentation (keyboard ← / →) |

API endpoints (FastAPI, documentation at `/docs`): `POST /materials/predict`,
`POST /materials/batch-predict`, `POST /materials/analyze`, `GET /materials/similar`,
`GET /materials/structure`, `GET /materials/experiments`, `GET /materials/models-info`,
`POST /molecules/predict`, `POST /molecules/properties`, `GET /design/applications`,
`POST /design/score`, `GET /design/candidates`.

## Run the application

```bash
# backend (http://localhost:8000/docs)
uv run uvicorn app.main:app --reload
# frontend in development (http://localhost:5173/ui/)
cd frontend && npm install && npm run dev
```

or everything in one container (the API also serves the interface at `/ui/`):

```bash
docker build -t matssl . && docker run -p 8000:8000 matssl   # open http://localhost:8000/ui/
```

## Tests

```bash
uv run pytest -q && uv run ruff check app tests
```
