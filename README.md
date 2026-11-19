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

## MatSSL Studio (web application)

| Page | What it shows |
|---|---|
| Overview | key figures, processing pipeline, deployed models with test scores |
| Composition Studio | periodic table (element cards, property heat maps), formula editor, atomic fractions, 3D crystal from the Alexandria OPTIMADE API |
| Results | probability gauges with calibrated thresholds, profile radar, applicability domain and chemical-space map, SHAP explanation, nearest known materials (self-supervised embedding), chemistry checks and association rules, CSV/JSON export |
| Research Lab | datasets and shift, SSL comparisons, learning and label-budget curves, model zoo, deployed models (ROC, confusion matrices), GA and applicability domain, association rules and clusters, autoencoder; works offline from `dashboard.json` |
| Compare | up to four compositions side by side |
| Demo tour | scripted walk-through for the presentation (keyboard ← / →) |

API endpoints (FastAPI, documentation at `/docs`): `POST /materials/predict`,
`POST /materials/batch-predict`, `POST /materials/analyze`, `GET /materials/similar`,
`GET /materials/structure`, `GET /materials/experiments`, `GET /materials/models-info`.

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
