"""Generate the project notebooks (friend's layout) with nbformat.

notebooks/<task>/<task>_data_enhancing_notebook.ipynb
notebooks/<task>/<task>_eda_notebook.ipynb
notebooks/<task>/<task>_training_notebook.ipynb
notebooks/band_gap/band_gap.ipynb
notebooks/unsupervised/unsupervised_and_self_supervised.ipynb

Execute afterwards with:
  uv run jupyter nbconvert --to notebook --execute --inplace notebooks/**/*.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

TASKS = {
    "metallicity": dict(title="Metallicity (Materials Project)", source="Materials Project "
                        "(Matbench `matbench_mp_is_metal`)", raw="matbench/matbench_mp_is_metal"
                        ".parquet", target="is_metal (1 = metal, 0 = non-metal)",
                        loader="labeled_metallicity"),
    "magnetism": dict(title="Magnetism (JARVIS-DFT)", source="JARVIS-DFT 3D (NIST)",
                      raw="jarvis/jarvis_dft_3d.parquet",
                      target="is_magnetic (|magmom| >= 0.5 muB, ground-state polymorph)",
                      loader="labeled_magnetism"),
    "stability": dict(title="Thermodynamic Stability (Alexandria)", source="Alexandria PBE 3D "
                      "(5 chunks, 500,000 entries)", raw="alexandria/alexandria_pbe3d.parquet",
                      target="is_stable (min E_above_hull <= 0.05 eV/atom)",
                      loader="labeled_stability"),
}

SETUP = """import sys, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path.cwd().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
sns.set_theme(style="whitegrid")
"""


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


def save(nb, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    nbf.write(nb, path)


# ============================================================================ enhancing
def enhancing(task, t):
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(f"# Data Enhancing – {t['title']}\n\nRaw database → one labeled row per unique "
           f"composition → 142 composition descriptors (Magpie + stoichiometry + valence "
           f"orbitals).\n\n* **Source:** {t['source']}\n* **Target:** {t['target']}"),
        code(SETUP + f'TASK = "{task}"\nRAW = ROOT / "data" / "raw"\nPROC = ROOT / "data" / '
             '"processed"'),
        code(f'# Cell 1: Load the raw database\nraw = pd.read_parquet(RAW / "{t["raw"]}")\n'
             'print(f"raw entries: {len(raw):,}")\nraw.head()'),
        code("# Cell 2: Simple stats\nraw.describe(include='all').T.head(15)"),
        code("# Cell 3: One label per unique composition (reduced formula, polymorph rule)\n"
             f"import prepare_datasets as P\nlab = P.{t['loader']}()\n"
             'print(f"unique compositions: {len(lab):,}")\n'
             'print(lab["target"].value_counts().rename({0: "negative", 1: "positive"}))\n'
             "lab.head()"),
        code("# Cell 4: Composition featurisation (vectorised Magpie, identical to matminer)\n"
             "from featurize import featurize_formulas, feature_names\n"
             'examples = pd.Series(["LiFePO4", "Fe2O3", "GaAs", "NdFeB", "SrTiO3"])\n'
             "X, ok = featurize_formulas(examples)\nX.index = examples\n"
             'print(len(feature_names()), "descriptors")\n'
             'X[["MagpieData mean Electronegativity", "MagpieData range Electronegativity", '
             '"MagpieData mean CovalentRadius", "MagpieData mean NValence", '
             '"MagpieData mean GSmagmom", "0-norm", "frac d valence electrons"]]'),
        code("# Cell 5: Check against matminer's reference implementation\n"
             "from matminer.featurizers.composition import ElementProperty, Stoichiometry, "
             "ValenceOrbital\nfrom pymatgen.core import Composition\n"
             'ep, st, vo = ElementProperty.from_preset("magpie"), Stoichiometry(), '
             'ValenceOrbital(props=["frac"])\n'
             "ref = np.array([ep.featurize(Composition(f)) + st.featurize(Composition(f)) + "
             "vo.featurize(Composition(f)) for f in examples], float)\n"
             'print("max |ours - matminer| =", np.nanmax(np.abs(X.values - ref)))'),
        code("# Cell 6: Featurise the whole labeled set and the GNoME unlabeled pool\n"
             "import time\nt = time.time()\n"
             "feats, ok = featurize_formulas(lab['formula'])\n"
             'print(f"featurised {ok.sum():,} compositions in {time.time()-t:.1f}s")\n'
             f'df = pd.read_parquet(PROC / "{task}_features.parquet")\n'
             'gn = pd.read_parquet(PROC / "unlabeled_gnome_features.parquet", '
             'columns=["formula"])\n'
             'print(f"saved labeled table: {df.shape} | GNoME unlabeled pool: {len(gn):,}")'),
    ]
    save(nb, NB / task / f"{task}_data_enhancing_notebook.ipynb")


# ============================================================================ EDA
def eda(task, t):
    fig = f'FIG = ROOT / "report" / "figures" / "{task}"; FIG.mkdir(parents=True, exist_ok=True)'
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(f"# EDA for Enhanced Materials Data – {t['title']}"),
        md("## Labeled Data"),
        code(SETUP + fig + f'\nTASK = "{task}"\n'
             "# Step 1: load the enhanced dataset\n"
             'df_labeled = pd.read_parquet(ROOT / "data/processed" / f"{TASK}_features.parquet")\n'
             "feat = [c for c in df_labeled.columns if c.startswith(('MagpieData', 'frac ')) "
             "or c.endswith('-norm')]\n"
             'print(df_labeled.shape)\ndf_labeled[["formula", "target"] + feat[:6]].head()'),
        code('# Step 2: feature groups and data types\nprint("=" * 80)\n'
             'print("COLUMN NAMES AND DATA TYPES")\nprint("=" * 80)\n'
             "groups = pd.Series([c.split(' ')[2] if c.startswith('MagpieData') else "
             "('stoichiometry' if c.endswith('-norm') else 'valence orbital') for c in feat])"
             ".value_counts()\nprint(df_labeled[feat].dtypes.value_counts())\ngroups"),
        code('# Step 3: summary statistics\nkey = ["MagpieData mean Electronegativity", '
             '"MagpieData range Electronegativity", "MagpieData mean CovalentRadius", '
             '"MagpieData mean NValence", "MagpieData mean GSbandgap", '
             '"MagpieData mean GSmagmom", "MagpieData mean MeltingT", "0-norm"]\n'
             "df_labeled[key].describe().T"),
        code("# Step 4: missing values\nmiss = df_labeled[feat].isna().sum()\n"
             'print(f"features with missing values: {(miss > 0).sum()} / {len(feat)}; '
             'max missing rate {miss.max() / len(df_labeled):.4%}")\n'
             "miss[miss > 0].sort_values(ascending=False).head(10)"),
        code("# Step 5: target distribution and balance\n"
             "fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))\n"
             "vc = df_labeled['target'].value_counts().sort_index()\n"
             "ax[0].bar(['negative (0)', 'positive (1)'], vc.values, "
             "color=['#94a3b8', '#4f46e5'])\n"
             "for i, v in enumerate(vc.values): ax[0].text(i, v, f'{v:,}', ha='center', "
             "va='bottom')\nax[0].set_title('Class counts')\n"
             "ax[1].pie(vc.values, labels=['negative', 'positive'], autopct='%1.1f%%', "
             "colors=['#94a3b8', '#4f46e5'])\nax[1].set_title('Class balance')\n"
             "plt.tight_layout(); plt.savefig(FIG / 'target_distribution.png', dpi=150); "
             "plt.show()"),
        code("# Step 6: key descriptors by class\nfig, axes = plt.subplots(2, 4, figsize=(20, 8))\n"
             "for ax, c in zip(axes.flat, key):\n"
             "    for y, col in ((0, '#94a3b8'), (1, '#4f46e5')):\n"
             "        ax.hist(df_labeled.loc[df_labeled.target == y, c].dropna(), bins=60, "
             "alpha=.6, density=True, color=col, label=f'class {y}')\n"
             "    ax.set_title(c.replace('MagpieData ', ''), fontsize=10)\naxes[0, 0].legend()\n"
             "plt.tight_layout(); plt.savefig(FIG / 'descriptors_by_class.png', dpi=150); "
             "plt.show()"),
        code("# Step 7: Pearson correlation matrix (threshold 0.5)\n"
             'print("=" * 80); print("CORRELATION ANALYSIS (Threshold: 0.5)"); print("=" * 80)\n'
             "corr_t = df_labeled[feat].corrwith(df_labeled['target']).abs().sort_values("
             "ascending=False)\ntop = corr_t.head(20).index.tolist()\n"
             "C = df_labeled[top + ['target']].corr()\n"
             "plt.figure(figsize=(15, 12)); sns.heatmap(C, cmap='coolwarm', center=0, "
             "annot=False)\nplt.title('Correlation map: 20 descriptors most correlated with "
             "the target')\nplt.tight_layout(); plt.savefig(FIG / 'correlation_map.png', "
             "dpi=150); plt.show()\nfull = df_labeled[feat].corr().abs()\n"
             "pairs = (full.where(np.triu(np.ones(full.shape), 1).astype(bool)).stack())\n"
             'print(f"descriptor pairs with |r| > 0.5: {(pairs > 0.5).sum():,} of {len(pairs):,}")\n'
             'print("top correlations with target:"); corr_t.head(10)'),
        md("## Unlabeled Data"),
        code("# Load the unlabeled pool (GNoME), excluding compositions present in the "
             "labeled set\n"
             'df_unlabeled = pd.read_parquet(ROOT / "data/processed/unlabeled_gnome_features.'
             'parquet")\n'
             "df_unlabeled = df_unlabeled[~df_unlabeled.formula.isin(set(df_labeled.formula))]\n"
             'print(f"Labeled data:   {len(df_labeled):,} compositions")\n'
             'print(f"Unlabeled data: {len(df_unlabeled):,} compositions (GNoME)")\n'
             'print(f"Ratio unlabeled/labeled: {len(df_unlabeled) / len(df_labeled):.1f}")'),
        code("# Compare feature distributions between labeled and unlabeled data\n"
             "fig, axes = plt.subplots(2, 4, figsize=(20, 8))\n"
             "sample_u = df_unlabeled.sample(min(100000, len(df_unlabeled)), random_state=0)\n"
             "for ax, c in zip(axes.flat, key):\n"
             "    ax.hist(df_labeled[c].dropna(), bins=60, alpha=.55, density=True, "
             "label='labeled')\n"
             "    ax.hist(sample_u[c].dropna(), bins=60, alpha=.55, density=True, "
             "label='unlabeled (GNoME)')\n"
             "    ax.set_title(c.replace('MagpieData ', ''), fontsize=10)\n"
             "axes[0, 0].legend()\nplt.suptitle('Distribution: labeled vs unlabeled')\n"
             "plt.tight_layout(); plt.savefig(FIG / 'labeled_vs_unlabeled.png', dpi=150); "
             "plt.show()"),
        code("# Statistical tests: are the distributions similar? (Kolmogorov-Smirnov)\n"
             "from scipy.stats import ks_2samp\nrows = []\nfor c in key:\n"
             "    s, p = ks_2samp(df_labeled[c].dropna(), sample_u[c].dropna())\n"
             "    rows.append({'feature': c, 'KS statistic': s, 'p-value': p, "
             "'similar (KS<0.1)': s < 0.1})\npd.DataFrame(rows)"),
        code("# Box plots for outlier comparison\nfig, axes = plt.subplots(2, 4, figsize=(20, 8))\n"
             "for ax, c in zip(axes.flat, key):\n"
             "    ax.boxplot([df_labeled[c].dropna(), sample_u[c].dropna()], "
             "tick_labels=['labeled', 'unlabeled'], showfliers=True, flierprops={'markersize': 1})\n"
             "    ax.set_title(c.replace('MagpieData ', ''), fontsize=10)\n"
             "plt.suptitle('Outliers (box plots)'); plt.tight_layout(); "
             "plt.savefig(FIG / 'outliers.png', dpi=150); plt.show()\n"
             "z = (df_labeled[feat] - df_labeled[feat].mean()) / df_labeled[feat].std()\n"
             'print(f"values beyond 3 std: {(z.abs() > 3).sum().sum():,} '
             '({(z.abs() > 3).mean().mean():.2%} of all values)")'),
        code("# Chemical rules: number of elements, charge balance, element frequency\n"
             "from pymatgen.core import Composition\n"
             "smp = df_labeled.sample(3000, random_state=0)\n"
             "def balanced(f):\n    try:\n        return len(Composition(f).oxi_state_guesses("
             "max_sites=-50)) > 0\n    except Exception:\n        return False\n"
             "smp = smp.assign(charge_balanced=smp.formula.map(balanced))\n"
             "els = pd.Series([e.symbol for f in df_labeled.formula.sample(30000, random_state=0)"
             " for e in Composition(f).elements]).value_counts()\n"
             "fig, axes = plt.subplots(1, 3, figsize=(21, 5))\n"
             "df_labeled['0-norm'].value_counts().sort_index().plot.bar(ax=axes[0], "
             "color='#4f46e5'); axes[0].set_title('Number of elements per composition')\n"
             "smp.groupby('charge_balanced')['target'].mean().plot.bar(ax=axes[1], "
             "color=['#94a3b8', '#059669']); axes[1].set_title('Positive rate vs charge balance')\n"
             "els.head(30).plot.bar(ax=axes[2], color='#0ea5e9'); axes[2].set_title("
             "'30 most frequent elements')\nplt.tight_layout(); plt.savefig(FIG / "
             "'chemical_rules.png', dpi=150); plt.show()\n"
             'print(f"charge-balanced fraction: {smp.charge_balanced.mean():.1%}")'),
        md("## Data Processing"),
        code('# Step 1: keep all informative features (drop constant ones)\nprint("=" * 80)\n'
             'print("STEP 1: FEATURE SELECTION - KEEPING ALL FEATURES")\nprint("=" * 80)\n'
             "keep = [c for c in feat if df_labeled[c].nunique() > 1]\n"
             'print(f"{len(keep)} informative descriptors kept, {len(feat) - len(keep)} '
             'constant removed")'),
        code('# Step 2: handle outliers (clipping at mean +- 3 std) and impute missing values\n'
             'print("STEP 2: HANDLE OUTLIERS (CLIPPING)")\n'
             "X_raw = df_labeled[keep].fillna(df_labeled[keep].median())\n"
             "lo, hi = X_raw.mean() - 3 * X_raw.std(), X_raw.mean() + 3 * X_raw.std()\n"
             "n_clip = ((X_raw < lo) | (X_raw > hi)).sum().sum()\n"
             "X_clip = X_raw.clip(lo, hi, axis=1)\n"
             'print(f"clipped values: {n_clip:,} ({n_clip / X_raw.size:.2%})")'),
        code('# Step 3: z-score normalisation (StandardScaler) – before vs after\n'
             'print("STEP 3: Z-SCORE NORMALIZATION")\nfrom sklearn.preprocessing import '
             "StandardScaler\nscaler = StandardScaler().fit(X_clip)\n"
             "X_scaled = pd.DataFrame(scaler.transform(X_clip), columns=keep)\n"
             "show = key[:4]\nfig, axes = plt.subplots(2, 4, figsize=(20, 8))\n"
             "for j, c in enumerate(show):\n"
             "    axes[0, j].hist(X_raw[c], bins=60, color='#94a3b8'); "
             "axes[0, j].set_title('BEFORE: ' + c.replace('MagpieData ', ''), fontsize=10)\n"
             "    axes[1, j].hist(X_scaled[c], bins=60, color='#4f46e5'); "
             "axes[1, j].set_title('AFTER (clip + z-score)', fontsize=10)\n"
             "plt.tight_layout(); plt.savefig(FIG / 'normalization_before_after.png', dpi=150); "
             "plt.show()\n"
             "pd.DataFrame({'mean before': X_raw[show].mean(), 'std before': X_raw[show].std(), "
             "'mean after': X_scaled[show].mean(), 'std after': X_scaled[show].std()}).round(3)"),
        code('# Step 4-5: combine with target, final dataset statistics\n'
             'print("=" * 80); print("FINAL OUTPUT OF THE PREPROCESSING PIPELINE"); '
             'print("=" * 80)\n'
             'print(f"Labeled compositions  : {len(df_labeled):,}")\n'
             'print(f"Unlabeled compositions: {len(df_unlabeled):,}")\n'
             'print(f"Descriptors           : {len(keep)}")\n'
             "print(f\"Positive rate         : {df_labeled['target'].mean():.3f}\")\n"
             'print("The scaler is fitted on training data only inside the training pipeline '
             '(pipeline/run_ssl_experiments.py) and saved to app/models/ for inference.")'),
    ]
    save(nb, NB / task / f"{task}_eda_notebook.ipynb")


# ============================================================================ training
def training(task, t):
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(f"# 🔬 Semi-Supervised Learning for {t['title']}\n"
           "## Demonstrating SSL with limited labeled data\n\n---\n\n"
           "### 🎯 Objective\nTrain with only **1,000 labeled compositions** and compare a "
           "supervised baseline with five semi-supervised methods (Label Propagation, Label "
           "Spreading, Self-Training, Co-Training, Cluster-then-Label) and self-supervised "
           "pre-training, while the unlabeled pool grows from 1,000 to 100,000 compositions.\n\n"
           "The full sweep (3 seeds × 4 pool sizes × 2 unlabeled-pool settings, Optuna "
           "tuning) is run by `pipeline/run_ssl_experiments.py`; this notebook loads its "
           "results and re-runs every method once live."),
        code(SETUP + f'TASK = "{task}"\nPRED = ROOT / "data" / "predictions" / TASK\n'
             "from IPython.display import Image, display"),
        code('print("=" * 100); print("STEP 1: LOADING EXPERIMENT RESULTS"); print("=" * 100)\n'
             "summ = {p: json.loads((PRED / p / f'ssl_summary_{TASK}.json').read_text()) "
             "for p in ('indist', 'gnome') if (PRED / p / f'ssl_summary_{TASK}.json').exists()}\n"
             "res = {p: pd.read_csv(PRED / p / f'ssl_results_{TASK}.csv') for p in summ}\n"
             "for p, s in summ.items():\n    print(p, json.dumps(s['training_setup'], indent=1))"),
        code('print("STEP 2-3: BASELINE MODEL ZOO (supervised, 1,000 labels, mean over 3 seeds)")\n'
             "r = res['indist']\nzoo = r[r.Method == 'Baseline'].groupby('Config')[["
             "'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Composite_Score']]"
             ".agg(['mean', 'std'])\nzoo.sort_values(('Composite_Score', 'mean'), "
             "ascending=False).round(4)"),
        code('print("STEP 4: OPTUNA HYPER-PARAMETER SEARCH (25 trials, 5-fold CV)")\n'
             "pd.DataFrame({'CV composite': summ['indist']['optuna_cv_scores'], "
             "'best params': {k: str(v) for k, v in summ['indist']['tuned_params'].items()}})"),
        code('print("STEP 4b: SELF-SUPERVISED PRE-TRAINING (masked autoencoder on 554k GNoME)")\n'
             "r[r.Method == 'Self-Supervised (AE)'].groupby('Config')[['F1-Score', 'ROC-AUC', "
             "'Composite_Score']].agg(['mean', 'std']).round(4)"),
        code('print("STEP 5-8: SEMI-SUPERVISED SWEEP – mean composite per pool size")\n'
             "for p, r_ in res.items():\n    print(f'--- unlabeled pool: {p}')\n"
             "    ssl = r_[~r_.Method.isin(['Baseline', 'Self-Supervised (AE)'])]\n"
             "    display(ssl.pivot_table(index=['Method', 'Config'], columns='N_Unlabeled', "
             "values='Composite_Score', aggfunc='mean').round(4))"),
        code('print("STEP 9: COMPREHENSIVE RESULTS COMPARISON (best configuration per method)")\n'
             "for p, s in summ.items():\n    print(f'--- unlabeled pool: {p}')\n"
             "    b = pd.DataFrame(s['best_per_method'])[['Method', 'Model_Config', "
             "'Composite_Score_mean', 'Composite_Score_std', 'F1-Score_mean', 'ROC-AUC_mean', "
             "'Accuracy_mean']]\n    display(b.round(4))\n"
             "    print('SSL vs best baseline: %+.2f %%' % "
             "s['ssl_improvement_over_best_baseline_pct'])"),
        code('print("STEP 10: LEARNING CURVES")\nfor p in summ:\n'
             "    display(Image(str(PRED / p / f'{TASK}_learning_curves.png')))"),
        code('print("STEP 11: BEST MODELS COMPARISON (balanced composite metric)")\nfor p in summ:\n'
             "    display(Image(str(PRED / p / f'{TASK}_methods_comparison_balanced.png')))"),
        code('print("STEP 12: LABEL-BUDGET CURVE AND FULLY-SUPERVISED UPPER BOUND")\nfor p in summ:\n'
             "    display(Image(str(PRED / p / f'{TASK}_label_budget_curve.png')))\n"
             "    print(p, 'fully supervised:', summ[p]['fully_supervised_upper_bound'])"),
        md("### Live run: every method once (seed 0, 1,000 labels, 5,000 unlabeled)"),
        code("import ssl_core as core\nfrom run_ssl_experiments import Preprocessor, "
             "stratified_take\nfrom featurize import view_columns\n"
             "from sklearn.model_selection import train_test_split\n"
             "lab = pd.read_parquet(ROOT / 'data/processed' / f'{TASK}_features.parquet')\n"
             "feat = [c for c in lab.columns if c.startswith(('MagpieData', 'frac ')) or "
             "c.endswith('-norm')]\n"
             "tr, te = train_test_split(np.arange(len(lab)), test_size=0.2, "
             "stratify=lab.target, random_state=42)\n"
             "prep = Preprocessor().fit(lab.iloc[tr][feat])\n"
             "Xtr, ytr = prep.transform(lab.iloc[tr][feat]), lab.target.values[tr]\n"
             "Xte, yte = prep.transform(lab.iloc[te][feat]), lab.target.values[te]\n"
             "rng = np.random.default_rng(0)\nli = stratified_take(ytr, 1000, rng)\n"
             "rest = rng.permutation(np.setdiff1d(np.arange(len(ytr)), li))[:5000]\n"
             "keep = stratified_take(yte, min(10000, len(yte)), rng)\n"
             "d = core.Data(Xtr[li], ytr[li], Xtr[rest], Xte[keep], yte[keep])\n"
             "v1n, v2n = view_columns()\nv2 = [i for i, c in enumerate(prep.cols) if c in v2n "
             "or (any(k in c for k in ('Valence', 'Unfilled', 'Column', 'Row')) and "
             "(c.endswith(' mean') or c.endswith(' avg_dev')))]\n"
             "v1 = [i for i in range(len(prep.cols)) if i not in v2]\n"
             "best = summ['indist']['best_supervised_baseline']['config'].split(' ')[0]\n"
             "params = summ['indist']['tuned_params'].get(best, {})\n"
             "runs = [core.run_baseline(d, best, params=params)[0],\n"
             "        core.run_label_propagation(d, 'knn', n_neighbors=10)[0],\n"
             "        core.run_label_propagation(d, 'knn', n_neighbors=10, spreading=True)[0],\n"
             "        core.run_self_training(d, best, threshold=0.9, max_iter=5, "
             "params=params)[0],\n"
             "        core.run_co_training(d, v1, v2, best, threshold=0.85, n_rounds=8, "
             "params=params)[0],\n"
             "        core.run_cluster_then_label(d, best, params=params)[0]]\n"
             "pd.DataFrame(runs)[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', "
             "'ROC-AUC', 'Composite_Score', 'Training_Time']].round(4)"),
        code('print("STEP 13: DEPLOYED MODEL (selected architecture re-trained on all '
             'training labels)")\nfor p, s in summ.items():\n'
             "    print(p, s['best_ssl_model']['Model_Config'], '->', "
             "s['deployment_model_test_metrics'])"),
    ]
    save(nb, NB / task / f"{task}_training_notebook.ipynb")


def band_gap():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Traditional Materials Informatics – Photovoltaic Band Gap\n\nModel comparison, "
           "**genetic-algorithm** descriptor selection (DEAP) and **applicability-domain** "
           "assessment (PCA, K-Means, distance), mirroring a classical QSAR/QSPR workflow. "
           "Full run: `pipeline/band_gap_ga_ad.py`."),
        code(SETUP + 'OUT = ROOT / "data/predictions/band_gap"\n'
             "from IPython.display import Image, display\n"
             "S = json.loads((OUT / 'band_gap_summary.json').read_text())\nS['dataset']"),
        code("# 1. Model comparison on all descriptors\npd.DataFrame(S['model_comparison']).round(4)"),
        code("# 2. Genetic algorithm\nga = pd.read_csv(OUT / 'band_gap_ga_history.csv')\n"
             "display(ga.round(4))\ndisplay(Image(str(OUT / 'band_gap_ga_evolution.png')))\n"
             "{k: v for k, v in S['ga'].items() if k != 'selected_descriptors'}"),
        code("# Selected descriptors\npd.Series(S['ga']['selected_descriptors'])"),
        code("# 3. Test performance: all vs GA-selected descriptors\n"
             "pd.DataFrame([S['test_all_descriptors'], S['test_ga_descriptors']]).round(4)"),
        code("# 4. Applicability domain\nfor f in ('band_gap_ad_pca.png', 'band_gap_ad_kmeans.png', "
             "'band_gap_ad_distance.png'):\n    display(Image(str(OUT / f)))\n"
             "{k: v for k, v in S['applicability_domain'].items() if k != 'cluster_share'}"),
        code("# 5. Use the saved predictor\nimport joblib\n"
             "from run_ssl_experiments import Preprocessor\nimport __main__\n"
             "__main__.Preprocessor = Preprocessor\nfrom featurize import featurize_formulas\n"
             "pkg = joblib.load(ROOT / 'app/models/band_gap_model.pkl')\n"
             "prep = joblib.load(ROOT / 'app/models/band_gap_scaler.pkl')\n"
             "for f in ['CdTe', 'GaAs', 'Si', 'CsPbI3', 'Cu2ZnSnS4', 'Al2O3', 'Fe']:\n"
             "    X = prep.transform(featurize_formulas(pd.Series([f]))[0])[:, "
             "pkg['selected_idx']]\n"
             "    p = pkg['model'].predict_proba(X)[0, 1]\n"
             "    d = pkg['nn'].kneighbors(X, n_neighbors=5)[0].mean()\n"
             "    print(f\"{f:<10} P(PV window)={p:.3f}  inside AD={d <= pkg['ad_threshold']}\")"),
    ]
    save(nb, NB / "band_gap" / "band_gap.ipynb")


def unsupervised():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Unsupervised & Self-Supervised Learning on Materials\n\n1. **Association rules** "
           "(Apriori) – which elements co-occur with stability, magnetism, metallicity, PV "
           "band gap.\n2. **Clustering** – PCA + K-Means on labeled and unlabeled materials.\n"
           "3. **Self-supervised pre-training** – masked-feature autoencoder on 554k GNoME "
           "compositions."),
        code(SETUP + 'U = ROOT / "data/predictions/unsupervised"\n'
             'SS = ROOT / "data/predictions/self_supervised"\n'
             "from IPython.display import Image, display"),
        code("# 1. Association rules (support, confidence, lift)\n"
             "for t in ['stability', 'magnetism', 'metallicity', 'band_gap']:\n"
             "    r = pd.read_csv(U / f'association_rules_{t}.csv')\n"
             "    print(f'=== {t}: {len(r)} rules'); display(r.head(8).round(4))\n"
             "    display(Image(str(U / f'association_rules_{t}.png')))"),
        code("# Element co-occurrence rules on the unlabeled GNoME database\n"
             "display(pd.read_csv(U / 'association_rules_gnome_elements.csv').round(4))"),
        code("# 2. Clustering\nfor f in ('kmeans_elbow_silhouette.png', 'kmeans_pca_clusters.png', "
             "'kmeans_cluster_property_rates.png'):\n    display(Image(str(U / f)))\n"
             "pd.read_csv(U / 'kmeans_cluster_profiles.csv')"),
        code("# 3. Self-supervised masked autoencoder\n"
             "h = pd.read_csv(SS / 'autoencoder_training_history.csv')\n"
             "fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n"
             "ax[0].plot(h.epoch, h.loss, 'o-'); ax[0].set(title='Reconstruction loss', "
             "xlabel='epoch')\nax[1].plot(h.epoch, h.val_r2_all, 'o-', label='all features')\n"
             "ax[1].plot(h.epoch, h.val_r2_masked_features, 's-', label='masked features')\n"
             "ax[1].set(title='Validation R²', xlabel='epoch'); ax[1].legend()\n"
             "FIG = ROOT / 'report/figures/unsupervised'; FIG.mkdir(parents=True, exist_ok=True)\n"
             "plt.tight_layout(); plt.savefig(FIG / 'autoencoder_training.png', dpi=150); plt.show()\n"
             "json.loads((SS / 'autoencoder_summary.json').read_text())"),
    ]
    save(nb, NB / "unsupervised" / "unsupervised_and_self_supervised.ipynb")


if __name__ == "__main__":
    for task, t in TASKS.items():
        enhancing(task, t)
        eda(task, t)
        training(task, t)
    band_gap()
    unsupervised()
    print("notebooks written to", NB)
