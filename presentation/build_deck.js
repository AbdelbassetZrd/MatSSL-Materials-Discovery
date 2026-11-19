// Build presentation/ML_Project_Presentation.pptx from the experiment outputs.
// Numbers: report/generated/values.json + data/predictions/*/*/ssl_summary_*.json
// Figures: report/figures/*            Run: node build_deck.js   (after report/make_results.py)
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("./apply_theme.js");

const ROOT = path.resolve(__dirname, "..");
const FIG = (p) => path.join(ROOT, "report", "figures", p);
const has = (p) => fs.existsSync(FIG(p));
const V = JSON.parse(fs.readFileSync(path.join(ROOT, "report", "generated", "values.json"), "utf8"));
const summary = (task, pool) => {
  const f = path.join(ROOT, "data", "predictions", task, pool, `ssl_summary_${task}.json`);
  return fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, "utf8")) : null;
};
const OUT = path.join(__dirname, "ML_Project_Presentation.pptx");

const THEME = {
  name: "MatSSL Crystal",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1F3B", lt1: "FFFFFF", dk2: "312E81", lt2: "EEF2FF",
    accent1: "4F46E5", accent2: "0891B2", accent3: "F59E0B", accent4: "059669",
    accent5: "64748B", accent6: "DC2626", hlink: "4F46E5", folHlink: "7C3AED",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.title = "Materials Discovery with Semi-Supervised Learning";
pres.author = "Laref Ibrahim, Nemouchi Faycal Abdelhadi, Zerradi Abdelbasset, Medelci Djezzar Aymen";
pres.company = "ESI-SBA";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.3, w: 11.7, h: 1.6,
        fontSize: 40, bold: true, color: C.background1, valign: "middle" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.95, w: 11.7, h: 1.0,
        fontSize: 20, color: "CADCFC" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85,
        fontSize: 30, bold: true, color: C.text2, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "MatSSL · ESI-SBA Machine Learning mini-project",
        options: { x: 0.6, y: 7.05, w: 8, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.02, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5 },
});
pres.defineSlideMaster({
  title: "SECTION_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.6, w: 11.7, h: 1.2,
        fontSize: 40, bold: true, color: C.background1, valign: "middle" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.8, w: 11.7, h: 1.0,
        fontSize: 18, color: "CADCFC" }, text: "" } },
  ],
});

// ------------------------------------------------------------------ helpers
let section = null;
function newSlide(master, title, sectionTitle) {
  if (sectionTitle && sectionTitle !== section) {
    pres.addSection({ title: sectionTitle });
    section = sectionTitle;
  }
  const s = pres.addSlide({ masterName: master, sectionTitle: section });
  if (title) s.addText(title, { placeholder: "title" });
  return s;
}
function image(s, p, x, y, w, h, name) {
  if (has(p)) {
    s.addImage({ path: FIG(p), x, y, w, h, sizing: { type: "contain", w, h }, objectName: name || p });
  } else {
    s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: C.background2 },
      line: { color: C.accent5, dashType: "dash" }, objectName: "pending " + p });
    s.addText(`(figure pending: ${p})`, { x, y, w, h, align: "center", fontSize: 12,
      color: C.accent5, isTextBox: true });
  }
}
function card(s, x, y, w, h, head, body, colorKey, name) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12,
    fill: { color: C.background2 }, line: { color: C.background2 }, objectName: name + " card",
    shadow: { type: "outer", color: "1B1F3B", opacity: 0.12, blur: 6, offset: 2, angle: 90 } });
  s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.3, w: 0.5, h: 0.5,
    fill: { color: C[colorKey] }, line: { color: C[colorKey] }, objectName: name + " dot" });
  s.addText(head, { x: x + 0.95, y: y + 0.25, w: w - 1.15, h: 0.6, fontSize: 18, bold: true,
    color: C.text2, margin: 0, valign: "middle", isTextBox: true, objectName: name + " head" });
  s.addText(body, { x: x + 0.3, y: y + 1.0, w: w - 0.6, h: h - 1.2, fontSize: 14,
    color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: name + " body" });
}
function stat(s, x, y, w, big, label, colorKey, name) {
  s.addText(big, { x, y, w, h: 0.95, fontSize: 44, bold: true, color: C[colorKey],
    fontFace: THEME.headFontFace, margin: 0, isTextBox: true, objectName: name + " value" });
  s.addText(label, { x, y: y + 0.95, w, h: 0.75, fontSize: 13, color: C.text1, margin: 0,
    valign: "top", isTextBox: true, objectName: name + " label" });
}
function bullets(s, items, x, y, w, h, size = 16) {
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: true,
    breakLine: i < items.length - 1 } })), { x, y, w, h, fontSize: size, color: C.text1,
    paraSpaceAfter: 8, valign: "top", isTextBox: true, margin: 0 });
}
const f3 = (x) => (x == null || Number.isNaN(x) ? "–" : Number(x).toFixed(3));

// ================================================================== 1. title
let s = newSlide("TITLE_DARK", "Materials Discovery with Semi-Supervised Learning", "Introduction");
s.addText("Predicting stability, magnetism, metallicity and band gap of inorganic materials when labels are scarce",
  { placeholder: "body" });
s.addImage({ path: FIG("logos/crystal.png"), x: 0.8, y: 0.5, w: 1.5, h: 1.5,
  sizing: { type: "contain", w: 1.5, h: 1.5 }, objectName: "crystal logo" });
s.addImage({ path: FIG("logos/esi.png"), x: 11.1, y: 0.5, w: 1.45, h: 1.45,
  sizing: { type: "contain", w: 1.45, h: 1.45 }, objectName: "ESI logo" });
s.addText("Laref Ibrahim · Nemouchi Faycal Abdelhadi · Zerradi Abdelbasset · Medelci Djezzar Aymen",
  { x: 0.8, y: 5.6, w: 11.7, h: 0.5, fontSize: 16, color: C.background1, margin: 0, isTextBox: true });
s.addText("ESI-SBA · Machine Learning (Dr. N. Dif) · Mini-project",
  { x: 0.8, y: 6.15, w: 11.7, h: 0.4, fontSize: 13, color: "CADCFC", margin: 0, isTextBox: true });
s.addNotes("Introduce the team and the theme: materials discovery with semi-supervised learning (MI-SSL).");

// ================================================================== 2. problem
s = newSlide("CONTENT", "The challenge: stability and functionality", "Introduction");
card(s, 0.6, 1.55, 5.9, 2.15, "Predicting stability",
  "A material must lie on or near the convex hull to be synthesized. Testing every candidate with DFT costs hours to days of computing.",
  "accent1", "stability");
card(s, 6.85, 1.55, 5.9, 2.15, "Predicting functionality",
  "Metal or insulator, magnetic or not, band gap in the solar window: these decide whether a material is useful.",
  "accent2", "function");
s.addText("“Is this composition stable enough to be made, and does it have the property we need?”",
  { x: 0.6, y: 4.4, w: 12.1, h: 0.8, fontSize: 20, italic: true, color: C.text2, align: "center",
    isTextBox: true, margin: 0 });
stat(s, 0.9, 5.3, 3.6, "≈ hours", "of DFT computing per labeled material", "accent6", "dft");
stat(s, 4.9, 5.3, 3.6, "554k", "unlabeled compositions available (GNoME)", "accent1", "gnome");
stat(s, 8.9, 5.3, 3.6, "1,000", "labels used to train each model in our study", "accent3", "budget");
s.addNotes("Labels are expensive (DFT or experiments) while unlabeled candidate compositions are abundant: the data dilemma that motivates semi-supervised learning.");

// ================================================================== 3. approach
s = newSlide("CONTENT", "Our answer: three kinds of learning without full labels", "Introduction");
card(s, 0.6, 1.5, 3.85, 3.6, "Semi-supervised",
  "Label Propagation, Label Spreading, Self-Training, Co-Training and Cluster-then-Label learn from 1,000 labels plus up to 100,000 unlabeled materials.",
  "accent1", "ssl");
card(s, 4.75, 1.5, 3.85, 3.6, "Unsupervised",
  "Apriori association rules (elements ⇒ property) and PCA + K-Means clustering of the materials space.",
  "accent2", "unsup");
card(s, 8.9, 1.5, 3.85, 3.6, "Self-supervised",
  "A masked autoencoder pre-trained on 554k unlabeled GNoME compositions provides a learned materials embedding.",
  "accent3", "selfsup");
s.addText("Plus a traditional pipeline for the photovoltaic band gap: genetic-algorithm descriptor selection and applicability domain.",
  { x: 0.6, y: 5.5, w: 12.1, h: 0.8, fontSize: 16, color: C.text1, isTextBox: true, margin: 0 });
s.addNotes("The assignment allows unsupervised (association), semi-supervised, weakly or self-supervised learning. We cover three of them; supervised models are only the comparison baseline.");

// ================================================================== 4. datasets
s = newSlide("CONTENT", "Large, open materials databases", "Data");
const rows = [
  ["Task", "Source", "Unique compositions", "Positive rate"],
  ["Metallicity", "Materials Project (Matbench)", V.MetalN, V.MetalPos + " %"],
  ["Magnetism", "JARVIS-DFT (NIST)", V.MagN, V.MagPos + " %"],
  ["Thermodynamic stability", "Alexandria (500,000 DFT entries)", V.StabN, V.StabPos + " %"],
  ["PV band gap (traditional)", "Materials Project (Matbench)", V.GapN, V.GapPos + " %"],
  ["Unlabeled pool", "GNoME (Google DeepMind)", "554,054", "–"],
];
s.addTable(rows.map((r, i) => r.map((c) => ({ text: String(c), options: {
  bold: i === 0, color: i === 0 ? C.background1 : C.text1,
  fill: { color: i === 0 ? C.text2 : (i % 2 ? C.background1 : C.background2) } } }))),
  { x: 0.6, y: 1.5, w: 12.1, colW: [3.0, 4.6, 2.5, 2.0], fontSize: 15, rowH: 0.55,
    border: { type: "solid", color: "E2E8F0", pt: 1 }, objectName: "datasets table" });
s.addText("One row per unique composition (reduced formula): duplicated polymorphs are merged, which prevents train/test leakage. Stratified 80/20 split; the test set is never used for training or tuning.",
  { x: 0.6, y: 5.3, w: 12.1, h: 1.0, fontSize: 15, color: C.text1, isTextBox: true, margin: 0 });
s.addNotes("All datasets are open and need no API key. Around 1.1 million compositions in total.");

// ================================================================== 5. features
s = newSlide("CONTENT", "From a chemical formula to 142 descriptors", "Data");
image(s, "schematics/mi_principle.png", 0.6, 1.4, 12.1, 3.0, "MI principle");
image(s, "schematics/descriptor_examples.png", 0.6, 4.5, 8.0, 2.3, "descriptor examples");
stat(s, 9.0, 4.6, 3.7, "3.6 s", "for 240,000 compositions with our vectorised Magpie featurizer (identical to matminer)", "accent4", "speed");
s.addNotes("Magpie statistics (min, max, range, mean, average deviation, mode) of 22 elemental properties, plus stoichiometry and valence-orbital fractions.");

// ================================================================== 6. EDA
s = newSlide("CONTENT", "Exploring the data: labeled vs unlabeled", "Data");
image(s, "stability/labeled_vs_unlabeled.png", 0.6, 1.4, 8.4, 4.3, "labeled vs unlabeled");
bullets(s, [
  "Kolmogorov–Smirnov tests: GNoME is close to Alexandria (KS mostly < 0.25)…",
  "…but far from the Materials Project data (KS up to 0.48)",
  "This distribution shift explains much of the SSL behaviour",
  "0 missing values; 0.8–1 % of values clipped as outliers (mean ± 3σ)",
], 9.3, 1.5, 3.5, 4.8, 15);
s.addNotes("Show that the unlabeled pool does not come from exactly the same distribution as the labeled sets.");

// ================================================================== 7. preprocessing
s = newSlide("CONTENT", "Preprocessing: before and after", "Data");
image(s, "stability/normalization_before_after.png", 0.6, 1.4, 12.1, 4.7, "normalization");
s.addText("Median imputation → clipping at mean ± 3σ → z-score (StandardScaler fitted on training data only, saved for inference)",
  { x: 0.6, y: 6.25, w: 12.1, h: 0.6, fontSize: 15, color: C.text1, isTextBox: true, margin: 0 });
s.addNotes("Requirement 6: visualizations of the dataset before and after preprocessing.");

// ================================================================== 8. methods
s = newSlide("CONTENT", "Semi-supervised methods compared", "Methods");
const meth = [
  ["Self-Training", "Pseudo-labels with confidence ≥ τ (0.75 / 0.9), iteratively", "accent1"],
  ["Co-Training", "Two views: elemental properties vs stoichiometry/valence", "accent2"],
  ["Label Propagation", "Similarity graph (RBF / 10-NN), hard labels", "accent3"],
  ["Label Spreading", "Normalized graph, soft labels (α = 0.2 / 0.8)", "accent4"],
  ["Cluster-then-Label", "K-Means (k = 100), label-pure clusters transfer labels", "accent5"],
  ["Baselines", "RF, ExtraTrees, XGBoost, LightGBM, LogReg, kNN, MLP + Optuna", "accent6"],
];
meth.forEach(([h, b, col], i) => {
  const x = 0.6 + (i % 3) * 4.1, y = 1.6 + Math.floor(i / 3) * 2.6;
  card(s, x, y, 3.85, 2.2, h, b, col, "method " + i);
});
s.addNotes("Graph methods use class mass normalization (Zhu 2003) to avoid collapsing to the majority class. Tree learners: Random Forest reference (as in the previous QSAR project) and a tuned LightGBM.");

// ================================================================== 9. protocol
s = newSlide("CONTENT", "Experimental protocol", "Methods");
const steps = [
  ["1,000", "labeled compositions (stratified)"],
  ["1k → 100k", "unlabeled: GNoME or same DB (labels hidden)"],
  ["3 seeds", "every configuration, mean ± std"],
  ["Composite", "= (F1 + ROC-AUC) / 2 for model selection"],
];
steps.forEach(([big, lab], i) => {
  const x = 0.6 + i * 3.1;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.6, w: 2.8, h: 2.6, rectRadius: 0.12,
    fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "step " + i });
  s.addText(big, { x: x + 0.2, y: 1.85, w: 2.4, h: 0.9, fontSize: 30, bold: true, color: C.accent1,
    fontFace: THEME.headFontFace, margin: 0, isTextBox: true });
  s.addText(lab, { x: x + 0.2, y: 2.8, w: 2.4, h: 1.2, fontSize: 14, color: C.text1, margin: 0,
    isTextBox: true });
  if (i < 3) s.addText("→", { x: x + 2.8, y: 2.5, w: 0.3, h: 0.6, fontSize: 24, color: C.accent5,
    align: "center", margin: 0, isTextBox: true });
});
const runs = ["metallicity", "magnetism", "stability"].flatMap((t) => ["indist", "gnome"].map((p) => summary(t, p)))
  .filter(Boolean).reduce((a, S) => a + S.n_training_runs, 0);
stat(s, 0.6, 4.7, 5.8, runs ? runs.toLocaleString("en-US") : "–", "models trained and evaluated in the SSL campaign (incl. Optuna trials)", "accent3", "runs");
stat(s, 6.9, 4.7, 5.8, "20,000", "test compositions evaluated per task (never seen in training)", "accent4", "test");
s.addNotes("Multiple training sessions varying algorithms and hyper-parameters (requirement 4).");

// ================================================================== 10-12. results per task
const ORDER = ["Baseline", "Self-Supervised (AE)", "Self-Training", "Co-Training", "Cluster-then-Label",
  "Label Propagation", "Label Spreading"];
const TASKS = [["metallicity", "Metallicity (Materials Project)"], ["magnetism", "Magnetism (JARVIS-DFT)"],
  ["stability", "Stability (Alexandria)"]];
for (const [task, title] of TASKS) {
  s = newSlide("CONTENT", `Results: ${title}`, "Results");
  const SI = summary(task, "indist"), SG = summary(task, "gnome");
  const series = [];
  for (const [S, name] of [[SI, "Unlabeled = same database"], [SG, "Unlabeled = GNoME"]]) {
    if (!S) continue;
    const by = Object.fromEntries(S.best_per_method.map((r) => [r.Method, r.Composite_Score_mean]));
    series.push({ name, labels: ORDER, values: ORDER.map((m) => +(by[m] || 0).toFixed(4)) });
  }
  if (series.length) {
    const all = series.flatMap((x) => x.values).filter((v) => v > 0);
    s.addChart(pres.charts.BAR, series, {
      x: 0.6, y: 1.35, w: 7.9, h: 5.5, barDir: "bar", barGrouping: "clustered",
      chartColors: series.length > 1 ? [HEX.accent1, HEX.accent3] : [HEX.accent1, HEX.accent1],
      valAxisMinVal: Math.max(0, Math.floor((Math.min(...all) - 0.05) * 10) / 10),
      valAxisMaxVal: Math.min(1, Math.ceil((Math.max(...all) + 0.03) * 20) / 20),
      showValue: true, dataLabelFormatCode: "0.000", dataLabelFontSize: 9, dataLabelColor: HEX.dk1,
      catAxisLabelFontSize: 11, valAxisLabelFontSize: 10, catAxisLabelColor: HEX.dk1,
      valAxisLabelColor: HEX.accent5, valGridLine: { color: "E2E8F0", size: 0.5 },
      catGridLine: { style: "none" }, showLegend: series.length > 1, legendPos: "b", legendFontSize: 11,
      catAxisOrientation: "maxMin", catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
      dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt",
      showTitle: true, title: "Composite score of the best configuration of each method (1,000 labels, 3 seeds)",
      titleFontSize: 12, titleColor: HEX.dk2, titleFontFace: "+mn-lt", objectName: task + " chart",
    });
  } else {
    image(s, "missing.png", 0.6, 1.35, 7.9, 5.5);
  }
  const best = SI || SG;
  if (best) {
    const S = [SI, SG].filter(Boolean).sort((a, b) => b.best_model_overall.Composite_Score_mean - a.best_model_overall.Composite_Score_mean)[0];
    stat(s, 8.9, 1.5, 3.9, f3(S.best_model_overall.Composite_Score_mean),
      `best: ${S.best_model_overall.Method} (${S.best_model_overall.Model_Config.split(" | ")[0]})`, "accent1", task + " best");
    stat(s, 8.9, 3.35, 3.9, (S.ssl_improvement_over_best_baseline_pct >= 0 ? "+" : "") + S.ssl_improvement_over_best_baseline_pct.toFixed(2) + " %",
      "best SSL vs tuned supervised baseline", S.ssl_improvement_over_best_baseline_pct >= 0 ? "accent4" : "accent6", task + " gain");
    stat(s, 8.9, 5.2, 3.9, f3(S.fully_supervised_upper_bound.Composite_Score),
      `fully supervised upper bound (${S.training_setup.n_train_pool.toLocaleString("en-US")} labels)`, "accent5", task + " full");
  }
  s.addNotes(`Selection criterion: highest mean composite score over 3 seeds. Pool sizes 1k-100k; the chart shows each method's best configuration.`);
}

// ================================================================== 13. budget curve
s = newSlide("CONTENT", "How much do labels matter?", "Results");
image(s, "results/magnetism/indist/magnetism_label_budget_curve.png", 0.6, 1.4, 6.0, 4.6, "budget magnetism");
image(s, "results/stability/indist/stability_learning_curves.png", 6.8, 1.4, 6.0, 4.6, "learning stability");
s.addText("Left: label-budget curve (250 → 5,000 labels) against the fully-supervised bound. Right: composite score as the unlabeled pool grows.",
  { x: 0.6, y: 6.15, w: 12.1, h: 0.7, fontSize: 14, color: C.text1, isTextBox: true, margin: 0 });
s.addNotes("Adding labels helps much more than adding unlabeled data; external unlabeled data (GNoME) can even hurt because of distribution shift.");

// ================================================================== 14. GA
s = newSlide("CONTENT", "Band gap: genetic-algorithm descriptor selection", "Traditional pipeline");
image(s, "results/band_gap/band_gap_ga_evolution.png", 0.6, 1.4, 8.2, 3.6, "GA evolution");
stat(s, 9.2, 1.45, 3.6, `142 → ${V.GAFinalN}`, "descriptors kept by the GA (−63 %)", "accent1", "ga n");
stat(s, 9.2, 3.25, 3.6, V.GapGAAUC, `test ROC-AUC with ${V.GAFinalN} descriptors (all 142: ${V.GapAllAUC})`, "accent4", "ga auc");
stat(s, 0.9, 5.2, 3.8, V.GapTunedFone, `test F1 after threshold calibration (was ${V.GapGAFone})`, "accent3", "ga f1");
stat(s, 5.0, 5.2, 3.8, V.GAFinalAUC, `CV ROC-AUC of the best subset (all descriptors: ${V.GAAllCV})`, "accent2", "ga cv");
s.addNotes("Population 100, 20 generations, crossover 0.8, mutation 0.2, elite 5, fitness = ROC-AUC − 0.05 × fraction of descriptors. Best model family: Random Forest. Target = PBE band gap in 1.1-1.7 eV.");

// ================================================================== 15. AD
s = newSlide("CONTENT", "Applicability domain: when to trust a prediction", "Traditional pipeline");
image(s, "results/band_gap/band_gap_ad_distance.png", 0.6, 1.4, 8.2, 3.6, "AD distance");
stat(s, 9.2, 1.45, 3.6, V.ADFracIn + " %", `of ${V.ADNTest} test compositions inside the domain`, "accent4", "ad in");
stat(s, 9.2, 3.25, 3.6, `${V.ADRecInT} vs ${V.ADRecOutT}`, "recall of PV materials inside vs outside the domain", "accent6", "ad recall");
bullets(s, ["PCA: 2 components explain " + V.ADPCAVar + " % of the variance",
  "K-Means (k = 6): train and test cluster shares match within 0.5 %",
  "The web app reports the AD status with every prediction"], 0.6, 5.3, 12.1, 1.5, 15);
s.addNotes("Distance-based AD: mean distance to the 5 nearest training compositions compared with its 95th percentile on the training set.");

// ================================================================== 16. unsupervised
s = newSlide("CONTENT", "Unsupervised learning rediscovers chemistry", "Unsupervised");
image(s, "results/unsupervised/association_rules_magnetism.png", 0.6, 1.4, 6.2, 4.3, "rules magnetism");
image(s, "results/unsupervised/kmeans_cluster_property_rates.png", 7.0, 1.4, 5.8, 4.3, "clusters");
bullets(s, ["{Mn, O} ⇒ magnetic: 95 % confidence, lift 4.6 · rare-earth pairs ⇒ stable (lift 2.2) · Se, S, Br ⇒ PV band gap",
  `K-Means finds ${V.KBest} material families without labels: intermetallics (≈ 90 % metals) vs ionic O/F/S compounds (22 % metals, 31 % magnetic)`],
  0.6, 5.85, 12.1, 1.1, 14);
s.addNotes("Association rules = the unsupervised technique explicitly named in the assignment. Self-supervised autoencoder: masked-feature reconstruction R² = " + V.AERtwo + ".");

// ================================================================== 17. best models
s = newSlide("CONTENT", "Selected models and selection criteria", "Results");
const reg = fs.existsSync(path.join(ROOT, "app", "models", "registry.json"))
  ? JSON.parse(fs.readFileSync(path.join(ROOT, "app", "models", "registry.json"), "utf8")) : {};
const tr = [["Task", "Deployed method", "Configuration", "Test ROC-AUC", "Test F1"]];
for (const [t, n] of [["metallicity", "Metallicity"], ["magnetism", "Magnetism"], ["stability", "Stability"], ["band_gap", "PV band gap"]]) {
  const r = reg[t];
  tr.push([n, r ? r.method : "pending", r ? String(r.config) : "–",
    r ? f3(r.test_metrics["ROC-AUC"]) : "–", r ? f3(r.test_metrics["F1-Score"]) : "–"]);
}
s.addTable(tr.map((r, i) => r.map((c) => ({ text: String(c), options: { bold: i === 0,
  color: i === 0 ? C.background1 : C.text1, fill: { color: i === 0 ? C.text2 : (i % 2 ? C.background1 : C.background2) } } }))),
  { x: 0.6, y: 1.45, w: 12.1, colW: [2.0, 3.0, 3.9, 1.6, 1.6], fontSize: 13, rowH: 0.55,
    border: { type: "solid", color: "E2E8F0", pt: 1 }, objectName: "models table" });
bullets(s, ["Criterion: highest mean Composite = (F1 + ROC-AUC)/2 over 3 seeds at 1,000 labels",
  "Deployment: the selected architecture is re-trained on all training labels (+ GNoME unlabeled pool)",
  "Ties are broken in favour of the simpler model"], 0.6, 4.6, 12.1, 2.0, 15);
s.addNotes("Requirement 6: the best model and the criteria used to choose it.");

// ================================================================== 18-19. app
function shotPair(title, left, right, capL, capR, notes) {
  s = newSlide("CONTENT", title, "Application");
  [[left, capL], [right, capR]].forEach(([p, cap], i) => {
    const x = 0.6 + i * 6.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.35, w: 5.95, h: 4.75, rectRadius: 0.08,
      fill: { color: C.background2 }, line: { color: "E2E8F0", width: 1 }, objectName: "frame " + p });
    image(s, p, x + 0.1, 1.45, 5.75, 4.55, p);
    s.addText(cap, { x, y: 6.2, w: 5.95, h: 0.7, fontSize: 14, color: C.text1, margin: 0,
      valign: "top", isTextBox: true, objectName: "caption " + p });
  });
  s.addNotes(notes);
}
shotPair("MatSSL Studio: compose, predict, explain",
  "screenshots/deck/studio.png", "screenshots/deck/results.png",
  "Periodic-table editor synced with the formula; 3D crystal from the Alexandria database",
  "Four calibrated predictions, applicability domain and position in chemical space",
  "React + Vite frontend, FastAPI backend, one Docker image. Live demo: LiFePO4 then Nd2Fe14B. Every prediction shows the deployed method, its test scores and whether the material lies inside the applicability domain.");
shotPair("Why this prediction? Explanations and similar materials",
  "screenshots/deck/explain.png", "screenshots/deck/compare.png",
  "Exact SHAP contributions (LightGBM) and nearest known materials in the self-supervised embedding",
  "Compare up to four candidates, e.g. battery cathodes LiFePO4 · LiMnPO4 · LiCoO2",
  "The 32-d masked-autoencoder embedding powers the nearest-material search; association rules mined by Apriori are matched to the query.");
shotPair("Research Lab: every experiment, interactive",
  "screenshots/deck/lab_ssl.png", "screenshots/deck/lab_models.png",
  "SSL comparison, learning curves and label budget for every task and unlabeled pool",
  "Deployed models with test confusion matrices and ROC curves",
  "The Research Lab reads a static JSON export of all result files, so it also works offline during the presentation.");

// ================================================================== 19. discussion
s = newSlide("CONTENT", "Discussion and limitations", "Conclusion");
bullets(s, [
  "With a strong tuned learner, SSL gains are small: semi-supervised learning is not a universal solution",
  "Unlabeled data must come from the same distribution: the external GNoME pool shifts the data and can hurt",
  "Composition-only descriptors limit stability prediction (structure matters)",
  "DFT-PBE band gaps underestimate experimental gaps",
], 0.6, 1.5, 6.4, 5.0, 16);
image(s, "schematics/active_learning_loop.png", 7.2, 1.4, 5.6, 4.9, "active learning loop");
s.addNotes("Future work: semi-supervised active learning, structure-based graph neural networks, uncertainty estimates.");

// ================================================================== 20. conclusion
s = newSlide("SECTION_DARK", "Conclusion", "Conclusion");
s.addText("1.1 million compositions · 5 SSL methods · unsupervised rules and clusters · self-supervised embedding · GA + applicability domain · deployed web app",
  { placeholder: "body" });
s.addText("Thank you — questions?", { x: 0.8, y: 5.4, w: 11.7, h: 0.8, fontSize: 24, bold: true,
  color: C.background1, isTextBox: true, margin: 0 });
s.addNotes("Summarize the findings and open the discussion.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
