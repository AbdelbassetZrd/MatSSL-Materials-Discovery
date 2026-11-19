import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Boxes, Download, Scale } from "lucide-react";
import { api } from "../api.js";
import { useComposition } from "../lib/composition.jsx";
import { ErrorBox, Loading, Segmented } from "../components/ui.jsx";
import ProfileRadar from "../components/ProfileRadar.jsx";
import { ADBadge, ChemicalSpaceMap, Explanation, NeighborsTable, PropertyCard } from "../components/ResultParts.jsx";
import { TASK } from "../lib/theme.js";

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

function StructureBanner({ r }) {
  if (r.structure) {
    const s = r.structure;
    return (
      <div className="flex items-start gap-2 rounded-xl bg-cyan/10 px-4 py-3 text-sm text-ink">
        <Boxes size={18} className="mt-0.5 shrink-0 text-cyan" />
        <span><b>Structure-aware models used.</b> Lowest-energy known structure: {s.source} <span className="font-mono">{s.id}</span>, space group {s.spacegroup}, {s.n_atoms_primitive} atoms in the primitive cell, density {s.density.toFixed(2)} g/cm³{s.e_above_hull != null && <> · DFT reference: {s.e_above_hull.toFixed(3)} eV/atom above the convex hull</>}.</span>
      </div>
    );
  }
  const why = { unknown: "No known crystal structure for this composition", offline: "Structure database unreachable", unused: "No structure-aware model installed" }[r.structure_status];
  return why ? (
    <div className="rounded-xl bg-surface px-4 py-3 text-sm text-ink-soft"><b className="text-ink">{why}:</b> the predictions use the composition-only semi-supervised models.</div>
  ) : null;
}

export default function Results() {
  const [params] = useSearchParams();
  const ctx = useComposition();
  const formula = params.get("formula") || ctx.formula;
  useEffect(() => {
    if (params.get("formula") && params.get("formula") !== ctx.text) ctx.setFormulaText(params.get("formula"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params]);

  const pred = useQuery({ queryKey: ["predict", formula], queryFn: () => api.predict(formula), enabled: !!formula });
  const ana = useQuery({ queryKey: ["analyze", formula], queryFn: () => api.analyze(formula), enabled: !!formula });
  const sim = useQuery({ queryKey: ["similar", formula], queryFn: () => api.similar(formula, 8), enabled: !!formula, retry: 0 });
  const exp = useQuery({ queryKey: ["experiments"], queryFn: api.experiments });
  const [why, setWhy] = useState("magnetism");

  if (!formula) return <p>Build a composition in the <Link className="text-brand underline" to="/studio">Studio</Link> first.</p>;
  if (pred.isLoading) return <Loading text={`Running the models on ${formula}…`} />;
  if (pred.error) return <ErrorBox error={pred.error} />;
  const r = pred.data;
  const tasks = r.predictions.map((p) => p.task);
  const whyTask = tasks.includes(why) ? why : tasks[0];
  const a = ana.data;

  const exportCSV = () => {
    const rows = [["formula", "task", "prediction", "probability", "threshold", "method", "test_roc_auc"],
      ...r.predictions.map((p) => [formula, p.task, p.prediction_label, p.probability.toFixed(4), p.threshold, p.method, p.test_metrics?.["ROC-AUC"] ?? ""])];
    download(`${formula}_predictions.csv`, rows.map((x) => x.join(",")).join("\n"), "text/csv");
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <Link to="/studio" className="text-sm text-ink-mute hover:text-ink inline-flex items-center gap-1"><ArrowLeft size={14} /> Studio</Link>
          <h1 className="font-display text-4xl font-bold mt-1">
            <span className="font-mono">{a?.reduced_formula || formula}</span>
          </h1>
          {a && <p className="text-ink-soft text-sm">{a.n_elements} elements · formula weight {a.weight.toFixed(2)} u · {a.chemical_rules.charge_balanced ? "charge-balanced" : "not charge-balanced"}</p>}
        </div>
        <div className="flex gap-2">
          <Link className="btn-ghost text-sm" to={`/compare?f=${encodeURIComponent(formula)}`}><Scale size={16} /> Compare</Link>
          <button className="btn-ghost text-sm" onClick={exportCSV}><Download size={16} /> CSV</button>
          <button className="btn-ghost text-sm" onClick={() => download(`${formula}_predictions.json`, JSON.stringify({ ...r, analysis: a }, null, 2), "application/json")}><Download size={16} /> JSON</button>
        </div>
      </div>

      <ADBadge ad={r.applicability_domain} />
      <StructureBanner r={r} />

      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        {r.predictions.map((p) => <PropertyCard key={p.task} p={p} reference={p.task === "stability" ? r.structure?.e_above_hull : null} />)}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card p-5">
          <h2 className="font-display text-xl font-bold">Materials profile</h2>
          <p className="text-xs text-ink-soft">Predicted probability of each property (0–100%).</p>
          <ProfileRadar series={[{ name: formula, predictions: r.predictions }]} height={320} />
        </div>
        <div className="card p-5">
          <h2 className="font-display text-xl font-bold">Where is this material in chemical space?</h2>
          <p className="text-xs text-ink-soft">PCA of the 52 GA-selected descriptors · 3,000 training compositions · the star is your material.</p>
          <ChemicalSpaceMap points={exp.data?.band_gap?.chemical_space} query={r.applicability_domain?.pca} height={320} />
        </div>
      </div>

      <div className="card p-5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <div>
            <h2 className="font-display text-xl font-bold">Why this prediction?</h2>
            <p className="text-xs text-ink-soft">Exact SHAP values for gradient-boosting based models; descriptor percentiles otherwise.</p>
          </div>
          <Segmented size="md" value={whyTask} onChange={setWhy} options={tasks.map((t) => ({ value: t, label: TASK[t]?.title || t }))} />
        </div>
        <Explanation p={r.predictions.find((p) => p.task === whyTask)} />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)] gap-6">
        <div className="card p-5">
          <h2 className="font-display text-xl font-bold mb-1">Nearest known materials</h2>
          {sim.isLoading ? <Loading /> : sim.error ? <ErrorBox error={sim.error} /> : <NeighborsTable data={sim.data} />}
        </div>
        <div className="card p-5 space-y-4">
          <div>
            <h2 className="font-display text-xl font-bold">Chemistry checks</h2>
            {a && (
              <ul className="mt-2 space-y-1 text-sm">
                <li className={a.chemical_rules.charge_balanced ? "text-good" : "text-amber-700"}>
                  {a.chemical_rules.charge_balanced ? "✓" : "!"} Charge balance
                  {a.chemical_rules.oxidation_states && <span className="text-ink-soft"> – {Object.entries(a.chemical_rules.oxidation_states).map(([el, v]) => `${el}${v > 0 ? "+" : ""}${v}`).join(", ")}</span>}
                </li>
                <li className={a.chemical_rules.n_elements_ok ? "text-good" : "text-amber-700"}>{a.chemical_rules.n_elements_ok ? "✓" : "!"} {a.n_elements} elements (training range ≤ 6)</li>
              </ul>
            )}
          </div>
          <div>
            <h3 className="font-semibold">Association rules that apply</h3>
            <p className="text-xs text-ink-soft mb-2">Mined without supervision (Apriori) on {">"}650k compositions.</p>
            <div className="flex flex-wrap gap-2">
              {a?.association_rules_matched?.length ? a.association_rules_matched.map((m) => (
                <span key={m.rule} className="chip !cursor-default">
                  <span className="font-mono text-xs">{m.rule.replace("->", "⇒")}</span>
                  <b className="text-brand">{(100 * m.confidence).toFixed(0)}%</b>
                  <span className="text-ink-mute text-xs">lift {m.lift.toFixed(1)}</span>
                </span>
              )) : <span className="text-sm text-ink-mute">No mined rule matches this element set.</span>}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
