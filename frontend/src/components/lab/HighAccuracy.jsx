import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartTooltip, SectionHeader } from "../ui.jsx";
import { Confusion } from "./Models.jsx";
import { AXIS, GRID, SERIES, TASK, big, fmt, pct } from "../../lib/theme.js";

const ORDER = ["metallicity", "magnetism", "stability", "band_gap"];
const STEPS = ["Composition only", "+ structure", "+ tuning", "+ stacking", "+ accuracy threshold"];

export default function HighAccuracy({ data }) {
  const H = data.high_accuracy || {};
  const tasks = ORDER.filter((t) => H[t]);
  if (!tasks.length) {
    return <div className="card p-8 text-center text-ink-mute">The structure-aware models are still training.</div>;
  }
  const chart = tasks.map((t) => {
    const row = { task: TASK[t].title };
    H[t].ablation.forEach((r, i) => { row[STEPS[i]] = r.Accuracy; });
    return row;
  });
  const all = tasks.flatMap((t) => H[t].ablation.map((r) => r.Accuracy));
  const lo = Math.max(0, Math.floor((Math.min(...all) - 0.03) * 50) / 50);
  return (
    <div className="space-y-6">
      <SectionHeader title="High-accuracy track: all labels + crystal structure + ensemble"
        subtitle="Beyond the 1,000-label SSL study: every structure of each database, 142 composition + 28 crystal-structure descriptors (symmetry, density, packing, bond lengths, coordination), Optuna-tuned gradient boosting, a stacked ensemble and a decision threshold chosen on the training set to maximise accuracy. Train/test split grouped by formula: no composition is on both sides." />
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        {tasks.map((t) => {
          const S = H[t], f = S.final, base = S.ablation[0];
          const gain = f.Accuracy - base.Accuracy;
          return (
            <div key={t} className="card p-5 space-y-3">
              <div>
                <div className="label">{TASK[t].source}</div>
                <div className="font-display text-xl font-bold">{TASK[t].title}</div>
                <div className="text-xs text-ink-mute">{big(S.dataset.n_structures)} structures · {big(S.dataset.n_formulas)} formulas</div>
              </div>
              <div className="flex items-end gap-3">
                <div className="font-display text-4xl font-bold text-brand">{pct(f.Accuracy)}</div>
                <div className={`pb-1 text-xs font-semibold ${gain > 0.0005 ? "text-good" : "text-ink-mute"}`}>
                  {Math.abs(gain) < 0.0005 ? "±0.0" : `${gain > 0 ? "+" : "−"}${Math.abs(100 * gain).toFixed(1)}`} pts vs composition only
                </div>
              </div>
              <div className="grid grid-cols-3 gap-2 text-center">
                {[["Balanced", f.Balanced_Accuracy], ["F1", f["F1-Score"]], ["ROC-AUC", f["ROC-AUC"]]].map(([l, v]) => (
                  <div key={l} className="rounded-lg bg-surface py-2"><div className="label">{l}</div><div className="font-display text-lg font-bold">{fmt(v)}</div></div>
                ))}
              </div>
              <Confusion cm={S.confusion} pos={TASK[t].positive} neg={TASK[t].negative} />
              <div className="text-[11px] text-ink-mute">{big(S.dataset.n_test)} held-out test structures · threshold {S.threshold.toFixed(2)} · {S.base_models.join(" + ")}</div>
            </div>
          );
        })}
      </div>
      <div className="card p-5">
        <SectionHeader title="What each ingredient adds" subtitle="Test accuracy after each step (same held-out structures for every step)." />
        <ResponsiveContainer width="100%" height={360}>
          <BarChart data={chart} margin={{ right: 20, bottom: 5 }}>
            <CartesianGrid {...GRID} vertical={false} />
            <XAxis dataKey="task" {...AXIS} />
            <YAxis domain={[lo, 1]} tickFormatter={(v) => `${Math.round(100 * v)}%`} {...AXIS} />
            <Tooltip content={<ChartTooltip digits={4} />} cursor={{ fill: "#F1F5F9" }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {STEPS.map((s, i) => <Bar isAnimationActive={false} key={s} dataKey={s} fill={SERIES[i]} radius={[4, 4, 0, 0]} />)}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="card p-5 overflow-x-auto">
        <SectionHeader title="Ablation table" subtitle="Accuracy, balanced accuracy, F1 and ROC-AUC on the held-out test set." />
        <table className="w-full text-sm">
          <thead><tr className="text-left text-xs text-ink-mute border-b border-line"><th className="py-2">Task</th><th>Step</th><th>Accuracy</th><th>Balanced acc.</th><th>F1</th><th>ROC-AUC</th></tr></thead>
          <tbody>
            {tasks.flatMap((t) => H[t].ablation.map((r, i) => (
              <tr key={t + i} className={`border-b border-line/60 ${i === H[t].ablation.length - 1 ? "bg-brand-50 font-semibold" : ""}`}>
                <td className="py-1.5">{i === 0 ? TASK[t].title : ""}</td><td>{r.Model.replace(/^\d\. /, "")}</td>
                <td className="font-mono">{fmt(r.Accuracy, 4)}</td><td className="font-mono">{fmt(r.Balanced_Accuracy, 4)}</td>
                <td className="font-mono">{fmt(r["F1-Score"], 4)}</td><td className="font-mono">{fmt(r["ROC-AUC"], 4)}</td>
              </tr>
            )))}
          </tbody>
        </table>
        <p className="mt-3 text-xs text-ink-soft">For the PV band gap only {pct(H.band_gap?.dataset.positive_rate)} of structures are positive, so a model answering “no” everywhere would already reach {pct(1 - (H.band_gap?.dataset.positive_rate ?? 0))} accuracy: balanced accuracy is the fairer number there.</p>
      </div>
    </div>
  );
}
