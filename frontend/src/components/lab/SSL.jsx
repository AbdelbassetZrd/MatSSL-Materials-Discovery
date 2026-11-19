import { useState } from "react";
import {
  Bar, BarChart, CartesianGrid, ErrorBar, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip,
  XAxis, YAxis, Cell, LabelList,
} from "recharts";
import { ChartTooltip, SectionHeader, Segmented, Stat } from "../ui.jsx";
import { AXIS, GRID, METHOD_COLOR, METHOD_ORDER, POOL, TASK, fmt, big } from "../../lib/theme.js";

const TASKS = ["metallicity", "magnetism", "stability"];
const DASH = { "Self-Training": "", "Co-Training": "6 3", "Label Propagation": "2 3", "Label Spreading": "8 3 2 3", "Cluster-then-Label": "4 4" };

export default function SSL({ data }) {
  const [task, setTask] = useState("magnetism");
  const avail = Object.entries(data.ssl[task] || {}).filter(([, v]) => v).map(([k]) => k);
  const [poolPref, setPool] = useState("indist");
  const pool = avail.includes(poolPref) ? poolPref : avail[0];
  const S = pool ? data.ssl[task][pool] : null;

  const taskOptions = TASKS.map((t) => ({ value: t, label: TASK[t].title }));
  if (!S) {
    return (
      <div className="space-y-4">
        <Segmented size="md" value={task} onChange={setTask} options={taskOptions} />
        <div className="card p-8 text-center text-ink-mute">Experiments for {TASK[task].title} are still running.</div>
      </div>
    );
  }

  const byMethod = Object.fromEntries(S.best_per_method.map((r) => [r.Method, r]));
  const bars = METHOD_ORDER.filter((m) => byMethod[m]).map((m) => ({
    method: m, composite: byMethod[m].Composite_Score_mean, err: byMethod[m].Composite_Score_std || 0,
    config: byMethod[m].Model_Config,
  }));
  const base = S.best_baseline.composite_score;
  const lo = Math.max(0, Math.floor((Math.min(...bars.map((b) => b.composite)) - 0.05) * 20) / 20);

  const ns = [...new Set(Object.values(S.pool_curves).flatMap((c) => c.points.map((p) => p.n)))].sort((a, b) => a - b);
  const curve = ns.map((n) => {
    const row = { n };
    Object.entries(S.pool_curves).forEach(([m, c]) => { const p = c.points.find((x) => x.n === n); if (p) row[m] = p.composite; });
    return row;
  });
  const budgets = [...new Set(S.budget_curve.map((b) => b.budget))].sort((a, b) => a - b);
  const curves = [...new Set(S.budget_curve.map((b) => b.curve))];
  const budget = budgets.map((b) => {
    const row = { budget: b };
    S.budget_curve.filter((x) => x.budget === b).forEach((x) => { row[x.curve] = x.composite; });
    return row;
  });
  const imp = S.improvement_pct;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <Segmented size="md" value={task} onChange={setTask} options={taskOptions} />
        <Segmented size="md" value={pool} onChange={setPool}
          options={["indist", "gnome"].map((p) => ({ value: p, label: `Unlabeled: ${POOL[p].label}${avail.includes(p) ? "" : " (running)"}` }))} />
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <Stat value={fmt(S.best_overall.Composite_Score_mean)} label={`Best: ${S.best_overall.Method}`} sub={S.best_overall.Model_Config} />
        <Stat value={`${imp >= 0 ? "+" : ""}${imp.toFixed(2)}%`} label="best SSL vs tuned baseline" sub={`${S.best_ssl.Method} vs ${S.best_baseline.config}`} tone={imp >= 0 ? "good" : "amber"} />
        <Stat value={fmt(S.fully_supervised.Composite_Score)} label="fully-supervised upper bound" sub={`${big(S.setup.n_train_pool)} labels instead of 1,000`} tone="cyan" />
        <Stat value={big(S.n_runs)} label="models trained" sub={`${S.setup.seeds} seeds · ${Math.round(S.runtime_min)} min`} tone="ink" />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="card p-5">
          <SectionHeader title="Best configuration of each method" subtitle={`Composite = (F1 + ROC-AUC)/2, mean ± std over 3 seeds, 1,000 labels, ${big(S.setup.n_test_evaluated)} test compositions. Dashed line = best tuned supervised baseline.`} />
          <ResponsiveContainer width="100%" height={330}>
            <BarChart data={bars} layout="vertical" margin={{ left: 20, right: 50 }}>
              <CartesianGrid {...GRID} horizontal={false} />
              <XAxis type="number" domain={[lo, "auto"]} {...AXIS} tickFormatter={(v) => v.toFixed(2)} />
              <YAxis type="category" dataKey="method" width={140} {...AXIS} />
              <ReferenceLine x={base} stroke="#1B1F3B" strokeDasharray="5 4" label={{ value: "baseline", position: "top", fill: "#1B1F3B", fontSize: 11 }} />
              <Tooltip content={<ChartTooltip digits={4} />} cursor={{ fill: "#F1F5F9" }} />
              <Bar isAnimationActive={false} dataKey="composite" name="composite" radius={4} barSize={20}>
                {bars.map((b) => <Cell key={b.method} fill={METHOD_COLOR[b.method]} />)}
                <ErrorBar dataKey="err" width={4} stroke="#1B1F3B" direction="x" />
                <LabelList dataKey="composite" position="insideRight" formatter={(v) => v.toFixed(3)} fill="#fff" fontSize={11} fontWeight={600} offset={8} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="card p-5">
          <SectionHeader title="Does more unlabeled data help?" subtitle="Composite score of each method's best configuration vs the number of unlabeled compositions (log scale)." />
          <ResponsiveContainer width="100%" height={330}>
            <LineChart data={curve} margin={{ right: 20, bottom: 10 }}>
              <CartesianGrid {...GRID} />
              <XAxis dataKey="n" scale="log" domain={["dataMin", "dataMax"]} type="number" ticks={ns} tickFormatter={(v) => (v >= 1000 ? `${+(v / 1000).toFixed(1)}k` : v)} {...AXIS} />
              <YAxis domain={[(lo) => Math.floor((lo - 0.01) * 50) / 50, (hi) => Math.ceil((hi + 0.01) * 50) / 50]} tickFormatter={(v) => v.toFixed(2)} {...AXIS} />
              <ReferenceLine y={base} stroke="#1B1F3B" strokeDasharray="5 4" label={{ value: "baseline", position: "insideTopRight", fill: "#1B1F3B", fontSize: 11 }} />
              <Tooltip content={<ChartTooltip digits={4} labelFormatter={(v) => `${big(v)} unlabeled`} />} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {Object.keys(S.pool_curves).map((m) => (
                <Line isAnimationActive={false} key={m} dataKey={m} name={m} stroke={METHOD_COLOR[m]} strokeWidth={2} strokeDasharray={DASH[m]} dot={{ r: 4 }} activeDot={{ r: 6 }} connectNulls />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="card p-5">
          <SectionHeader title="How much do labels matter?" subtitle="Label-budget curve: best supervised baseline vs best SSL configuration; dotted = fully supervised with all training labels." />
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={budget} margin={{ right: 20 }}>
              <CartesianGrid {...GRID} />
              <XAxis dataKey="budget" scale="log" type="number" domain={["dataMin", "dataMax"]} ticks={budgets} tickFormatter={(v) => big(v)} {...AXIS} />
              <YAxis domain={[(lo) => Math.floor((lo - 0.01) * 50) / 50, (hi) => Math.ceil((hi + 0.01) * 50) / 50]} tickFormatter={(v) => v.toFixed(2)} {...AXIS} />
              <ReferenceLine y={S.fully_supervised.Composite_Score} stroke="#059669" strokeDasharray="2 3" label={{ value: "fully supervised", position: "insideBottomRight", fill: "#059669", fontSize: 11 }} />
              <Tooltip content={<ChartTooltip digits={4} labelFormatter={(v) => `${big(v)} labels`} />} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {curves.map((c) => (
                <Line isAnimationActive={false} key={c} dataKey={c} name={c} stroke={c.startsWith("Best SSL") ? "#4f46e5" : "#1B1F3B"} strokeDasharray={c.startsWith("Best SSL") ? "" : "5 4"} strokeWidth={2} dot={{ r: 4 }} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="card p-5 overflow-x-auto">
          <SectionHeader title="Supervised model zoo" subtitle="Same 1,000 labels; Optuna (TPE, 25 trials, 5-fold CV) tuned the two best families." />
          <table className="w-full text-sm">
            <thead><tr className="text-left text-xs text-ink-mute border-b border-line"><th className="py-2">Model</th><th>F1</th><th>ROC-AUC</th><th>Composite</th></tr></thead>
            <tbody>
              {S.model_zoo.map((z) => (
                <tr key={z.model} className={`border-b border-line/60 ${z.model === S.best_baseline.config ? "bg-brand-50 font-semibold" : ""}`}>
                  <td className="py-1.5">{z.model}</td><td className="font-mono">{fmt(z.f1)}</td><td className="font-mono">{fmt(z.auc)}</td>
                  <td className="font-mono">{fmt(z.composite)} <span className="text-ink-mute text-xs">± {fmt(z.composite_std)}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-3 text-xs text-ink-soft">
            {Object.entries(S.tuned_params).map(([k, p]) => (
              <div key={k} className="mt-1"><b>{k}</b> (CV {fmt(S.optuna_cv[k])}): <span className="font-mono">{Object.entries(p).map(([a, b]) => `${a}=${typeof b === "number" ? +b.toFixed(3) : b}`).join(", ")}</span></div>
            ))}
          </div>
        </div>
      </div>

      <div className="card p-5 overflow-x-auto">
        <SectionHeader title="Full comparison table" subtitle="Best configuration of every method (mean over 3 seeds)." />
        <table className="w-full text-sm">
          <thead><tr className="text-left text-xs text-ink-mute border-b border-line">
            <th className="py-2">Method</th><th>Configuration</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>Composite</th><th>Time (s)</th></tr></thead>
          <tbody>
            {S.best_per_method.map((r) => (
              <tr key={r.Method} className="border-b border-line/60">
                <td className="py-1.5 font-semibold"><span className="inline-block h-2.5 w-2.5 rounded-sm mr-2" style={{ background: METHOD_COLOR[r.Method] }} />{r.Method}</td>
                <td className="font-mono text-xs">{r.Model_Config}</td>
                {["Accuracy_mean", "Precision_mean", "Recall_mean", "F1-Score_mean", "ROC-AUC_mean"].map((k) => <td key={k} className="font-mono">{fmt(r[k])}</td>)}
                <td className="font-mono font-semibold">{fmt(r.Composite_Score_mean)}</td>
                <td className="font-mono text-ink-mute">{fmt(r.Training_Time_mean, 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
