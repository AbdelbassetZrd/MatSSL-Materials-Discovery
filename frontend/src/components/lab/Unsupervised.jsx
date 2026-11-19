import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { ChartTooltip, SectionHeader, Segmented } from "../ui.jsx";
import { AXIS, GRID, SERIES, TASK, fmt } from "../../lib/theme.js";

const ITEM = { stability: "STABLE", magnetism: "MAGNETIC", metallicity: "METAL", band_gap: "PV_GAP" };

export default function Unsupervised({ data }) {
  const U = data.unsupervised;
  const [task, setTask] = useState("magnetism");
  const [sort, setSort] = useState("lift");
  const rules = useMemo(() => [...(U.rules[task] || [])].sort((a, b) => b[sort] - a[sort]).slice(0, 15), [U, task, sort]);
  const maxLift = Math.max(...rules.map((r) => r.lift), 1);
  const k = U.summary.clustering.best_k;
  const clusters = Array.from({ length: k }, (_, c) => U.cluster_points.filter((p) => p.cluster === c));
  const prof = U.cluster_profiles.map((p) => ({
    name: `cluster ${p.cluster}`, Stable: p.STABLE_rate, Magnetic: p.MAGNETIC_rate, Metal: p.METAL_rate, "PV gap": p.PV_GAP_rate, top: p.top_elements,
  }));
  return (
    <div className="space-y-6">
      <div className="card p-5">
        <SectionHeader title="Association rules (Apriori)"
          subtitle="Each composition is a transaction of elements; rules {elements} ⇒ property are scored by support, confidence and lift (lift > 1 = more likely than average)."
          right={<div className="flex flex-wrap gap-2">
            <Segmented size="md" value={task} onChange={setTask} options={Object.keys(ITEM).map((t) => ({ value: t, label: TASK[t].title }))} />
            <Segmented size="md" value={sort} onChange={setSort} options={[{ value: "lift", label: "by lift" }, { value: "confidence", label: "by confidence" }, { value: "support", label: "by support" }]} />
          </div>} />
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead><tr className="text-left text-xs text-ink-mute border-b border-line"><th className="py-2">Rule</th><th>Support</th><th>Confidence</th><th className="w-1/3">Lift</th></tr></thead>
            <tbody>
              {rules.map((r) => (
                <tr key={r.antecedents} className="border-b border-line/60">
                  <td className="py-1.5 font-mono">{"{"}{r.antecedents}{"}"} ⇒ <b className="text-brand">{ITEM[task]}</b></td>
                  <td className="font-mono">{fmt(r.support, 4)}</td>
                  <td className="font-mono">{(100 * r.confidence).toFixed(1)}%</td>
                  <td><div className="flex items-center gap-2"><div className="h-2 rounded bg-brand" style={{ width: `${(100 * r.lift) / maxLift}%` }} /><span className="font-mono text-xs">{r.lift.toFixed(2)}</span></div></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="card p-5">
          <SectionHeader title={`K-Means clusters (k = ${k}, silhouette)`} subtitle="PCA projection of labeled and unlabeled compositions – no labels used." />
          <ResponsiveContainer width="100%" height={340}>
            <ScatterChart margin={{ right: 10, bottom: 10 }}>
              <CartesianGrid {...GRID} />
              <XAxis type="number" dataKey="x" name="PC1" {...AXIS} />
              <YAxis type="number" dataKey="y" name="PC2" {...AXIS} />
              <ZAxis range={[12, 12]} />
              <Tooltip cursor={false} content={({ active, payload }) => active && payload?.length ? <div className="rounded-lg border border-line bg-white px-2 py-1 text-xs shadow-card">cluster {payload[0].payload.cluster} · {payload[0].payload.source}</div> : null} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {clusters.map((pts, c) => <Scatter isAnimationActive={false} key={c} name={`cluster ${c}`} data={pts} fill={SERIES[c]} fillOpacity={0.55} />)}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div className="card p-5">
          <SectionHeader title="What the clusters contain" subtitle="Property rates inside each cluster (labels used only to describe the clusters, not to build them)." />
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={prof} margin={{ right: 10 }}>
              <CartesianGrid {...GRID} vertical={false} />
              <XAxis dataKey="name" {...AXIS} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(100 * v)}%`} {...AXIS} />
              <Tooltip content={<ChartTooltip digits={3} />} cursor={{ fill: "#F1F5F9" }} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {["Stable", "Magnetic", "Metal", "PV gap"].map((s, i) => <Bar isAnimationActive={false} key={s} dataKey={s} fill={SERIES[i]} radius={[4, 4, 0, 0]} />)}
            </BarChart>
          </ResponsiveContainer>
          <div className="mt-2 space-y-0.5 text-xs text-ink-soft">{prof.map((p) => <div key={p.name}><b>{p.name}</b>: {p.top}</div>)}</div>
        </div>
      </div>
      <div className="card p-5">
        <SectionHeader title="Element co-occurrence in GNoME (no property)" subtitle="Which elements appear together in the new compositions proposed by the generative search." />
        <div className="flex flex-wrap gap-2">
          {U.gnome_rules.slice(0, 14).map((r) => (
            <span key={r.antecedents + r.consequents} className="chip !cursor-default font-mono text-xs">{r.antecedents} ⇒ {r.consequents} <b className="text-brand">lift {r.lift.toFixed(1)}</b></span>
          ))}
        </div>
      </div>
    </div>
  );
}
