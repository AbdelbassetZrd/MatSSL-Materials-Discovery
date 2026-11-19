import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, LabelList } from "recharts";
import { ChartTooltip, SectionHeader, Stat } from "../ui.jsx";
import { AXIS, GRID, fmt, big } from "../../lib/theme.js";

export default function BandGap({ data }) {
  const B = data.band_gap;
  if (!B) return <div className="card p-8 text-center text-ink-mute">Band-gap results not available.</div>;
  const S = B.summary, ga = S.ga, ad = S.applicability_domain;
  const cats = Object.entries(ga.descriptor_categories).filter(([, v]) => v).map(([k, v]) => ({ name: k, n: v })).sort((a, b) => b.n - a.n);
  return (
    <div className="space-y-6">
      <SectionHeader title="Traditional pipeline: photovoltaic band gap"
        subtitle={`Target: DFT-PBE band gap in 1.1–1.7 eV (${(100 * S.dataset.positive_rate).toFixed(1)}% of ${big(S.dataset.n)} compositions). PBE underestimates experimental gaps, so GaAs or CdTe fall below this window.`} />
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <Stat value={`142 → ${ga.final_n_features}`} label="descriptors kept by the GA" sub="−63 % dimensionality" />
        <Stat value={fmt(S.test_ga_descriptors["ROC-AUC"])} label="test ROC-AUC (GA descriptors)" sub={`all 142 descriptors: ${fmt(S.test_all_descriptors["ROC-AUC"])}`} tone="cyan" />
        <Stat value={fmt(S.test_ga_descriptors_tuned_threshold["F1-Score"])} label="test F1, calibrated threshold" sub={`was ${fmt(S.test_ga_descriptors["F1-Score"])} at 0.5 · t = ${B.threshold.toFixed(2)}`} tone="amber" />
        <Stat value={`${(100 * ad.fraction_inside).toFixed(1)}%`} label="test compositions inside the AD" sub={`${big(ad.n_inside)} / ${big(ad.n_test)}`} tone="good" />
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="card p-5">
          <SectionHeader title="(A) Fitness evolution" subtitle="Population 100 · 20 generations · crossover 0.8 · mutation 0.2 · elite 5 · fitness = ROC-AUC − 0.05·(features / total)." />
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={B.ga_history} margin={{ right: 20 }}>
              <CartesianGrid {...GRID} />
              <XAxis dataKey="generation" {...AXIS} />
              <YAxis domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(3)} {...AXIS} />
              <Tooltip content={<ChartTooltip digits={4} labelFormatter={(v) => `generation ${v}`} />} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line isAnimationActive={false} dataKey="best_auc" name="CV ROC-AUC of best subset" stroke="#0891b2" strokeWidth={2} dot={{ r: 3 }} />
              <Line isAnimationActive={false} dataKey="max_fitness" name="best fitness" stroke="#4f46e5" strokeWidth={2} dot={{ r: 3 }} />
              <Line isAnimationActive={false} dataKey="avg_fitness" name="average fitness" stroke="#eb6834" strokeWidth={2} strokeDasharray="5 4" dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="card p-5">
          <SectionHeader title="(B) Feature count evolution" subtitle="Descriptors used by the best individual and on average in the population." />
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={B.ga_history} margin={{ right: 20 }}>
              <CartesianGrid {...GRID} />
              <XAxis dataKey="generation" {...AXIS} />
              <YAxis domain={["auto", "auto"]} {...AXIS} />
              <Tooltip content={<ChartTooltip digits={1} labelFormatter={(v) => `generation ${v}`} />} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line isAnimationActive={false} dataKey="best_n_features" name="best individual" stroke="#4f46e5" strokeWidth={2} dot={{ r: 3 }} />
              <Line isAnimationActive={false} dataKey="avg_n_features" name="population average" stroke="#eb6834" strokeWidth={2} strokeDasharray="5 4" dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="card p-5 overflow-x-auto">
          <SectionHeader title="Model comparison (all 142 descriptors)" subtitle="Held-out test set; the best family is used for the GA model." />
          <table className="w-full text-sm">
            <thead><tr className="text-left text-xs text-ink-mute border-b border-line"><th className="py-2">Model</th><th>ROC-AUC</th><th>PR-AUC</th><th>F1</th><th>Time</th></tr></thead>
            <tbody>
              {S.model_comparison.map((m, i) => (
                <tr key={m.Model} className={`border-b border-line/60 ${i === 0 ? "bg-brand-50 font-semibold" : ""}`}>
                  <td className="py-1.5">{m.Model}</td><td className="font-mono">{fmt(m["ROC-AUC"])}</td><td className="font-mono">{fmt(m["PR-AUC"])}</td>
                  <td className="font-mono">{fmt(m["F1-Score"])}</td><td className="font-mono text-ink-mute">{fmt(m.Training_Time, 1)} s</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card p-5">
          <SectionHeader title="Selected descriptor categories" subtitle={`${ga.final_n_features} descriptors chosen by the genetic algorithm.`} />
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={cats} layout="vertical" margin={{ left: 20, right: 40 }}>
              <CartesianGrid {...GRID} horizontal={false} />
              <XAxis type="number" allowDecimals={false} {...AXIS} />
              <YAxis type="category" dataKey="name" width={170} {...AXIS} />
              <Tooltip content={<ChartTooltip digits={0} />} cursor={{ fill: "#F1F5F9" }} />
              <Bar isAnimationActive={false} dataKey="n" name="descriptors" fill="#4f46e5" radius={4} barSize={18}>
                <LabelList dataKey="n" position="right" fill="#1B1F3B" fontSize={12} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div className="card p-5">
        <SectionHeader title="Applicability domain" subtitle={`Mean distance to the 5 nearest training compositions vs the 95th percentile of training distances (threshold ${ad.distance_threshold_95pct.toFixed(2)}).`} />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div className="rounded-xl bg-surface p-4"><div className="label">PCA</div><div className="mt-1">2 components explain <b>{(100 * ad.pca_variance_2pc).toFixed(1)}%</b> of the variance; train and test overlap.</div></div>
          <div className="rounded-xl bg-surface p-4"><div className="label">K-Means (k = 6)</div><div className="mt-1">Train and test cluster shares agree within <b>0.5%</b>.</div></div>
          <div className="rounded-xl bg-surface p-4"><div className="label">Reliability</div><div className="mt-1">Recall of PV materials: <b>{fmt(ad.metrics_inside_tuned.Recall)}</b> inside vs <b>{fmt(ad.metrics_outside_tuned.Recall)}</b> outside the domain.</div></div>
        </div>
      </div>
    </div>
  );
}
