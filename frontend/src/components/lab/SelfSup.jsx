import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartTooltip, SectionHeader, Stat } from "../ui.jsx";
import { AXIS, GRID, POOL, TASK, fmt, big } from "../../lib/theme.js";

export default function SelfSup({ data }) {
  const A = data.self_supervised;
  const last = A.history[A.history.length - 1];
  const rows = [];
  Object.entries(data.ssl).forEach(([task, pools]) => Object.entries(pools).forEach(([pool, S]) => {
    if (!S) return;
    const ae = S.best_per_method.find((r) => r.Method === "Self-Supervised (AE)");
    rows.push({ task, pool, base: S.best_baseline.composite_score, ae: ae?.Composite_Score_mean, cfg: ae?.Model_Config });
  }));
  return (
    <div className="space-y-6">
      <SectionHeader title="Self-supervised pre-training: masked autoencoder"
        subtitle="Pretext task without labels: hide 25% of the 142 descriptors of each GNoME composition and reconstruct them (142 → 128 → 32 → 128 → 142). The 32-d bottleneck is a learned materials embedding." />
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <Stat value={big(A.summary.n_unlabeled_samples)} label="unlabeled compositions" sub="GNoME, no labels used" />
        <Stat value={fmt(last.val_r2_all)} label="validation R² (all descriptors)" tone="cyan" />
        <Stat value={fmt(last.val_r2_masked_features)} label="R² on the hidden descriptors" tone="amber" />
        <Stat value={A.summary.embedding_dim} label="embedding dimensions" sub="used by SSL, Label Spreading & similarity search" tone="good" />
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="card p-5">
          <SectionHeader title="Training curve" subtitle="Reconstruction quality on 27,702 held-out compositions." />
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={A.history} margin={{ right: 20 }}>
              <CartesianGrid {...GRID} />
              <XAxis dataKey="epoch" {...AXIS} />
              <YAxis domain={[0.78, 0.93]} tickFormatter={(v) => v.toFixed(2)} {...AXIS} />
              <Tooltip content={<ChartTooltip digits={4} labelFormatter={(v) => `epoch ${v}`} />} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line isAnimationActive={false} dataKey="val_r2_all" name="R² all descriptors" stroke="#4f46e5" strokeWidth={2} dot={{ r: 3 }} />
              <Line isAnimationActive={false} dataKey="val_r2_masked_features" name="R² masked descriptors" stroke="#eb6834" strokeWidth={2} strokeDasharray="5 4" dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="card p-5 overflow-x-auto">
          <SectionHeader title="Downstream effect (1,000 labels)" subtitle="Tuned learner on raw descriptors vs with the self-supervised embedding." />
          <table className="w-full text-sm">
            <thead><tr className="text-left text-xs text-ink-mute border-b border-line"><th className="py-2">Task</th><th>Pool</th><th>Baseline</th><th>+ embedding</th><th>Δ</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.task + r.pool} className="border-b border-line/60">
                  <td className="py-1.5 font-semibold">{TASK[r.task].title}</td>
                  <td className="text-xs text-ink-soft">{POOL[r.pool].label}</td>
                  <td className="font-mono">{fmt(r.base)}</td>
                  <td className="font-mono">{fmt(r.ae)} <span className="block text-[10px] text-ink-mute">{r.cfg}</span></td>
                  <td className={`font-mono font-semibold ${r.ae >= r.base ? "text-good" : "text-amber-700"}`}>{r.ae == null ? "–" : `${r.ae >= r.base ? "+" : ""}${(r.ae - r.base).toFixed(3)}`}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 text-xs text-ink-soft">The embedding is most useful as a compact similarity space: it powers the “nearest known materials” search on the Results page.</p>
        </div>
      </div>
    </div>
  );
}
