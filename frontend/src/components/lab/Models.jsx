import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartTooltip, SectionHeader } from "../ui.jsx";
import { AXIS, GRID, SERIES, TASK, fmt, big } from "../../lib/theme.js";

function Confusion({ cm, pos, neg }) {
  const [[tn, fp], [fn, tp]] = cm;
  const total = tn + fp + fn + tp;
  const cell = (v, good) => (
    <div className={`grid place-items-center rounded-lg p-3 ${good ? "bg-brand text-white" : "bg-surface text-ink"}`}>
      <div className="font-display text-xl font-bold">{big(v)}</div>
      <div className={`text-[10px] ${good ? "text-white/80" : "text-ink-mute"}`}>{((100 * v) / total).toFixed(1)}%</div>
    </div>
  );
  return (
    <div className="grid grid-cols-[70px_1fr_1fr] gap-1.5 text-xs items-center">
      <span />
      <span className="text-center text-ink-mute">pred. {neg}</span>
      <span className="text-center text-ink-mute">pred. {pos}</span>
      <span className="text-ink-mute">true {neg}</span>{cell(tn, true)}{cell(fp, false)}
      <span className="text-ink-mute">true {pos}</span>{cell(fn, false)}{cell(tp, true)}
    </div>
  );
}

export default function Models({ data }) {
  const dep = { ...data.deployed };
  if (data.band_gap) {
    dep.band_gap = {
      method: "GA-selected descriptors + Random Forest",
      config: `${data.band_gap.summary.ga.final_n_features} descriptors, threshold ${data.band_gap.threshold.toFixed(2)}`,
      test_metrics: data.band_gap.summary.test_ga_descriptors_tuned_threshold,
      roc: data.band_gap.roc, confusion: data.band_gap.confusion,
      n_test: data.band_gap.summary.dataset.n_test, selection_pool: "–",
    };
  }
  const tasks = Object.keys(TASK).filter((t) => dep[t]);
  const grid = Array.from({ length: 61 }, (_, i) => ({ fpr: i / 60 }));
  const rocData = grid.map((g) => {
    const row = { fpr: g.fpr, chance: g.fpr };
    tasks.forEach((t) => {
      const pts = dep[t].roc;
      const j = pts.findIndex((p) => p.fpr >= g.fpr);
      row[t] = j <= 0 ? pts[0]?.tpr ?? 0 : pts[j - 1].tpr + ((pts[j].tpr - pts[j - 1].tpr) * (g.fpr - pts[j - 1].fpr)) / Math.max(1e-9, pts[j].fpr - pts[j - 1].fpr);
    });
    return row;
  });
  return (
    <div className="space-y-6">
      <SectionHeader title="Selected models and selection criterion"
        subtitle="For each task the architecture with the highest mean composite score (F1 + ROC-AUC)/2 over 3 seeds at 1,000 labels is selected, then re-trained on all training labels (+ GNoME unlabeled pool) for deployment. Ties favour the simpler model." />
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        {tasks.map((t) => {
          const d = dep[t];
          return (
            <div key={t} className="card p-5 space-y-3">
              <div>
                <div className="label">{TASK[t].source}</div>
                <div className="font-display text-xl font-bold">{TASK[t].title}</div>
                <div className="text-sm text-brand font-semibold">{d.method}</div>
                <div className="font-mono text-[11px] text-ink-mute">{d.config}</div>
              </div>
              <div className="grid grid-cols-3 gap-2 text-center">
                {[["ROC-AUC", "ROC-AUC"], ["F1", "F1-Score"], ["Acc.", "Accuracy"]].map(([l, k]) => (
                  <div key={k} className="rounded-lg bg-surface py-2"><div className="label">{l}</div><div className="font-display text-lg font-bold">{fmt(d.test_metrics?.[k])}</div></div>
                ))}
              </div>
              <Confusion cm={d.confusion} pos={TASK[t].positive} neg={TASK[t].negative} />
              <div className="text-[11px] text-ink-mute">{big(d.n_test)} held-out test compositions{d.selection_pool && d.selection_pool !== "–" ? ` · selected with the ${d.selection_pool === "indist" ? "same-database" : "GNoME"} pool` : ""}</div>
            </div>
          );
        })}
      </div>
      <div className="card p-5">
        <SectionHeader title="ROC curves of the deployed models" subtitle="Held-out test sets; the diagonal is a random classifier." />
        <ResponsiveContainer width="100%" height={380}>
          <LineChart data={rocData} margin={{ right: 20, bottom: 20, left: 10 }}>
            <CartesianGrid {...GRID} />
            <XAxis dataKey="fpr" type="number" domain={[0, 1]} ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]} tickFormatter={(v) => v.toFixed(1)} {...AXIS}
              label={{ value: "False positive rate", position: "insideBottom", offset: -5, fill: "#64748B", fontSize: 12 }} />
            <YAxis domain={[0, 1]} ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]} tickFormatter={(v) => v.toFixed(1)} {...AXIS}
              label={{ value: "True positive rate", angle: -90, position: "insideLeft", fill: "#64748B", fontSize: 12 }} />
            <Tooltip content={<ChartTooltip digits={3} labelFormatter={(v) => `FPR ${v.toFixed(2)}`} />} />
            <Legend verticalAlign="top" wrapperStyle={{ fontSize: 12, paddingBottom: 8 }} />
            <Line isAnimationActive={false} dataKey="chance" name="random" stroke="#94A3B8" strokeDasharray="4 4" dot={false} />
            {tasks.map((t, i) => (
              <Line isAnimationActive={false} key={t} dataKey={t} name={`${TASK[t].title} (AUC ${fmt(dep[t].test_metrics?.["ROC-AUC"])})`} stroke={SERIES[i]} strokeWidth={2.5} dot={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
