import { CheckCircle2, AlertTriangle, Magnet, Scale, Sun, Zap } from "lucide-react";
import {
  Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
  ZAxis, CartesianGrid, Legend,
} from "recharts";
import Gauge from "./Gauge.jsx";
import { ChartTooltip } from "./ui.jsx";
import { AXIS, GRID, NEG, POS, TASK, fmt } from "../lib/theme.js";

const ICONS = { metallicity: Zap, magnetism: Magnet, stability: Scale, band_gap: Sun };

export function PropertyCard({ p }) {
  const Icon = ICONS[p.task] || Zap;
  const t = TASK[p.task];
  return (
    <div className={`card p-5 flex flex-col ${p.prediction ? "ring-1 ring-brand/30" : ""}`}>
      <div className="flex items-center gap-2">
        <span className={`grid h-9 w-9 place-items-center rounded-full ${p.prediction ? "bg-brand text-white" : "bg-surface text-ink-soft"}`}>
          <Icon size={18} />
        </span>
        <div>
          <div className="font-semibold text-ink leading-tight">{t?.title || p.title}</div>
          <div className="text-[11px] text-ink-mute">{t?.source}</div>
        </div>
      </div>
      <div className="mt-2 flex justify-center">
        <Gauge value={p.probability} threshold={p.threshold} positive={t?.positive || "positive"} />
      </div>
      <div className={`text-center text-lg font-bold ${p.prediction ? "text-brand" : "text-ink-soft"}`}>{p.prediction_label}</div>
      <div className="mt-3 space-y-1 text-xs text-ink-soft border-t border-line pt-3">
        <div className="flex justify-between"><span>Method</span><b className="text-ink text-right">{p.method || "–"}</b></div>
        <div className="flex justify-between gap-2"><span>Configuration</span><span className="text-right font-mono text-[11px]">{p.config || "–"}</span></div>
        <div className="flex justify-between"><span>Test ROC-AUC · F1</span>
          <b className="text-ink">{fmt(p.test_metrics?.["ROC-AUC"])} · {fmt(p.test_metrics?.["F1-Score"])}</b></div>
      </div>
    </div>
  );
}

export function ADBadge({ ad }) {
  if (!ad) return null;
  return ad.inside ? (
    <div className="flex items-start gap-2 rounded-xl bg-emerald-50 px-4 py-3 text-sm text-emerald-800">
      <CheckCircle2 size={18} className="mt-0.5 shrink-0" />
      <span><b>Inside the applicability domain.</b> This composition resembles the training materials (distance {ad.distance.toFixed(2)} ≤ {ad.threshold.toFixed(2)}), so the predictions are reliable.</span>
    </div>
  ) : (
    <div className="flex items-start gap-2 rounded-xl bg-amber-50 px-4 py-3 text-sm text-amber-900">
      <AlertTriangle size={18} className="mt-0.5 shrink-0" />
      <span><b>Outside the applicability domain.</b> This composition is far from the training materials (distance {ad.distance.toFixed(2)} &gt; {ad.threshold.toFixed(2)}): treat the predictions with caution.</span>
    </div>
  );
}

// SHAP contributions (log-odds) as diverging bars, or descriptor percentiles per class.
export function Explanation({ p }) {
  const e = p.explanation;
  const t = TASK[p.task];
  if (!e) return <p className="text-sm text-ink-mute">No explanation available for this model.</p>;
  if (e.type === "shap") {
    const data = e.items.map((d) => ({ name: d.feature, value: +d.contribution.toFixed(3), raw: d.value }));
    return (
      <div>
        <p className="text-xs text-ink-soft mb-2">SHAP contributions (log-odds): <span style={{ color: POS }} className="font-semibold">→ {t?.positive}</span> · <span style={{ color: NEG }} className="font-semibold">← {t?.negative}</span></p>
        <ResponsiveContainer width="100%" height={36 * data.length + 30}>
          <BarChart data={data} layout="vertical" margin={{ left: 10, right: 30 }}>
            <CartesianGrid {...GRID} horizontal={false} />
            <XAxis type="number" {...AXIS} />
            <YAxis type="category" dataKey="name" width={190} {...AXIS} />
            <ReferenceLine x={0} stroke="#1B1F3B" />
            <Tooltip content={<ChartTooltip />} cursor={{ fill: "#F1F5F9" }} />
            <Bar isAnimationActive={false} dataKey="value" name="contribution" radius={4} barSize={18}>
              {data.map((d) => <Cell key={d.name} fill={d.value >= 0 ? POS : NEG} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  }
  return (
    <div className="space-y-2">
      <p className="text-xs text-ink-soft">Where this composition sits among {t?.positive} vs {t?.negative} training materials (percentile):</p>
      {e.items.map((d) => (
        <div key={d.feature} className="text-xs">
          <div className="flex justify-between text-ink"><span className="font-semibold">{d.feature}</span><span className="font-mono">{fmt(d.value, 2)}</span></div>
          <div className="mt-1 grid grid-cols-[90px_1fr_36px] items-center gap-2">
            <span style={{ color: POS }}>{t?.positive}</span>
            <div className="h-2 rounded bg-surface"><div className="h-2 rounded" style={{ width: `${d.pct_positive}%`, background: POS }} /></div>
            <span className="text-right font-mono">{d.pct_positive.toFixed(0)}</span>
            <span style={{ color: NEG }}>{t?.negative}</span>
            <div className="h-2 rounded bg-surface"><div className="h-2 rounded" style={{ width: `${d.pct_negative}%`, background: NEG }} /></div>
            <span className="text-right font-mono">{d.pct_negative.toFixed(0)}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

const yes = (v, a, b) => (v == null ? <span className="text-ink-mute">–</span> : v ? <b className="text-brand">{a}</b> : <span>{b}</span>);

export function NeighborsTable({ data }) {
  if (!data) return null;
  return (
    <div className="overflow-x-auto">
      {data.known && <p className="mb-2 text-xs text-emerald-700 font-semibold">✓ This exact composition exists in our labeled databases – its true labels are in the first row.</p>}
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs text-ink-mute border-b border-line">
            <th className="py-2 pr-3">Composition</th><th className="pr-3">Similarity distance</th><th className="pr-3">Metal</th>
            <th className="pr-3">Magnetic</th><th className="pr-3">PBE gap (eV)</th><th>Stable</th>
          </tr>
        </thead>
        <tbody>
          {data.neighbors.map((n, i) => (
            <tr key={n.formula} className={`border-b border-line/60 ${i === 0 && data.known ? "bg-emerald-50/60" : ""}`}>
              <td className="py-2 pr-3 font-mono font-semibold">{n.formula}</td>
              <td className="pr-3 font-mono text-ink-soft">{n.distance.toFixed(3)}</td>
              <td className="pr-3">{yes(n.is_metal, "metal", "non-metal")}</td>
              <td className="pr-3">{yes(n.is_magnetic, "magnetic", "no")}</td>
              <td className="pr-3 font-mono">{n.gap_pbe == null ? "–" : n.gap_pbe.toFixed(2)}</td>
              <td>{yes(n.is_stable, "stable", "unstable")}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-[11px] text-ink-mute">Distance in the 32-d self-supervised embedding learned on 554k unlabeled GNoME compositions.</p>
    </div>
  );
}

function StarShape({ cx, cy }) {
  const R = 13, r = 5.5;
  const pts = Array.from({ length: 10 }, (_, i) => {
    const a = (Math.PI / 5) * i - Math.PI / 2;
    const rad = i % 2 ? r : R;
    return `${cx + rad * Math.cos(a)},${cy + rad * Math.sin(a)}`;
  }).join(" ");
  return <polygon points={pts} fill="#F59E0B" stroke="#1B1F3B" strokeWidth={1.5} />;
}

// Training chemical space (PCA of the GA-selected descriptors) with the query as a star.
export function ChemicalSpaceMap({ points, query, height = 340 }) {
  if (!points) return null;
  const pos = points.filter((p) => p.pv === 1);
  const neg = points.filter((p) => p.pv === 0);
  const q = query ? [{ x: query[0], y: query[1], formula: "your material" }] : [];
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: 0 }}>
        <CartesianGrid {...GRID} />
        <XAxis type="number" dataKey="x" name="PC1" {...AXIS} label={{ value: "PC1", position: "insideBottom", offset: -10, fill: "#64748B", fontSize: 12 }} />
        <YAxis type="number" dataKey="y" name="PC2" {...AXIS} label={{ value: "PC2", angle: -90, position: "insideLeft", fill: "#64748B", fontSize: 12 }} />
        <ZAxis range={[14, 14]} />
        <Tooltip cursor={false} content={({ active, payload }) => active && payload?.length ? (
          <div className="rounded-lg border border-line bg-white px-2 py-1 text-xs shadow-card font-mono">{payload[0].payload.formula}</div>) : null} />
        <Legend wrapperStyle={{ fontSize: 12 }} verticalAlign="top" />
        <Scatter isAnimationActive={false} name="training: outside PV window" data={neg} fill="#CBD5E1" fillOpacity={0.6} />
        <Scatter isAnimationActive={false} name="training: in PV window" data={pos} fill={POS} fillOpacity={0.55} />
        {q.length > 0 && <Scatter isAnimationActive={false} name="your material" data={q} fill="#F59E0B" shape={StarShape} legendType="star" />}
      </ScatterChart>
    </ResponsiveContainer>
  );
}
