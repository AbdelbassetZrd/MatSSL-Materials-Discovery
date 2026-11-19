import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, LabelList } from "recharts";
import { ChartTooltip, SectionHeader } from "../ui.jsx";
import { AXIS, GRID, big } from "../../lib/theme.js";

function ksColor(v) {
  // sequential single-hue ramp: low shift = light, high shift = dark
  const t = Math.min(1, v / 0.5);
  const a = [238, 242, 255], b = [49, 46, 129];
  return `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * t)).join(",")})`;
}

export default function Datasets({ data }) {
  const ds = data.datasets;
  const bars = ds.map((d) => ({ name: d.title, positive: +(100 * d.positive_rate).toFixed(1) }));
  const feats = ds[0].ks_vs_gnome.map((k) => k.feature);
  return (
    <div className="space-y-6">
      <SectionHeader title="Four labeled datasets + one unlabeled pool"
        subtitle="One row per unique composition (duplicated polymorphs merged); stratified 80/20 split; the test set is never used for training or tuning." />
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-4">
        {ds.map((d) => (
          <div key={d.task} className="card p-5">
            <div className="label">{d.source}</div>
            <div className="font-display text-xl font-bold">{d.title}</div>
            <div className="mt-2 font-display text-3xl font-bold text-brand">{big(d.n)}</div>
            <div className="text-xs text-ink-soft">unique compositions · {(100 * d.positive_rate).toFixed(1)}% positive</div>
          </div>
        ))}
        <div className="card p-5">
          <div className="label">Google DeepMind</div>
          <div className="font-display text-xl font-bold">GNoME (unlabeled)</div>
          <div className="mt-2 font-display text-3xl font-bold text-amber-600">{big(data.kpis.unlabeled)}</div>
          <div className="text-xs text-ink-soft">new stable compositions, labels never used</div>
        </div>
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="card p-5">
          <h3 className="font-semibold mb-1">Class balance</h3>
          <p className="text-xs text-ink-soft mb-2">Share of positive compositions (metal / magnetic / stable / PV window).</p>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={bars} layout="vertical" margin={{ left: 10, right: 40 }}>
              <CartesianGrid {...GRID} horizontal={false} />
              <XAxis type="number" domain={[0, 60]} unit="%" {...AXIS} />
              <YAxis type="category" dataKey="name" width={110} {...AXIS} />
              <Tooltip content={<ChartTooltip digits={1} />} cursor={{ fill: "#F1F5F9" }} />
              <Bar isAnimationActive={false} dataKey="positive" name="positive %" fill="#4f46e5" radius={4} barSize={22}>
                <LabelList dataKey="positive" position="right" formatter={(v) => `${v}%`} fill="#1B1F3B" fontSize={12} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="card p-5 overflow-x-auto">
          <h3 className="font-semibold mb-1">Distribution shift: labeled data vs GNoME</h3>
          <p className="text-xs text-ink-soft mb-3">Kolmogorov–Smirnov statistic per descriptor (0 = identical distributions). Darker = larger shift.</p>
          <table className="text-xs">
            <thead><tr><th className="text-left pr-2"></th>{feats.map((f) => <th key={f} className="px-1 pb-2 font-medium text-ink-soft align-bottom" style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}>{f}</th>)}</tr></thead>
            <tbody>
              {ds.map((d) => (
                <tr key={d.task}>
                  <td className="pr-3 py-1 font-semibold whitespace-nowrap">{d.title}</td>
                  {d.ks_vs_gnome.map((k) => (
                    <td key={k.feature} className="p-0.5">
                      <div className="grid h-9 w-12 place-items-center rounded font-mono" style={{ background: ksColor(k.ks), color: k.ks > 0.27 ? "#fff" : "#1B1F3B" }}>{k.ks.toFixed(2)}</div>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
