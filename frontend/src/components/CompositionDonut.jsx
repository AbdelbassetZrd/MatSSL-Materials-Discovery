import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { SERIES } from "../lib/theme.js";
import { ChartTooltip } from "./ui.jsx";

export default function CompositionDonut({ counts, height = 200 }) {
  const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
  const data = Object.entries(counts).map(([el, n]) => ({ name: el, value: +((100 * n) / total).toFixed(1) }));
  if (!data.length) return <div className="text-ink-mute text-sm py-8 text-center">No element selected</div>;
  return (
    <div className="grid grid-cols-[1fr_auto] items-center gap-3">
      <ResponsiveContainer width="100%" height={height}>
        <PieChart>
          <Pie data={data} dataKey="value" nameKey="name" innerRadius="58%" outerRadius="92%" paddingAngle={2}
            stroke="#fff" strokeWidth={2} isAnimationActive={false}>
            {data.map((d, i) => <Cell key={d.name} fill={SERIES[i % SERIES.length]} />)}
          </Pie>
          <Tooltip content={<ChartTooltip digits={1} />} />
        </PieChart>
      </ResponsiveContainer>
      <ul className="space-y-1.5 text-sm pr-2">
        {data.map((d, i) => (
          <li key={d.name} className="flex items-center gap-2">
            <i className="h-3 w-3 rounded-sm" style={{ background: SERIES[i % SERIES.length] }} />
            <b className="w-7">{d.name}</b>
            <span className="font-mono text-ink-soft">{d.value}%</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
