import { Bar, BarChart, CartesianGrid, XAxis, YAxis, Legend, PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart, ResponsiveContainer, Tooltip } from "recharts";
import { ChartTooltip, } from "./ui.jsx";
import { SERIES } from "../lib/theme.js";

const SHORT = { metallicity: "Metal", magnetism: "Magnetic", stability: "Stable", band_gap: "PV gap" };

// One polygon per material: the predicted PROBABILITY of each property (not the 0/1 class).
export default function ProfileRadar({ series, height = 300 }) {
  const tasks = series[0]?.predictions.map((p) => p.task) || [];
  const data = tasks.map((t) => {
    const row = { axis: SHORT[t] || t };
    series.forEach((s) => {
      const p = s.predictions.find((x) => x.task === t);
      row[s.name] = p ? +(100 * p.probability).toFixed(1) : 0;
    });
    return row;
  });
  if (tasks.length < 3) {
    // a radar needs at least three axes: fall back to bars
    return (
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 10, right: 10 }}>
          <CartesianGrid stroke="#E2E8F0" vertical={false} />
          <XAxis dataKey="axis" tick={{ fill: "#1B1F3B", fontSize: 13 }} />
          <YAxis domain={[0, 100]} unit="%" tick={{ fill: "#475569", fontSize: 12 }} />
          <Tooltip content={<ChartTooltip digits={1} />} cursor={{ fill: "#F1F5F9" }} />
          {series.map((s, i) => <Bar isAnimationActive={false} key={s.name} dataKey={s.name} fill={SERIES[i]} radius={[4, 4, 0, 0]} barSize={36} />)}
        </BarChart>
      </ResponsiveContainer>
    );
  }
  return (
    <ResponsiveContainer width="100%" height={height}>
      <RadarChart data={data} outerRadius="72%">
        <PolarGrid stroke="#E2E8F0" />
        <PolarAngleAxis dataKey="axis" tick={{ fill: "#1B1F3B", fontSize: 13, fontWeight: 600 }} />
        <PolarRadiusAxis domain={[0, 100]} tick={{ fill: "#94A3B8", fontSize: 10 }} tickCount={5} angle={90} />
        {series.map((s, i) => (
          <Radar isAnimationActive={false} key={s.name} name={s.name} dataKey={s.name} stroke={SERIES[i]} fill={SERIES[i]}
            fillOpacity={series.length > 1 ? 0.12 : 0.25} strokeWidth={2} dot={{ r: 4 }} />
        ))}
        <Tooltip content={<ChartTooltip digits={1} />} />
        {series.length > 1 && <Legend wrapperStyle={{ fontSize: 13 }} />}
      </RadarChart>
    </ResponsiveContainer>
  );
}
