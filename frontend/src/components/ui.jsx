import { Loader2 } from "lucide-react";

export function Stat({ value, label, tone = "brand", sub }) {
  const color = { brand: "text-brand", cyan: "text-cyan", amber: "text-amber-600", good: "text-good", ink: "text-ink" }[tone];
  return (
    <div className="card p-5">
      <div className={`font-display text-4xl font-bold ${color}`}>{value}</div>
      <div className="mt-1 text-sm font-semibold text-ink">{label}</div>
      {sub && <div className="text-xs text-ink-mute mt-0.5">{sub}</div>}
    </div>
  );
}

export function SectionHeader({ title, subtitle, right }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-3 mb-4">
      <div>
        <h2 className="h-section">{title}</h2>
        {subtitle && <p className="text-ink-soft text-sm mt-1 max-w-3xl">{subtitle}</p>}
      </div>
      {right}
    </div>
  );
}

export function Loading({ text = "Loading…" }) {
  return (
    <div className="flex items-center gap-2 text-ink-mute py-10 justify-center">
      <Loader2 className="animate-spin" size={18} /> {text}
    </div>
  );
}

export function ErrorBox({ error }) {
  if (!error) return null;
  return <div className="rounded-xl bg-red-50 text-red-700 px-4 py-3 text-sm">{String(error.message || error)}</div>;
}

export function Tabs({ tabs, value, onChange }) {
  return (
    <div className="flex flex-wrap gap-1 rounded-2xl bg-white p-1 border border-line w-fit">
      {tabs.map((t) => (
        <button key={t.id} onClick={() => onChange(t.id)}
          className={`rounded-xl px-4 py-2 text-sm font-semibold transition ${value === t.id ? "bg-brand text-white" : "text-ink-soft hover:bg-surface"}`}>
          {t.label}
        </button>
      ))}
    </div>
  );
}

export function Segmented({ options, value, onChange, size = "sm" }) {
  return (
    <div className="inline-flex rounded-xl border border-line bg-white p-0.5">
      {options.map((o) => (
        <button key={o.value} onClick={() => onChange(o.value)}
          className={`rounded-lg px-3 ${size === "sm" ? "py-1 text-xs" : "py-1.5 text-sm"} font-semibold ${value === o.value ? "bg-ink text-white" : "text-ink-soft hover:text-ink"}`}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function ChartTooltip({ active, payload, label, digits = 3, labelFormatter }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl border border-line bg-white px-3 py-2 shadow-card text-xs">
      {label != null && <div className="font-semibold text-ink mb-1">{labelFormatter ? labelFormatter(label) : label}</div>}
      {payload.map((p) => (
        <div key={p.dataKey + p.name} className="flex items-center gap-2 text-ink-soft">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: p.color || p.fill }} />
          <span>{p.name}</span>
          <span className="ml-auto font-mono text-ink">{typeof p.value === "number" ? p.value.toFixed(digits) : p.value}</span>
        </div>
      ))}
    </div>
  );
}
