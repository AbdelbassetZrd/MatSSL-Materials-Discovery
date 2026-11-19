import { useMemo, useState } from "react";
import elements from "../data/elements.json";
import { Segmented } from "./ui.jsx";

const CAT = {
  alkali: ["Alkali metal", "#FECACA"], alkaline: ["Alkaline earth", "#FED7AA"],
  transition: ["Transition metal", "#FDE68A"], "post-transition": ["Post-transition", "#D9F99D"],
  metalloid: ["Metalloid", "#A7F3D0"], nonmetal: ["Non-metal", "#BAE6FD"],
  halogen: ["Halogen", "#C7D2FE"], noble: ["Noble gas", "#E9D5FF"],
  lanthanide: ["Lanthanide", "#FBCFE8"], actinide: ["Actinide", "#F5D0FE"],
};
const HEAT = { en: "Electronegativity", radius: "Atomic radius (Å)", melting: "Melting point (K)" };

// sequential single-hue ramp (light -> dark indigo)
function ramp(t) {
  const a = [238, 242, 255];
  const b = [49, 46, 129];
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",")})`;
}

export default function PeriodicTable({ counts, onChange }) {
  const [mode, setMode] = useState("cat");
  const [hover, setHover] = useState(null);
  const range = useMemo(() => {
    if (mode === "cat") return null;
    const v = elements.map((e) => e[mode]).filter((x) => x != null);
    return [Math.min(...v), Math.max(...v)];
  }, [mode]);
  const level = (e) => (e[mode] - range[0]) / (range[1] - range[0] || 1);
  const fill = (e) => {
    if (mode === "cat") return CAT[e.cat]?.[1] || "#F1F5F9";
    return e[mode] == null ? "#F1F5F9" : ramp(level(e));
  };
  const dark = (e) => mode !== "cat" && e[mode] != null && level(e) > 0.55;
  const toggle = (sym) => {
    const next = { ...counts };
    if (next[sym]) delete next[sym];
    else next[sym] = 1;
    onChange(next);
  };
  const shown = hover || elements.find((e) => counts[e.symbol]) || elements[25];

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <span className="label">Click elements to add / remove them</span>
        <Segmented
          value={mode}
          onChange={setMode}
          options={[
            { value: "cat", label: "Families" },
            { value: "en", label: "Electronegativity" },
            { value: "radius", label: "Radius" },
            { value: "melting", label: "Melting point" },
          ]}
        />
      </div>
      <div
        className="grid gap-[3px]"
        style={{ gridTemplateColumns: "repeat(18, minmax(0, 1fr))", gridTemplateRows: "repeat(7, 44px) 10px repeat(2, 44px)" }}
      >
        {elements.map((e) => {
          const sel = !!counts[e.symbol];
          return (
            <button
              key={e.z}
              onClick={() => toggle(e.symbol)}
              onMouseEnter={() => setHover(e)}
              onMouseLeave={() => setHover(null)}
              style={{ gridRow: e.row, gridColumn: e.group, background: sel ? "#4F46E5" : fill(e) }}
              aria-pressed={sel}
              aria-label={`${e.name}${sel ? " (selected)" : ""}`}
              className={`relative rounded-md border border-black/5 transition hover:scale-110 hover:z-10 hover:shadow-card ${
                sel ? "text-white ring-2 ring-brand-900" : dark(e) ? "text-white" : "text-ink"
              }`}
            >
              <span className={`absolute left-1 top-0.5 text-[9px] ${sel || dark(e) ? "text-white/80" : "text-ink/50"}`}>{e.z}</span>
              <span className="text-[15px] font-bold">{e.symbol}</span>
            </button>
          );
        })}
        <div
          className="rounded-xl border border-line bg-white p-3 text-xs leading-relaxed shadow-sm"
          style={{ gridRow: "1 / 4", gridColumn: "3 / 13" }}
        >
          <div className="flex items-baseline gap-2">
            <span className="font-display text-3xl font-bold text-ink">{shown.symbol}</span>
            <span className="text-sm font-semibold text-ink">{shown.name}</span>
            <span className="ml-auto text-ink-mute">Z = {shown.z}</span>
          </div>
          <div className="mt-1 grid grid-cols-2 gap-x-4 text-ink-soft">
            <span>Family: <b className="text-ink">{CAT[shown.cat]?.[0]}</b></span>
            <span>Electronegativity: <b className="text-ink">{shown.en ?? "–"}</b></span>
            <span>Atomic mass: <b className="text-ink">{shown.mass ?? "–"}</b></span>
            <span>Atomic radius: <b className="text-ink">{shown.radius ? `${shown.radius} Å` : "–"}</b></span>
            <span>Melting point: <b className="text-ink">{shown.melting ? `${shown.melting} K` : "–"}</b></span>
            <span>In composition: <b className="text-ink">{counts[shown.symbol] ?? 0}</b></span>
          </div>
        </div>
      </div>
      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-mute">
        {mode === "cat" ? (
          Object.values(CAT).map(([l, c]) => (
            <span key={l} className="flex items-center gap-1">
              <i className="h-2.5 w-2.5 rounded-sm" style={{ background: c }} />
              {l}
            </span>
          ))
        ) : (
          <span className="flex items-center gap-2">
            {HEAT[mode]}: {range[0].toFixed(1)}
            <i className="h-2.5 w-32 rounded" style={{ background: `linear-gradient(90deg, ${ramp(0)}, ${ramp(1)})` }} />
            {range[1].toFixed(1)}
          </span>
        )}
      </div>
    </div>
  );
}
