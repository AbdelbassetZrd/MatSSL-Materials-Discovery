import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useQueries } from "@tanstack/react-query";
import { Plus, X } from "lucide-react";
import { api } from "../api.js";
import ProfileRadar from "../components/ProfileRadar.jsx";
import { Loading } from "../components/ui.jsx";
import { SERIES, TASK } from "../lib/theme.js";

const PRESETS = [
  ["LiFePO4", "LiMnPO4", "LiCoO2"],
  ["CsPbI3", "CdTe", "GaAs"],
  ["Nd2Fe14B", "SmCo5", "Fe3O4"],
];

export default function Compare() {
  const [params, setParams] = useSearchParams();
  const list = (params.get("f") || "LiFePO4,LiMnPO4,LiCoO2").split(",").filter(Boolean).slice(0, 4);
  const [draft, setDraft] = useState("");
  const setList = (l) => setParams({ f: l.join(",") });
  const results = useQueries({
    queries: list.map((f) => ({ queryKey: ["predict", f], queryFn: () => api.predict(f) })),
  });
  const ready = results.every((r) => r.data);
  const series = ready ? results.map((r, i) => ({ name: list[i], predictions: r.data.predictions })) : [];
  const tasks = ready ? series[0].predictions.map((p) => p.task) : [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-3xl font-bold">Compare candidates</h1>
        <p className="text-ink-soft">Up to four compositions side by side – which one should be synthesized or computed next?</p>
      </div>
      <div className="card p-5 flex flex-wrap items-center gap-2">
        {list.map((f, i) => (
          <span key={f} className="chip font-mono !cursor-default" style={{ borderColor: SERIES[i] }}>
            <i className="h-2.5 w-2.5 rounded-full" style={{ background: SERIES[i] }} />
            {f}
            <button aria-label={`remove ${f}`} onClick={() => setList(list.filter((x) => x !== f))}><X size={14} /></button>
          </span>
        ))}
        {list.length < 4 && (
          <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); if (draft.trim()) { setList([...list, draft.trim()]); setDraft(""); } }}>
            <input value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="add formula…" aria-label="formula to add"
              className="rounded-xl border border-line px-3 py-1.5 font-mono text-sm outline-none focus:border-brand" />
            <button className="btn-ghost text-sm py-1.5"><Plus size={14} /> Add</button>
          </form>
        )}
        <span className="ml-auto flex flex-wrap gap-2 text-xs text-ink-mute">Presets:
          {PRESETS.map((p) => <button key={p[0]} className="chip text-xs" onClick={() => setList(p)}>{p.join(" · ")}</button>)}
        </span>
      </div>

      {!ready ? <Loading text="Running the models…" /> : (
        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)] gap-6">
          <div className="card p-5">
            <h2 className="font-display text-xl font-bold">Property profiles</h2>
            <ProfileRadar series={series} height={380} />
          </div>
          <div className="card p-5 overflow-x-auto">
            <h2 className="font-display text-xl font-bold mb-3">Side by side</h2>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-ink-mute border-b border-line">
                  <th className="py-2">Property</th>
                  {list.map((f, i) => <th key={f} className="font-mono" style={{ color: SERIES[i] }}>{f}</th>)}
                </tr>
              </thead>
              <tbody>
                {tasks.map((t) => (
                  <tr key={t} className="border-b border-line/60">
                    <td className="py-3 font-semibold">{TASK[t]?.title}</td>
                    {series.map((s) => {
                      const p = s.predictions.find((x) => x.task === t);
                      return (
                        <td key={s.name} className="pr-2">
                          <div className={p.prediction ? "font-bold text-brand" : "text-ink-soft"}>{p.prediction_label}</div>
                          <div className="mt-1 h-1.5 w-28 rounded bg-surface"><div className="h-1.5 rounded bg-brand" style={{ width: `${100 * p.probability}%` }} /></div>
                          <div className="font-mono text-xs text-ink-mute">{(100 * p.probability).toFixed(1)}%</div>
                        </td>
                      );
                    })}
                  </tr>
                ))}
                <tr>
                  <td className="py-3 font-semibold">Applicability domain</td>
                  {results.map((r, i) => (
                    <td key={list[i]} className={r.data.applicability_domain?.inside ? "text-good font-semibold" : "text-amber-700 font-semibold"}>
                      {r.data.applicability_domain?.inside ? "✓ inside" : "! outside"}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
            <p className="mt-3 text-xs text-ink-mute">Open a material in <Link className="text-brand underline" to={`/results?formula=${encodeURIComponent(list[0])}`}>Results</Link> for its explanation and nearest neighbours.</p>
          </div>
        </div>
      )}
    </div>
  );
}
