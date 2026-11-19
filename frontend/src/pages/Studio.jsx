import { useEffect } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { ArrowRight, Eraser } from "lucide-react";
import PeriodicTable from "../components/PeriodicTable.jsx";
import CompositionDonut from "../components/CompositionDonut.jsx";
import CrystalViewer from "../components/CrystalViewer.jsx";
import { useComposition } from "../lib/composition.jsx";

const LIBRARY = [
  { family: "Battery cathodes", items: ["LiFePO4", "LiCoO2", "LiMn2O4", "LiNiO2"] },
  { family: "Solar absorbers", items: ["CsPbI3", "CdTe", "GaAs", "Cu2ZnSnS4"] },
  { family: "Magnets", items: ["Nd2Fe14B", "SmCo5", "Fe3O4", "MnBi"] },
  { family: "Alloys & others", items: ["Ni3Al", "TiAl", "SrTiO3", "NaCl"] },
];

export default function Studio() {
  const nav = useNavigate();
  const [params] = useSearchParams();
  const { counts, setCounts, text, setFormulaText, error, formula } = useComposition();
  useEffect(() => {
    const f = params.get("formula");
    if (f) setFormulaText(f);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-display text-3xl font-bold">Composition Studio</h1>
          <p className="text-ink-soft">Build a material from the periodic table or type its formula, inspect its crystal, then run the models.</p>
        </div>
        <button className="btn-primary text-base px-5 py-3" disabled={!formula || !!error}
          onClick={() => nav(`/results?formula=${encodeURIComponent(formula)}`)}>
          Predict properties <ArrowRight size={18} />
        </button>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_380px] gap-6">
        <div className="card p-5">
          <PeriodicTable counts={counts} onChange={setCounts} />
        </div>
        <div className="space-y-4">
          <div className="card p-5">
            <div className="label mb-2">Formula</div>
            <input value={text} onChange={(e) => setFormulaText(e.target.value)} spellCheck={false}
              className={`w-full rounded-xl border-2 px-4 py-3 font-mono text-2xl font-semibold outline-none transition ${error ? "border-red-400" : "border-line focus:border-brand"}`}
              placeholder="e.g. LiFePO4" aria-label="Chemical formula" />
            <div className="mt-1 flex items-center justify-between text-xs">
              {error ? <span className="text-red-600">Invalid formula: {error}</span> : <span className="text-good font-semibold">✓ valid composition</span>}
              <button className="flex items-center gap-1 text-ink-mute hover:text-ink" onClick={() => setCounts({})}><Eraser size={12} /> clear</button>
            </div>
            {Object.keys(counts).length > 0 && (
              <div className="mt-4 space-y-2">
                {Object.entries(counts).map(([el, n]) => (
                  <div key={el} className="grid grid-cols-[44px_1fr_28px] items-center gap-2">
                    <span className="rounded-lg bg-brand-50 py-1 text-center font-bold text-brand">{el}</span>
                    <input type="range" min="0.5" max="20" step="0.5" value={n} aria-label={`amount of ${el}`}
                      onChange={(e) => setCounts({ ...counts, [el]: parseFloat(e.target.value) })} className="accent-brand" />
                    <span className="font-mono text-sm text-right">{n}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
          <div className="card p-5">
            <div className="label">Atomic fractions</div>
            <CompositionDonut counts={counts} height={210} />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_380px] gap-6">
        <div className="card p-5">
          <div className="flex items-center justify-between mb-3">
            <div className="label">3D crystal · lowest-energy known structure (Alexandria OPTIMADE)</div>
          </div>
          <CrystalViewer formula={formula} height={380} />
        </div>
        <div className="card p-5">
          <div className="label mb-3">Example library</div>
          <div className="space-y-4">
            {LIBRARY.map((g) => (
              <div key={g.family}>
                <div className="text-sm font-semibold mb-1.5">{g.family}</div>
                <div className="flex flex-wrap gap-2">
                  {g.items.map((f) => (
                    <button key={f} className={`chip font-mono ${f === formula ? "!border-brand bg-brand-50 text-brand" : ""}`} onClick={() => setFormulaText(f)}>{f}</button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
