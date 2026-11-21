import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, ChevronDown, ChevronRight, Info, Beaker } from "lucide-react";
import { ErrorBox, Loading, SectionHeader, Tabs } from "../components/ui.jsx";
import { useMolecule } from "../lib/molecule.jsx";
import { depict, loadRDKit } from "../lib/rdkit.js";
import { api } from "../api.js";
import { fmt } from "../lib/theme.js";

const VERDICT = {
  "worth testing": "bg-emerald-50 text-emerald-800 border-emerald-200",
  unclear: "bg-amber-50 text-amber-900 border-amber-200",
  unlikely: "bg-surface text-ink-mute border-line",
  "no model": "bg-surface text-ink-mute border-line",
};
const CONF = { high: "text-good", medium: "text-amber-600", low: "text-red-600", unknown: "text-ink-mute" };

const heat = (s) =>
  s == null ? "#F1F5F9" : `rgba(79,70,229,${(0.08 + 0.92 * Math.min(s, 1)).toFixed(2)})`;

function TargetTable({ targets }) {
  return (
    <table className="w-full text-xs mt-2">
      <thead>
        <tr className="text-left text-ink-mute border-b border-line">
          <th className="py-1">Protein</th><th>Evidence</th><th>P(active)</th>
          <th>Nearest known active</th><th>Reliability</th><th>Model AUC</th>
        </tr>
      </thead>
      <tbody>
        {targets.map((t) => (
          <tr key={t.chembl_id} className="border-b border-line/60">
            <td className="py-1.5">
              <span className="font-mono font-semibold">{t.symbol}</span>
              <span className="block text-[10px] text-ink-mute truncate max-w-[180px]">{t.name}</span>
            </td>
            <td className="font-mono">{fmt(t.association_score, 2)}</td>
            <td>
              {t.modelled ? (
                <div className="flex items-center gap-1.5">
                  <div className="h-2 w-14 rounded bg-surface overflow-hidden">
                    <div className="h-2 bg-brand" style={{ width: `${100 * t.probability}%` }} />
                  </div>
                  <span className="font-mono">{(100 * t.probability).toFixed(0)}%</span>
                </div>
              ) : <span className="text-ink-mute">no model</span>}
            </td>
            <td className="font-mono text-[10px] max-w-[150px] truncate" title={t.nearest_active || ""}>
              {t.similarity != null ? `${fmt(t.similarity, 2)} · ${t.nearest_active || ""}` : "–"}
            </td>
            <td className={CONF[t.confidence] || "text-ink-mute"}>
              {t.modelled ? t.confidence : "–"}
            </td>
            {/* the AUC over measured molecules, with the count of rarer-class molecules it
                rests on — a high AUC on a handful of them is not evidence */}
            <td className="font-mono text-[10px]">
              {t.modelled && (t.roc_auc_measured ?? t.test_roc_auc) != null ? (
                <span
                  className={t.roc_auc_measured != null && !t.well_supported ? "text-amber-700" : ""}
                  title={
                    t.roc_auc_measured != null
                      ? `measured molecules only, on ${t.n_measured_minority} of the rarer class`
                      : "all held-out molecules, including presumed-inactive ZINC decoys"
                  }
                >
                  {fmt(t.roc_auc_measured ?? t.test_roc_auc, 2)}
                  {t.n_measured_minority != null && (
                    <span className="text-ink-mute"> ·{t.n_measured_minority}</span>
                  )}
                </span>
              ) : "–"}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Ranking({ smiles }) {
  const [open, setOpen] = useState(null);
  const { data, isLoading, error } = useQuery({
    queryKey: ["screen", smiles],
    queryFn: () => api.screen(smiles),
    enabled: !!smiles,
    retry: 0,
  });
  if (isLoading) return <Loading text="Running every protein model…" />;
  if (error) return <ErrorBox error={error} />;
  if (!data) return null;
  const scored = data.results.filter((r) => r.score != null);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-4 text-sm">
        <span><b>{data.n_targets_modelled}</b> protein models run</span>
        <span><b>{data.target_hits.length}</b> predicted active:{" "}
          <span className="font-mono text-brand">
            {data.target_hits.slice(0, 6).map((h) => h.symbol).join(", ") || "none"}
          </span>
        </span>
        {data.toxicity_flags != null && (
          <span className={data.toxicity_flags >= 3 ? "text-amber-700" : ""}>
            <b>{data.toxicity_flags}</b> toxicity endpoints flagged
          </span>
        )}
      </div>

      <div className="space-y-2">
        {data.results.map((r) => {
          const isOpen = open === r.disease_id;
          return (
            <div key={r.disease_id} className={`rounded-xl border ${VERDICT[r.verdict] || VERDICT.unlikely}`}>
              <button
                className="flex w-full items-center gap-3 px-4 py-3 text-left"
                onClick={() => setOpen(isOpen ? null : r.disease_id)}
              >
                {isOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                <span className="font-semibold min-w-[150px]">{r.label}</span>
                <div className="h-2 flex-1 rounded bg-white/70 overflow-hidden max-w-[220px]">
                  <div className="h-2 rounded bg-brand"
                       style={{ width: `${100 * (r.score ?? 0)}%` }} />
                </div>
                <span className="font-mono text-sm w-12 text-right">
                  {r.score == null ? "–" : (100 * r.score).toFixed(0)}
                </span>
                <span className="text-xs font-semibold uppercase tracking-wide">{r.verdict}</span>
                {r.confidence && (
                  <span className={`text-[11px] ${CONF[r.confidence]}`}>{r.confidence} confidence</span>
                )}
              </button>
              {isOpen && (
                <div className="border-t border-current/10 px-4 py-3">
                  <p className="text-xs">{r.note}</p>
                  <TargetTable targets={r.targets} />
                </div>
              )}
            </div>
          );
        })}
      </div>
      {scored.length === 0 && (
        <p className="text-sm text-ink-mute">
          None of these illnesses has a protein model yet — run the training pipeline first.
        </p>
      )}
    </div>
  );
}

function Matrix({ molecules }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["screen-matrix", molecules.join("|")],
    queryFn: () => api.screenMatrix(molecules),
    enabled: molecules.length > 0,
    retry: 0,
  });
  if (!molecules.length) {
    return (
      <p className="text-sm text-ink-mute">
        Save molecules in the{" "}
        <Link className="text-brand underline" to="/molecules">Molecule Studio</Link> to compare
        them side by side.
      </p>
    );
  }
  if (isLoading) return <Loading text="Screening every molecule…" />;
  if (error) return <ErrorBox error={error} />;

  return (
    <div className="overflow-x-auto">
      <table className="text-xs border-separate border-spacing-0.5">
        <thead>
          <tr>
            <th className="sticky left-0 bg-white" />
            {data.diseases.map((d) => (
              <th key={d.id} className="h-28 w-9 align-bottom">
                <div className="origin-bottom-left -rotate-45 translate-y-2 whitespace-nowrap text-ink-soft">
                  {d.label}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.molecules.map((m) => (
            <tr key={m.smiles}>
              <td className="sticky left-0 bg-white pr-2 font-mono text-[10px] max-w-[160px] truncate"
                  title={m.smiles}>
                {m.smiles}
              </td>
              {m.scores.map((s) => (
                <td key={s.disease_id} className="h-8 w-9 text-center"
                    title={`${s.disease_id}: ${s.score == null ? "no model" : (100 * s.score).toFixed(0)} (${s.verdict})`}
                    style={{ background: heat(s.score) }}>
                  <span className={s.score > 0.55 ? "text-white font-semibold" : "text-ink-soft"}>
                    {s.score == null ? "" : (100 * s.score).toFixed(0)}
                  </span>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Screen() {
  const ctx = useMolecule();
  const [rdkit, setRdkit] = useState(null);
  const [tab, setTab] = useState("one");
  const [smiles, setSmiles] = useState(ctx.smiles);
  useEffect(() => {
    loadRDKit().then(setRdkit).catch(() => {});
  }, []);
  useEffect(() => setSmiles(ctx.smiles), [ctx.smiles]);

  const choices = useMemo(() => {
    const seen = new Set();
    return [{ smiles: ctx.smiles, name: "current molecule" }, ...ctx.saved].filter((m) => {
      if (!m.smiles || seen.has(m.smiles)) return false;
      seen.add(m.smiles);
      return true;
    });
  }, [ctx.smiles, ctx.saved]);
  const svg = useMemo(() => depict(rdkit, smiles, { width: 240, height: 150 }), [rdkit, smiles]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-4xl font-bold">Screen against illnesses</h1>
        <p className="text-ink-soft">
          Every illness ranked for your molecule, from the predicted activity against the
          proteins linked to it.
        </p>
      </div>

      <div className="flex items-start gap-2 rounded-xl bg-amber-50 px-4 py-3 text-sm text-amber-900">
        <AlertTriangle size={18} className="mt-0.5 shrink-0" />
        <span>
          <b>This is in-silico screening, not a treatment prediction.</b> A model trained on
          measured data can say whether a molecule is likely to bind a protein. Whether that
          helps a patient depends on absorption, dose, selectivity and safety — none of which
          are modelled here. Use it to decide what is worth testing, nothing more.
        </span>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[260px_minmax(0,1fr)] gap-6">
        <div className="card p-4 space-y-3 h-fit">
          <div className="label">Molecule</div>
          {/* eslint-disable-next-line react/no-danger */}
          {svg ? <div className="grid place-items-center" dangerouslySetInnerHTML={{ __html: svg }} />
               : <div className="h-[150px] grid place-items-center text-xs text-ink-mute">no structure</div>}
          <select
            className="w-full rounded-xl border border-line bg-surface px-2 py-2 font-mono text-xs"
            value={smiles}
            onChange={(e) => setSmiles(e.target.value)}
          >
            {choices.map((m) => (
              <option key={m.smiles} value={m.smiles}>
                {m.name ? `${m.name} — ` : ""}{m.smiles.slice(0, 38)}
              </option>
            ))}
          </select>
          <Link className="btn-ghost w-full justify-center text-sm" to="/molecules">
            <Beaker size={15} /> Build another
          </Link>
          <p className="flex gap-1.5 text-[11px] text-ink-mute">
            <Info size={12} className="mt-0.5 shrink-0" />
            Scores combine each protein's predicted activity, weighted by how strongly public
            evidence links it to the illness.
          </p>
        </div>

        <div className="card p-5">
          <SectionHeader
            title={tab === "one" ? "Illnesses ranked for this molecule" : "All my molecules"}
            right={<Tabs tabs={[{ id: "one", label: "One molecule" },
                                { id: "all", label: `My molecules (${ctx.saved.length})` }]}
                         value={tab} onChange={setTab} />}
          />
          {tab === "one"
            ? <Ranking smiles={smiles} />
            : <Matrix molecules={ctx.saved.map((m) => m.smiles)} />}
        </div>
      </div>
    </div>
  );
}
