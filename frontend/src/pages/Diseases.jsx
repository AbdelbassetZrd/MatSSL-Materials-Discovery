import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Info, Search, TestTube } from "lucide-react";
import { ErrorBox, Loading, SectionHeader } from "../components/ui.jsx";
import { depict, loadRDKit } from "../lib/rdkit.js";
import { api } from "../api.js";
import { fmt } from "../lib/theme.js";

function TargetRow({ t }) {
  return (
    <div className="grid grid-cols-[88px_1fr_auto] items-center gap-2 text-xs">
      <span className="font-mono font-semibold text-ink">{t.symbol}</span>
      <div className="h-2 rounded bg-surface overflow-hidden" title="evidence linking this protein to the illness">
        <div className="h-2 rounded bg-brand" style={{ width: `${100 * t.association_score}%` }} />
      </div>
      <span className={t.modelled ? "text-good" : "text-ink-mute"}>
        {t.modelled ? `model AUC ${fmt(t.test_roc_auc, 2)}` : "no model"}
      </span>
    </div>
  );
}

function Detail({ id, rdkit }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["disease", id],
    queryFn: () => api.disease(id),
    enabled: !!id,
  });
  if (isLoading) return <Loading />;
  if (error) return <ErrorBox error={error} />;
  if (!data) return null;
  return (
    <div className="space-y-4">
      <div>
        <h2 className="font-display text-2xl font-bold">{data.name}</h2>
        <p className="mt-1 text-sm text-ink-soft">{data.description || "No description."}</p>
      </div>
      <div>
        <div className="label mb-2">Associated proteins (Open Targets evidence score)</div>
        <div className="space-y-1.5">
          {data.targets.map((t) => <TargetRow key={t.chembl_id} t={t} />)}
        </div>
      </div>
      {data.known_drugs?.length > 0 && (
        <div>
          <div className="label mb-2">Already approved for this illness ({data.known_drugs.length})</div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {data.known_drugs.slice(0, 8).map((d) => {
              const svg = depict(rdkit, d.smiles, { width: 150, height: 90 });
              return (
                <div key={d.chembl_id} className="rounded-lg border border-line p-1.5 text-center">
                  {/* eslint-disable-next-line react/no-danger */}
                  {svg ? <div dangerouslySetInnerHTML={{ __html: svg }} /> : <div className="h-[90px]" />}
                  <div className="truncate text-[10px] text-ink-soft" title={d.name}>{d.name}</div>
                </div>
              );
            })}
          </div>
        </div>
      )}
      <p className="flex gap-2 rounded-xl bg-surface px-3 py-2 text-xs text-ink-soft">
        <Info size={14} className="mt-0.5 shrink-0" /> {data.disclaimer}
      </p>
    </div>
  );
}

export default function Diseases() {
  const [rdkit, setRdkit] = useState(null);
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(null);
  useEffect(() => {
    loadRDKit().then(setRdkit).catch(() => {});
  }, []);

  const { data, isLoading, error } = useQuery({ queryKey: ["diseases"], queryFn: api.diseases });
  const list = useMemo(() => {
    const all = data?.diseases || [];
    const q = query.trim().toLowerCase();
    return q ? all.filter((d) => (d.label + d.targets.join(" ")).toLowerCase().includes(q)) : all;
  }, [data, query]);

  useEffect(() => {
    if (!open && list.length) setOpen(list[0].id);
  }, [list, open]);

  if (isLoading) return <Loading text="Loading the illness knowledge base…" />;
  if (error) return <ErrorBox error={error} />;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-4xl font-bold">Illnesses</h1>
        <p className="text-ink-soft">
          Each illness is linked to the proteins that public evidence associates with it. Where
          enough activity has been measured, that protein has a model — and your molecules can be
          screened against it.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <div className="relative">
          <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-mute" />
          <input
            className="rounded-xl border border-line bg-white py-2 pl-9 pr-3 text-sm w-64"
            placeholder="search illness or protein…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <Link className="btn-primary text-sm ml-auto" to="/screen">
          <TestTube size={16} /> Screen my molecules
        </Link>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)] gap-6">
        <div className="space-y-2">
          {list.map((d) => (
            <button
              key={d.id}
              onClick={() => setOpen(d.id)}
              className={`card w-full p-4 text-left transition ${
                open === d.id ? "ring-2 ring-brand" : "hover:border-brand/40"
              }`}
            >
              <div className="flex items-baseline justify-between gap-2">
                <span className="font-display text-lg font-bold">{d.label}</span>
                <span className={`text-xs ${d.n_modelled ? "text-good" : "text-amber-700"}`}>
                  {d.n_modelled}/{d.n_targets} proteins modelled
                </span>
              </div>
              <div className="mt-1 font-mono text-[11px] text-ink-mute">
                {d.targets.join(" · ")}
              </div>
            </button>
          ))}
          {list.length === 0 && <p className="text-sm text-ink-mute">No illness matches.</p>}
        </div>
        <div className="card p-5">
          <SectionHeader title="Details" subtitle="Proteins, evidence strength and known drugs." />
          <Detail id={open} rdkit={rdkit} />
        </div>
      </div>
    </div>
  );
}
