import { Suspense, lazy, useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, BookmarkPlus, Check, Trash2 } from "lucide-react";
// Ketcher is a large WASM bundle: load it only when this page is opened
const MoleculeEditor = lazy(() => import("../components/MoleculeEditor.jsx"));
import { ErrorBox, Loading, SectionHeader } from "../components/ui.jsx";
import Boundary from "../components/Boundary.jsx";
import FragmentBuilder from "../components/FragmentBuilder.jsx";
import { useMolecule } from "../lib/molecule.jsx";
import { check, depict, loadRDKit } from "../lib/rdkit.js";
import { api } from "../api.js";
import { fmt } from "../lib/theme.js";

export const EXAMPLES = {
  "Common drugs": [
    ["Aspirin", "CC(=O)Oc1ccccc1C(=O)O"],
    ["Paracetamol", "CC(=O)Nc1ccc(O)cc1"],
    ["Ibuprofen", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"],
    ["Caffeine", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"],
  ],
  "Known toxic": [
    ["Nicotine", "CN1CCC[C@H]1c1cccnc1"],
    ["Bisphenol A", "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1"],
    ["Benzo[a]pyrene", "c1cc2ccc3cccc4c3c2c(c1)cc4"],
    ["Aflatoxin B1", "COc1cc2c(c3c1C(=O)C=CO3)C1C=COC1O2"],
  ],
  "Simple": [
    ["Ethanol", "CCO"],
    ["Benzene", "c1ccccc1"],
    ["Phenol", "Oc1ccccc1"],
    ["Glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"],
  ],
};

function Depiction({ rdkit, smiles, height = 210 }) {
  const svg = useMemo(() => depict(rdkit, smiles, { width: 300, height }), [rdkit, smiles, height]);
  if (!svg) {
    return (
      <div className="grid place-items-center text-sm text-ink-mute" style={{ height }}>
        Draw or type a molecule to see it here.
      </div>
    );
  }
  // eslint-disable-next-line react/no-danger
  return <div className="grid place-items-center" dangerouslySetInnerHTML={{ __html: svg }} />;
}

export default function MoleculeStudio() {
  const [params] = useSearchParams();
  const ctx = useMolecule();
  const [rdkit, setRdkit] = useState(null);
  const [text, setText] = useState(ctx.smiles);

  useEffect(() => {
    loadRDKit().then(setRdkit).catch(() => {});
  }, []);
  useEffect(() => {
    const s = params.get("smiles");
    if (s) {
      ctx.setSmiles(s);
      setText(s);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params]);
  useEffect(() => setText(ctx.smiles), [ctx.smiles]);

  const valid = useMemo(() => check(rdkit, text), [rdkit, text]);
  const props = useQuery({
    queryKey: ["mol-props", ctx.smiles],
    queryFn: () => api.moleculeProperties(ctx.smiles),
    enabled: !!ctx.smiles,
    retry: 0,
  });
  const p = props.data;

  const commit = (s) => {
    setText(s);
    if (check(rdkit, s).ok || !rdkit) ctx.setSmiles(s);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-display text-4xl font-bold">Molecule Studio</h1>
          <p className="text-ink-soft">
            Draw a molecule or type its SMILES, then run the toxicity models on it.
          </p>
        </div>
        <Link
          className={`btn-primary ${valid.ok ? "" : "pointer-events-none opacity-50"}`}
          to={`/molecule-results?smiles=${encodeURIComponent(ctx.smiles)}`}
        >
          Predict toxicity <ArrowRight size={16} />
        </Link>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)] gap-6">
        <div className="card p-5">
          <SectionHeader
            title="Draw your molecule"
            subtitle="Full chemical editor (Ketcher). Everything you draw is turned into a SMILES string below."
          />
          <Boundary
            title="The drawing canvas could not start"
            hint="Type or paste a SMILES string on the right instead — everything else works."
          >
            <Suspense fallback={<Loading text="Loading the chemical editor…" />}>
              <MoleculeEditor smiles={ctx.smiles} onChange={(s) => commit(s)} height={430} />
            </Suspense>
          </Boundary>
        </div>

        <div className="space-y-6">
          <div className="card p-5">
            <div className="label mb-1">SMILES</div>
            <textarea
              className="w-full rounded-xl border border-line bg-surface px-3 py-2 font-mono text-sm h-20 resize-none focus:outline-none focus:ring-2 focus:ring-brand/40"
              value={text}
              spellCheck={false}
              onChange={(e) => setText(e.target.value)}
              onBlur={() => commit(text)}
            />
            <div className="mt-1 text-xs">
              {!text.trim() ? (
                <span className="text-ink-mute">Empty</span>
              ) : valid.ok ? (
                <span className="text-good">
                  ✓ valid molecule · {valid.atoms} heavy atoms · {fmt(valid.weight, 1)} u
                </span>
              ) : (
                <span className="text-amber-700">! {valid.error || "checking…"}</span>
              )}
            </div>
            <div className="mt-3 flex gap-2">
              <button
                className="btn-ghost text-sm"
                disabled={!valid.ok}
                onClick={() => ctx.save(ctx.smiles)}
              >
                {ctx.isSaved ? <Check size={16} /> : <BookmarkPlus size={16} />}
                {ctx.isSaved ? "Saved" : "Save to my molecules"}
              </button>
            </div>
          </div>

          <div className="card p-5">
            <div className="label mb-2">2D structure</div>
            <Depiction rdkit={rdkit} smiles={valid.ok ? text : ctx.smiles} />
            {p && (
              <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs border-t border-line pt-3">
                <Row k="Formula" v={p.formula} mono />
                <Row k="Weight" v={`${fmt(p.descriptors.molecular_weight, 1)} u`} />
                <Row k="logP" v={fmt(p.descriptors.logP, 2)} />
                <Row k="TPSA" v={`${fmt(p.descriptors.TPSA, 1)} Å²`} />
                <Row k="QED" v={fmt(p.descriptors.QED, 2)} />
                <Row k="Rings" v={p.descriptors.rings} />
                <div className="col-span-2 mt-1">
                  <span className={p.lipinski.passes ? "text-good" : "text-amber-700"}>
                    {p.lipinski.passes ? "✓" : "!"} Lipinski rule of 5
                    {p.lipinski.violations > 0 && ` – ${p.lipinski.violations} violation(s)`}
                  </span>
                </div>
              </div>
            )}
            <ErrorBox error={props.error} />
          </div>
        </div>
      </div>

      <div className="card p-5">
        <SectionHeader
          title="Build a molecule from fragments"
          subtitle="No drawing needed: combine a ring system with a substituent. Every combination is a valid molecule."
        />
        <FragmentBuilder rdkit={rdkit} onPick={commit} />
      </div>

      <div className="card p-5">
        <SectionHeader title="Example library" subtitle="Click a molecule to load it into the editor." />
        <div className="space-y-3">
          {Object.entries(EXAMPLES).map(([group, items]) => (
            <div key={group}>
              <div className="label mb-1">{group}</div>
              <div className="flex flex-wrap gap-2">
                {items.map(([name, s]) => (
                  <button
                    key={s}
                    onClick={() => commit(s)}
                    className={`chip ${ctx.smiles === s ? "!border-brand !text-brand" : ""}`}
                  >
                    {name}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {ctx.saved.length > 0 && (
        <div className="card p-5">
          <SectionHeader
            title={`My molecules (${ctx.saved.length})`}
            subtitle="Saved in this browser. Used by the screening page."
          />
          <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-3">
            {ctx.saved.map((m) => (
              <div key={m.smiles} className="rounded-xl border border-line p-2 text-center">
                <button className="w-full" onClick={() => commit(m.smiles)}>
                  <Depiction rdkit={rdkit} smiles={m.smiles} height={110} />
                  <div className="truncate font-mono text-[10px] text-ink-mute">{m.smiles}</div>
                </button>
                <button
                  className="mt-1 text-[11px] text-ink-mute hover:text-red-600 inline-flex items-center gap-1"
                  onClick={() => ctx.remove(m.smiles)}
                >
                  <Trash2 size={11} /> remove
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Row({ k, v, mono }) {
  return (
    <>
      <span className="text-ink-mute">{k}</span>
      <span className={`text-right text-ink ${mono ? "font-mono" : ""}`}>{v}</span>
    </>
  );
}
