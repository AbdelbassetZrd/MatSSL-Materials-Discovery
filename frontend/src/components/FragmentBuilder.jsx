import { useMemo, useState } from "react";
import { depict } from "../lib/rdkit.js";

// Scaffold templates with one attachment point, so every combination is a valid molecule.
const SCAFFOLDS = [
  ["Benzene", "c1ccc({R})cc1"],
  ["Pyridine", "c1ccc({R})nc1"],
  ["Cyclohexane", "C1CCC({R})CC1"],
  ["Naphthalene", "c1ccc2cc({R})ccc2c1"],
  ["Indole", "c1ccc2[nH]cc({R})c2c1"],
  ["Piperazine", "C1CN(CCN1){R}"],
  ["Imidazole", "c1cn({R})cn1"],
  ["Thiophene", "c1ccc({R})s1"],
];
const GROUPS = [
  ["–H", ""],
  ["–OH", "O"],
  ["–NH2", "N"],
  ["–CH3", "C"],
  ["–COOH", "C(=O)O"],
  ["–CHO", "C=O"],
  ["–OCH3", "OC"],
  ["–Cl", "Cl"],
  ["–F", "F"],
  ["–Br", "Br"],
  ["–CF3", "C(F)(F)F"],
  ["–NO2", "[N+](=O)[O-]"],
  ["–C≡N", "C#N"],
  ["–SO2NH2", "S(=O)(=O)N"],
  ["–NHC(=O)CH3", "NC(C)=O"],
];

const build = (scaffold, group) =>
  group === "" ? scaffold.replace("({R})", "").replace("{R}", "") : scaffold.replace("{R}", group);

/**
 * Build a molecule without drawing it: pick a ring system and a substituent. Every
 * combination is assembled into a valid SMILES string, so it always produces a real molecule.
 */
export default function FragmentBuilder({ rdkit, onPick }) {
  const [scaffold, setScaffold] = useState(SCAFFOLDS[0][1]);
  const [group, setGroup] = useState(GROUPS[4][1]);
  const smiles = useMemo(() => build(scaffold, group), [scaffold, group]);
  const svg = useMemo(() => depict(rdkit, smiles, { width: 260, height: 150 }), [rdkit, smiles]);

  const Row = ({ title, items, value, set }) => (
    <div>
      <div className="label mb-1">{title}</div>
      <div className="flex flex-wrap gap-1.5">
        {items.map(([label, v]) => (
          <button
            key={label}
            onClick={() => set(v)}
            className={`chip !px-2 !py-1 !text-xs ${value === v ? "!border-brand !text-brand !bg-brand-50" : ""}`}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  );

  return (
    <div className="space-y-3">
      <Row title="Ring system" items={SCAFFOLDS} value={scaffold} set={setScaffold} />
      <Row title="Substituent" items={GROUPS} value={group} set={setGroup} />
      <div className="flex items-center gap-4 rounded-xl bg-surface p-3">
        <div className="shrink-0">
          {/* eslint-disable-next-line react/no-danger */}
          {svg ? <div dangerouslySetInnerHTML={{ __html: svg }} /> : <div className="h-[150px] w-[260px]" />}
        </div>
        <div className="min-w-0 flex-1">
          <div className="label">Result</div>
          <div className="break-all font-mono text-xs text-ink">{smiles}</div>
          <button className="btn-primary mt-2 text-sm" onClick={() => onPick(smiles)}>
            Use this molecule
          </button>
        </div>
      </div>
    </div>
  );
}
