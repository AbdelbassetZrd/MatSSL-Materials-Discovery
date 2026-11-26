import { Fragment, useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  Bar, BarChart, CartesianGrid, Cell, PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar,
  RadarChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { AlertTriangle, ArrowLeft, Download, ShieldCheck } from "lucide-react";
import Gauge from "../components/Gauge.jsx";
import { ChartTooltip, ErrorBox, Loading, SectionHeader, Segmented } from "../components/ui.jsx";
import { useMolecule } from "../lib/molecule.jsx";
import { depict, loadRDKit } from "../lib/rdkit.js";
import { api } from "../api.js";
import { AXIS, GRID, NEG, POS, fmt } from "../lib/theme.js";

const RISK = {
  Low: { cls: "bg-emerald-50 text-emerald-800", Icon: ShieldCheck },
  Moderate: { cls: "bg-amber-50 text-amber-900", Icon: AlertTriangle },
  High: { cls: "bg-red-50 text-red-800", Icon: AlertTriangle },
};

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

function EndpointCard({ p }) {
  return (
    <div className={`card p-4 flex flex-col ${p.prediction ? "ring-1 ring-brand/30" : ""}`}>
      <div className="label">{p.group}</div>
      <div className="font-semibold text-ink leading-tight text-sm min-h-[34px]">{p.title}</div>
      <div className="mt-1 flex justify-center">
        <Gauge value={p.probability} threshold={p.threshold} positive="active" />
      </div>
      <div className={`text-center text-sm font-bold ${p.prediction ? "text-brand" : "text-ink-soft"}`}>
        {p.prediction_label}
      </div>
      <div className="mt-2 border-t border-line pt-2 text-[11px] text-ink-mute flex justify-between">
        <span>test ROC-AUC</span>
        <b className="text-ink">{fmt(p.test_metrics?.["ROC-AUC"])}</b>
      </div>
    </div>
  );
}

function Explanation({ p }) {
  const e = p?.explanation;
  if (!e) return <p className="text-sm text-ink-mute">No explanation available.</p>;
  const data = e.items.map((d) => ({
    name: d.kind === "substructure" ? `${d.feature} ◆` : d.feature,
    value: +d.contribution.toFixed(3),
  }));
  return (
    <div>
      <p className="text-xs text-ink-soft mb-2">
        SHAP contributions (log-odds):{" "}
        <span style={{ color: POS }} className="font-semibold">→ active</span> ·{" "}
        <span style={{ color: NEG }} className="font-semibold">← inactive</span>
        {e.items.some((d) => d.kind === "substructure") && " · ◆ substructure from the fingerprint"}
      </p>
      <ResponsiveContainer width="100%" height={36 * data.length + 30}>
        <BarChart data={data} layout="vertical" margin={{ left: 10, right: 30 }}>
          <CartesianGrid {...GRID} horizontal={false} />
          <XAxis type="number" {...AXIS} />
          <YAxis type="category" dataKey="name" width={230} {...AXIS} />
          <ReferenceLine x={0} stroke="#1B1F3B" />
          <Tooltip content={<ChartTooltip />} cursor={{ fill: "#F1F5F9" }} />
          <Bar isAnimationActive={false} dataKey="value" name="contribution" radius={4} barSize={18}>
            {data.map((d) => (
              <Cell key={d.name} fill={d.value >= 0 ? POS : NEG} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function MoleculeResults() {
  const [params] = useSearchParams();
  const ctx = useMolecule();
  const smiles = params.get("smiles") || ctx.smiles;
  const [rdkit, setRdkit] = useState(null);
  const [why, setWhy] = useState(null);

  useEffect(() => {
    loadRDKit().then(setRdkit).catch(() => {});
  }, []);

  const pred = useQuery({
    queryKey: ["mol-predict", smiles],
    queryFn: () => api.moleculePredict(smiles),
    enabled: !!smiles,
    retry: 0,
  });

  const svg = useMemo(() => depict(rdkit, smiles, { width: 300, height: 200 }), [rdkit, smiles]);
  if (!smiles) {
    return (
      <p>
        Draw a molecule in the{" "}
        <Link className="text-brand underline" to="/molecules">Molecule Studio</Link> first.
      </p>
    );
  }
  if (pred.isLoading) return <Loading text="Running the toxicity models…" />;
  if (pred.error) return <ErrorBox error={pred.error} />;
  const r = pred.data;
  const tox = r.predictions.filter((p) => p.endpoint !== "FDA_APPROVED");
  const approved = r.predictions.find((p) => p.endpoint === "FDA_APPROVED");
  const whyEp = tox.find((p) => p.endpoint === why) || tox[0];
  const { cls, Icon } = RISK[r.risk.level] || RISK.Low;
  // the endpoint codes are short and unique; the assay titles are not ("Androgen receptor"
  // appears twice), and duplicate axis labels make the radar collapse
  const radar = tox.map((p) => ({
    name: p.endpoint,
    value: +(100 * p.probability).toFixed(1),
  }));

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <Link to="/molecules" className="text-sm text-ink-mute hover:text-ink inline-flex items-center gap-1">
            <ArrowLeft size={14} /> Molecule Studio
          </Link>
          <h1 className="font-display text-3xl font-bold mt-1 font-mono break-all">{r.formula}</h1>
          <p className="text-ink-soft text-sm font-mono break-all">{r.smiles}</p>
        </div>
        <button
          className="btn-ghost text-sm"
          onClick={() => download(`${r.formula}_toxicity.json`, JSON.stringify(r, null, 2), "application/json")}
        >
          <Download size={16} /> JSON
        </button>
      </div>

      <div className={`flex items-start gap-2 rounded-xl px-4 py-3 text-sm ${cls}`}>
        <Icon size={18} className="mt-0.5 shrink-0" />
        <span>
          <b>{r.risk.level} predicted toxicity risk.</b> {r.risk.summary}
          {r.risk.flagged.length > 0 && `: ${r.risk.flagged.join(", ")}`}.
          {approved && ` Clinical approval model: ${approved.prediction_label.toLowerCase()} (${(100 * approved.probability).toFixed(0)}%).`}
          <span className="block mt-1 opacity-80">
            In-silico screening from the chemical structure only — for research and teaching, not a
            safety assessment.
          </span>
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)] gap-6">
        <div className="card p-5">
          <div className="label mb-2">Structure</div>
          {/* eslint-disable-next-line react/no-danger */}
          {svg ? <div className="grid place-items-center" dangerouslySetInnerHTML={{ __html: svg }} /> : null}
          <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs border-t border-line pt-3">
            {[
              ["Weight", `${fmt(r.descriptors.molecular_weight, 1)} u`],
              ["logP", fmt(r.descriptors.logP, 2)],
              ["TPSA", `${fmt(r.descriptors.TPSA, 1)} Å²`],
              ["QED", fmt(r.descriptors.QED, 2)],
              ["H-bond donors", r.descriptors.h_bond_donors],
              ["H-bond acceptors", r.descriptors.h_bond_acceptors],
            ].map(([k, v]) => (
              <Fragment key={k}>
                <span className="text-ink-mute">{k}</span>
                <span className="text-right text-ink">{v}</span>
              </Fragment>
            ))}
          </div>
        </div>

        <div className="card p-5">
          <SectionHeader
            title="Toxicity profile"
            subtitle="Predicted probability of being active in each assay (0–100%)."
          />
          <ResponsiveContainer width="100%" height={330}>
            <RadarChart data={radar} outerRadius="72%">
              <PolarGrid stroke="#E2E8F0" />
              <PolarAngleAxis dataKey="name" tick={{ fill: "#1B1F3B", fontSize: 10 }} />
              <PolarRadiusAxis domain={[0, 100]} tick={{ fill: "#94A3B8", fontSize: 9 }}
                tickCount={5} angle={90} />
              <Radar isAnimationActive={false} dataKey="value" name="probability (%)"
                stroke="#4f46e5" fill="#4f46e5" fillOpacity={0.3} strokeWidth={2}
                dot={{ r: 3 }} />
              <Tooltip content={<ChartTooltip digits={1} />} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-3">
        {tox.map((p) => <EndpointCard key={p.endpoint} p={p} />)}
      </div>

      <div className="card p-5">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <div>
            <h2 className="font-display text-xl font-bold">Why this prediction?</h2>
            <p className="text-xs text-ink-soft">
              Exact SHAP values of the LightGBM member of the stacked ensemble.
            </p>
          </div>
          <Segmented size="md" value={whyEp?.endpoint} onChange={setWhy}
            options={tox.map((p) => ({ value: p.endpoint, label: p.endpoint }))} />
        </div>
        <Explanation p={whyEp} />
      </div>
    </div>
  );
}
