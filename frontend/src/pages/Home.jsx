import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Boxes, BrainCircuit, Database, Network, Rocket, Sparkles } from "lucide-react";
import { api } from "../api.js";
import { Stat } from "../components/ui.jsx";
import { CrystalLogo } from "../components/Layout.jsx";
import { TASK, big, fmt } from "../lib/theme.js";

const PIPELINE = [
  { icon: Database, title: "4 open databases", text: "Materials Project, JARVIS-DFT, Alexandria, GNoME" },
  { icon: Boxes, title: "142 descriptors", text: "Magpie statistics of 22 elemental properties" },
  { icon: Network, title: "Learning with few labels", text: "5 SSL methods · association rules · self-supervised AE" },
  { icon: BrainCircuit, title: "Model selection", text: "3 seeds · composite = (F1 + ROC-AUC)/2" },
  { icon: Rocket, title: "Deployed", text: "FastAPI + React, one Docker image" },
];

export default function Home() {
  const nav = useNavigate();
  const { data } = useQuery({ queryKey: ["experiments"], queryFn: api.experiments });
  const k = data?.kpis;
  const deployed = data?.deployed || {};
  return (
    <div className="space-y-8">
      <section className="relative overflow-hidden rounded-3xl bg-white border border-line shadow-card px-8 py-10 md:py-14">
        <div className="absolute -right-10 -top-10 opacity-[0.08]"><CrystalLogo size={320} /></div>
        <span className="chip !border-brand/30 text-brand"><Sparkles size={14} /> ESI-SBA · Machine Learning mini-project</span>
        <h1 className="mt-4 font-display text-4xl md:text-5xl font-bold leading-tight max-w-3xl">
          Discovering materials when labels are scarce
        </h1>
        <p className="mt-4 max-w-2xl text-lg text-ink-soft">
          Every material label costs a DFT calculation. We predict <b>metallicity</b>, <b>magnetism</b>,{" "}
          <b>thermodynamic stability</b> and the <b>photovoltaic band gap</b> of any composition with semi-supervised,
          unsupervised and self-supervised learning on more than a million compositions.
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          <button className="btn-primary text-base px-5 py-3" onClick={() => nav("/studio")}>Build a composition <ArrowRight size={18} /></button>
          <button className="btn-ghost text-base px-5 py-3" onClick={() => nav("/lab")}>Explore the study</button>
        </div>
      </section>

      <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Stat value={k ? `${(k.compositions / 1e6).toFixed(2)}M` : "–"} label="compositions" sub={k && `${big(k.labeled)} labeled · ${big(k.unlabeled)} unlabeled`} />
        <Stat value={k ? k.databases : "–"} label="open databases" sub="no API key required" tone="cyan" />
        <Stat value={k ? big(k.models_trained) : "–"} label="models trained" sub="incl. Optuna trials, 3 seeds" tone="amber" />
        <Stat value="1,000" label="labels per experiment" sub="the rest is unlabeled" tone="good" />
      </section>

      <section className="card p-6">
        <h2 className="h-section mb-4">How it works</h2>
        <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
          {PIPELINE.map(({ icon: Icon, title, text }, i) => (
            <div key={title} className="relative rounded-2xl bg-surface p-4">
              <div className="flex items-center gap-2">
                <span className="grid h-8 w-8 place-items-center rounded-full bg-brand text-white text-sm font-bold">{i + 1}</span>
                <Icon size={18} className="text-brand" />
              </div>
              <div className="mt-2 font-semibold">{title}</div>
              <div className="text-sm text-ink-soft">{text}</div>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h2 className="h-section mb-4">Deployed models</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {Object.keys(TASK).map((t) => {
            const d = deployed[t] || (t === "band_gap" && data?.band_gap ? {
              method: "GA descriptors + Random Forest",
              test_metrics: data.band_gap.summary.test_ga_descriptors_tuned_threshold,
            } : null);
            return (
              <div key={t} className="card p-5">
                <div className="label">{TASK[t].source}</div>
                <div className="font-display text-xl font-bold mt-1">{TASK[t].title}</div>
                <div className="mt-2 text-sm text-ink-soft min-h-[40px]">{d?.method || "training in progress…"}</div>
                <div className="mt-3 flex gap-6">
                  <div><div className="label">ROC-AUC</div><div className="font-display text-2xl font-bold text-brand">{fmt(d?.test_metrics?.["ROC-AUC"])}</div></div>
                  <div><div className="label">F1</div><div className="font-display text-2xl font-bold text-cyan">{fmt(d?.test_metrics?.["F1-Score"])}</div></div>
                </div>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}
