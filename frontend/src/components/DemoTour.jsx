import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ChevronLeft, ChevronRight, X } from "lucide-react";

const STEPS = [
  { to: "/", title: "1.1 million compositions", text: "Four open databases (Materials Project, JARVIS-DFT, Alexandria, GNoME). The question: can unlabeled materials make our models better when labels are scarce?" },
  { to: "/studio?formula=LiFePO4", title: "Build any composition", text: "Click elements on the periodic table or type a formula. Each composition becomes 142 Magpie descriptors in milliseconds." },
  { to: "/results?formula=LiFePO4", title: "LiFePO₄ – battery cathode", text: "Four deployed models predict metallicity, magnetism, stability and the photovoltaic band gap, with an explanation and the nearest known materials." },
  { to: "/results?formula=CsPbI3", title: "CsPbI₃ – solar perovskite", text: "The band-gap model (genetic-algorithm descriptors) places CsPbI₃ in the photovoltaic window; the chemical-space map shows where it sits." },
  { to: "/results?formula=Nd2Fe14B", title: "Nd₂Fe₁₄B – permanent magnet", text: "The strongest commercial magnet: magnetic and metallic. The 'why' panel shows the descriptors that drive the decision." },
  { to: "/lab?tab=ssl", title: "The semi-supervised study", text: "Baseline vs five SSL methods, 3 seeds, unlabeled pools from 1k to 100k, two kinds of unlabeled data. Gains are small and external data can hurt." },
  { to: "/lab?tab=models", title: "Selected models", text: "The best architecture per task by composite score (mean of F1 and ROC-AUC), re-trained on all labels: test ROC curves and confusion matrices." },
  { to: "/lab?tab=unsupervised", title: "Unsupervised chemistry", text: "Apriori rules rediscover chemistry ({Mn, O} ⇒ magnetic, 95%) and K-Means finds material families without labels." },
  { to: "/compare?f=LiFePO4,LiMnPO4,LiCoO2", title: "Compare candidates", text: "Side-by-side profiles help choose which material to test next – the motivation for future active learning." },
];

export default function DemoTour({ onClose }) {
  const [i, setI] = useState(0);
  const nav = useNavigate();
  useEffect(() => {
    nav(STEPS[i].to);
  }, [i, nav]);
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "ArrowRight") setI((x) => Math.min(STEPS.length - 1, x + 1));
      if (e.key === "ArrowLeft") setI((x) => Math.max(0, x - 1));
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  const s = STEPS[i];
  return (
    <div className="fixed bottom-5 right-5 z-50 w-[380px] max-w-[calc(100vw-2rem)] card p-5 border-brand/30">
      <div className="flex items-center justify-between mb-2">
        <span className="label">Demo tour · {i + 1} / {STEPS.length}</span>
        <button onClick={onClose} aria-label="Close tour" className="text-ink-mute hover:text-ink"><X size={18} /></button>
      </div>
      <h3 className="font-display text-xl font-bold mb-1">{s.title}</h3>
      <p className="text-sm text-ink-soft leading-relaxed">{s.text}</p>
      <div className="mt-4 flex items-center gap-2">
        <button className="btn-ghost text-sm" disabled={i === 0} onClick={() => setI(i - 1)}><ChevronLeft size={16} /> Back</button>
        <div className="flex-1 flex justify-center gap-1">
          {STEPS.map((_, k) => <span key={k} className={`h-1.5 w-4 rounded-full ${k === i ? "bg-brand" : "bg-line"}`} />)}
        </div>
        {i < STEPS.length - 1
          ? <button className="btn-primary text-sm" onClick={() => setI(i + 1)}>Next <ChevronRight size={16} /></button>
          : <button className="btn-primary text-sm" onClick={onClose}>Finish</button>}
      </div>
      <p className="mt-2 text-[11px] text-ink-mute">Keyboard: ← → to navigate, Esc to close</p>
    </div>
  );
}
