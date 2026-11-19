import { useEffect, useState } from "react";
import { api } from "./api.js";
import { parseFormula, toFormula } from "./formula.js";
import Home from "./components/Home.jsx";
import PeriodicTable from "./components/PeriodicTable.jsx";
import FormulaInput from "./components/FormulaInput.jsx";
import CrystalViewer from "./components/CrystalViewer.jsx";
import AnalysisPanel from "./components/AnalysisPanel.jsx";
import ResultsPanel from "./components/ResultsPanel.jsx";

const STEPS = [
  { id: "home", label: "Start" },
  { id: "editor", label: "Composition editor" },
  { id: "input", label: "Formula input" },
  { id: "viewer", label: "3D crystal" },
  { id: "analyze", label: "Analyze" },
  { id: "results", label: "Results" },
];

export default function App() {
  const [step, setStep] = useState("home");
  const [counts, setCounts] = useState({ Li: 1, Fe: 1, P: 1, O: 4 });
  const [formulaText, setFormulaText] = useState("LiFePO4");
  const [formulaError, setFormulaError] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [results, setResults] = useState(null);
  const [models, setModels] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.modelsInfo().then((r) => setModels(r.models)).catch(() => setModels([]));
    // deep links, e.g. ?step=results&formula=Fe2O3 (used for demos and report screenshots)
    const q = new URLSearchParams(window.location.search);
    const f = q.get("formula");
    if (f) updateText(f);
    const s = q.get("step");
    if (s === "analyze" && f) runAnalysis(f);
    else if (s === "results" && f) runPrediction(f);
    else if (s) setStep(s);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // two-way sync: editor <-> formula text
  const updateCounts = (next) => {
    setCounts(next);
    setFormulaText(toFormula(next));
    setFormulaError(null);
  };
  const updateText = (text) => {
    setFormulaText(text);
    try {
      const parsed = parseFormula(text);
      setCounts(parsed);
      setFormulaError(null);
    } catch (e) {
      setFormulaError(e.message);
    }
  };

  const formula = toFormula(counts);

  async function runAnalysis(f = formula) {
    setBusy(true);
    setError(null);
    try {
      setAnalysis(await api.analyze(f));
      setStep("analyze");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function runPrediction(f = formula) {
    setBusy(true);
    setError(null);
    try {
      const [a, r] = await Promise.all([api.analyze(f), api.predict(f)]);
      setAnalysis(a);
      setResults(r);
      setStep("results");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand" onClick={() => setStep("home")}>
          <span className="logo">⬢</span> MatSSL
          <span className="tag">Semi-supervised materials discovery</span>
        </div>
        <nav className="steps">
          {STEPS.map((s, i) => (
            <button key={s.id} className={step === s.id ? "active" : ""} onClick={() => setStep(s.id)}>
              <span className="num">{i + 1}</span>
              {s.label}
            </button>
          ))}
        </nav>
      </header>

      {step !== "home" && (
        <div className="formula-bar">
          <span className="label">Current composition</span>
          <code className="formula">{formula || "—"}</code>
          <div className="spacer" />
          <button className="secondary" disabled={!formula || busy} onClick={() => runAnalysis()}>
            Analyze material
          </button>
          <button className="primary" disabled={!formula || busy} onClick={() => runPrediction()}>
            {busy ? "Running models…" : "Predict properties"}
          </button>
        </div>
      )}
      {error && <div className="error-banner">{error}</div>}

      <main>
        {step === "home" && <Home models={models} onStart={() => setStep("editor")} />}
        {step === "editor" && <PeriodicTable counts={counts} onChange={updateCounts} />}
        {step === "input" && (
          <FormulaInput text={formulaText} error={formulaError} counts={counts} onChange={updateText} />
        )}
        {step === "viewer" && <CrystalViewer formula={formula} counts={counts} />}
        {step === "analyze" && <AnalysisPanel analysis={analysis} onRun={() => runAnalysis()} busy={busy} />}
        {step === "results" && <ResultsPanel results={results} models={models} onRun={() => runPrediction()} busy={busy} />}
      </main>
      <footer>
        ESI-SBA · Machine Learning mini-project · Laref Ibrahim · Nemouchi Faycal Abdelhadi · Zerradi
        Abdelbasset · Medelci Djezzar Aymen
      </footer>
    </div>
  );
}
