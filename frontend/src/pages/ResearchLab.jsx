import { useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api.js";
import { ErrorBox, Loading, Tabs } from "../components/ui.jsx";
import Datasets from "../components/lab/Datasets.jsx";
import SSL from "../components/lab/SSL.jsx";
import Models from "../components/lab/Models.jsx";
import BandGap from "../components/lab/BandGap.jsx";
import Unsupervised from "../components/lab/Unsupervised.jsx";
import SelfSup from "../components/lab/SelfSup.jsx";

const TABS = [
  { id: "datasets", label: "Datasets" },
  { id: "ssl", label: "SSL experiments" },
  { id: "models", label: "Best models" },
  { id: "bandgap", label: "Band gap · GA & AD" },
  { id: "unsupervised", label: "Unsupervised" },
  { id: "selfsup", label: "Self-supervised" },
];

export default function ResearchLab() {
  const [params, setParams] = useSearchParams();
  const tab = params.get("tab") || "ssl";
  const { data, isLoading, error } = useQuery({ queryKey: ["experiments"], queryFn: api.experiments });
  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-3xl font-bold">Research Lab</h1>
        <p className="text-ink-soft">Every experiment behind the deployed models – live from the result files (works offline).</p>
      </div>
      <Tabs tabs={TABS} value={tab} onChange={(t) => setParams({ tab: t })} />
      {isLoading ? <Loading text="Loading experiment results…" /> : error ? <ErrorBox error={error} /> : (
        <>
          {tab === "datasets" && <Datasets data={data} />}
          {tab === "ssl" && <SSL data={data} />}
          {tab === "models" && <Models data={data} />}
          {tab === "bandgap" && <BandGap data={data} />}
          {tab === "unsupervised" && <Unsupervised data={data} />}
          {tab === "selfsup" && <SelfSup data={data} />}
        </>
      )}
    </div>
  );
}
