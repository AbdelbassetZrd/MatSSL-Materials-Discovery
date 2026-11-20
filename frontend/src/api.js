const BASE = import.meta.env.VITE_API_URL || "";

async function call(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
  return body;
}

const post = (path, body) => call(path, { method: "POST", body: JSON.stringify(body) });
const q = (f) => encodeURIComponent(f);

export const api = {
  modelsInfo: () => call("/materials/models-info"),
  predict: (formula) => post("/materials/predict", { formula }),
  analyze: (formula) => post("/materials/analyze", { formula }),
  similar: (formula, k = 8) => call(`/materials/similar?formula=${q(formula)}&k=${k}`),
  structure: (formula) => call(`/materials/structure?formula=${q(formula)}`),
  // molecules
  moleculeModels: () => call("/molecules/models-info"),
  moleculePredict: (smiles) => post("/molecules/predict", { smiles }),
  moleculeProperties: (smiles) => post("/molecules/properties", { smiles }),
  // design lab
  designApplications: () => call("/design/applications"),
  designScore: (formula) => post("/design/score", { formula }),
  designCandidates: (application, limit = 30, elements = "", exclude = "") =>
    call(`/design/candidates?application=${q(application)}&limit=${limit}`
      + (elements.trim() ? `&elements=${q(elements.trim())}` : "")
      + (exclude.trim() ? `&exclude=${q(exclude.trim())}` : "")),
  // Research Lab data: API first, then the static copy bundled with the app (offline demo)
  experiments: async () => {
    try {
      return await call("/materials/experiments");
    } catch {
      const r = await fetch(`${import.meta.env.BASE_URL}data/dashboard.json`);
      if (!r.ok) throw new Error("dashboard data not available");
      return r.json();
    }
  },
};
