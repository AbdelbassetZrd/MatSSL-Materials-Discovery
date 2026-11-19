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

export const api = {
  modelsInfo: () => call("/materials/models-info"),
  predict: (formula) => call("/materials/predict", { method: "POST", body: JSON.stringify({ formula }) }),
  analyze: (formula) => call("/materials/analyze", { method: "POST", body: JSON.stringify({ formula }) }),
  structure: (formula) => call(`/materials/structure?formula=${encodeURIComponent(formula)}`),
};
