export default function Home({ models, onStart }) {
  return (
    <section className="home">
      <div className="hero">
        <h1>Discover materials with less labeled data</h1>
        <p>
          Enter any chemical composition and four models predict whether it is a <b>metal</b>, whether
          it is <b>magnetic</b>, whether it is <b>thermodynamically stable</b>, and whether its band gap
          falls in the <b>photovoltaic window</b>. The models are trained with semi-supervised
          learning: a few thousand labeled materials plus hundreds of thousands of unlabeled ones from
          the GNoME database.
        </p>
        <button className="primary big" onClick={onStart}>
          Start: build a composition →
        </button>
      </div>
      <div className="cards">
        <div className="card">
          <h3>1. Build</h3>
          <p>Click elements on the periodic table or type a formula such as LiFePO₄ or Ca₃(PO₄)₂.</p>
        </div>
        <div className="card">
          <h3>2. Inspect</h3>
          <p>See the lowest-energy crystal structure in 3D (Alexandria database) and the composition descriptors.</p>
        </div>
        <div className="card">
          <h3>3. Predict</h3>
          <p>Get probabilities from the best model of each task, with an applicability-domain check.</p>
        </div>
      </div>
      <h2>Deployed models</h2>
      <table className="models">
        <thead>
          <tr>
            <th>Task</th>
            <th>Data source</th>
            <th>Selected method</th>
            <th>Test ROC-AUC</th>
            <th>Test F1</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {models.length === 0 && (
            <tr>
              <td colSpan={6} className="muted">
                API not reachable – start the backend with <code>uv run uvicorn app.main:app</code>
              </td>
            </tr>
          )}
          {models.map((m) => (
            <tr key={m.task}>
              <td>{m.title}</td>
              <td>{m.source}</td>
              <td>{m.method || "—"}</td>
              <td>{m.test_metrics?.["ROC-AUC"]?.toFixed(3) ?? "—"}</td>
              <td>{m.test_metrics?.["F1-Score"]?.toFixed(3) ?? "—"}</td>
              <td>
                <span className={m.status === "Loaded" ? "pill ok" : "pill"}>{m.status}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
