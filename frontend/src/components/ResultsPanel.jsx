const ICON = { metallicity: "⚡", magnetism: "🧲", stability: "⚖️", band_gap: "☀️" };

export default function ResultsPanel({ results, models, onRun, busy }) {
  if (!results)
    return (
      <section>
        <h2>Results</h2>
        <p className="muted">Run the models on the current composition.</p>
        <button className="primary" onClick={onRun} disabled={busy}>Predict properties</button>
      </section>
    );
  const ad = results.applicability_domain;
  const metric = (task, k) => models.find((m) => m.task === task)?.test_metrics?.[k];
  return (
    <section className="results">
      <h2>Predictions for <code>{results.formula}</code></h2>
      {ad && (
        <div className={`ad ${ad.inside ? "inside" : "outside"}`}>
          {ad.inside
            ? "✓ Inside the applicability domain: this composition is similar to the training materials."
            : "⚠ Outside the applicability domain: this composition is far from the training materials, so treat the predictions with caution."}
          <span className="muted"> (distance {ad.distance.toFixed(2)} / threshold {ad.threshold.toFixed(2)}, cluster {ad.cluster})</span>
        </div>
      )}
      <div className="result-grid">
        {results.predictions.map((p) => (
          <div key={p.task} className={`result-card ${p.prediction ? "pos" : "neg"}`}>
            <div className="rc-head">
              <span className="icon">{ICON[p.task]}</span>
              <h3>{p.title}</h3>
            </div>
            <div className="rc-label">{p.prediction_label}</div>
            <div className="prob">
              <div className="prob-bar"><div style={{ width: `${100 * p.probability}%` }} /></div>
              <span>{(100 * p.probability).toFixed(1)}%</span>
            </div>
            <p className="muted small">
              Confidence {(100 * p.confidence).toFixed(0)}% · {p.method || "model"}
              {metric(p.task, "ROC-AUC") != null && ` · test ROC-AUC ${metric(p.task, "ROC-AUC").toFixed(3)}`}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}
