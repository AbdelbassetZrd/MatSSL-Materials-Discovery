const LABELS = {
  "MagpieData mean Electronegativity": "Mean electronegativity (Pauling)",
  "MagpieData range Electronegativity": "Electronegativity difference",
  "MagpieData mean CovalentRadius": "Mean covalent radius (pm)",
  "MagpieData mean NValence": "Mean valence electrons",
  "MagpieData mean GSbandgap": "Mean elemental band gap (eV)",
  "MagpieData mean GSmagmom": "Mean elemental magnetic moment (μB)",
  "MagpieData mean MeltingT": "Mean melting temperature (K)",
  "MagpieData mean AtomicWeight": "Mean atomic weight (u)",
  "0-norm": "Number of elements",
};

export default function AnalysisPanel({ analysis, onRun, busy }) {
  if (!analysis)
    return (
      <section>
        <h2>Analyze material</h2>
        <p className="muted">Compute the composition descriptors and chemistry checks.</p>
        <button className="primary" onClick={onRun} disabled={busy}>Analyze</button>
      </section>
    );
  const r = analysis.chemical_rules;
  return (
    <section className="analysis">
      <h2>Analysis of <code>{analysis.reduced_formula}</code></h2>
      <div className="analysis-grid">
        <div className="card">
          <h3>Chemistry checks</h3>
          <ul className="checks">
            <li className={r.charge_balanced ? "pass" : "fail"}>
              {r.charge_balanced ? "✓" : "✗"} Charge balance (common oxidation states)
              {r.oxidation_states && (
                <span className="muted">
                  {" "}
                  – {Object.entries(r.oxidation_states).map(([el, v]) => `${el}${v > 0 ? "+" : ""}${v}`).join(", ")}
                </span>
              )}
            </li>
            <li className={r.n_elements_ok ? "pass" : "fail"}>
              {r.n_elements_ok ? "✓" : "✗"} {analysis.n_elements} elements (≤ 6 inside training range)
            </li>
            <li className="pass">✓ Formula weight {analysis.weight.toFixed(2)} u</li>
          </ul>
        </div>
        <div className="card">
          <h3>Composition descriptors (Magpie)</h3>
          <table className="kv">
            <tbody>
              {Object.entries(analysis.descriptors).map(([k, v]) => (
                <tr key={k}>
                  <td>{LABELS[k] || k}</td>
                  <td className="num">{v == null ? "—" : v.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card wide">
          <h3>Association rules that apply (unsupervised Apriori mining)</h3>
          {analysis.association_rules_matched.length === 0 ? (
            <p className="muted">No mined rule matches this element set.</p>
          ) : (
            <table className="rules">
              <thead>
                <tr><th>Rule</th><th>Confidence</th><th>Lift</th></tr>
              </thead>
              <tbody>
                {analysis.association_rules_matched.map((m, i) => (
                  <tr key={i}>
                    <td><code>{m.rule}</code></td>
                    <td>{(100 * m.confidence).toFixed(1)}%</td>
                    <td>{m.lift.toFixed(2)}×</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </section>
  );
}
