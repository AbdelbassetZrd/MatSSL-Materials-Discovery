const EXAMPLES = [
  ["LiFePO4", "battery cathode"],
  ["GaAs", "semiconductor"],
  ["Fe2O3", "hematite"],
  ["CsPbI3", "perovskite absorber"],
  ["Nd2Fe14B", "permanent magnet"],
  ["Cu", "metal"],
  ["SrTiO3", "perovskite oxide"],
  ["Ca3(PO4)2", "phosphate"],
];

export default function FormulaInput({ text, error, counts, onChange }) {
  const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
  return (
    <section className="input-page">
      <h2>Formula input</h2>
      <p className="muted">Type a chemical formula. It is kept in sync with the composition editor.</p>
      <input
        className={`formula-input ${error ? "invalid" : ""}`}
        value={text}
        spellCheck={false}
        onChange={(e) => onChange(e.target.value)}
        placeholder="e.g. LiFePO4"
      />
      {error ? <p className="err">Invalid formula: {error}</p> : <p className="ok">✓ Valid formula</p>}
      <div className="examples">
        {EXAMPLES.map(([f, d]) => (
          <button key={f} className="chip" onClick={() => onChange(f)}>
            <b>{f}</b> <span>{d}</span>
          </button>
        ))}
      </div>
      <h3>Atomic fractions</h3>
      <div className="fractions">
        {Object.entries(counts).map(([el, n]) => (
          <div key={el} className="frac">
            <div className="frac-bar" style={{ height: `${(160 * n) / total}px` }} />
            <b>{el}</b>
            <span>{((100 * n) / total).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </section>
  );
}
