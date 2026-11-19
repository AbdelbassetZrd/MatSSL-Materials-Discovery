import elements from "../data/elements.json";

const CAT_LABEL = {
  alkali: "Alkali metal",
  alkaline: "Alkaline earth",
  transition: "Transition metal",
  "post-transition": "Post-transition metal",
  metalloid: "Metalloid",
  nonmetal: "Non-metal",
  halogen: "Halogen",
  noble: "Noble gas",
  lanthanide: "Lanthanide",
  actinide: "Actinide",
};

export default function PeriodicTable({ counts, onChange }) {
  const toggle = (sym) => {
    const next = { ...counts };
    if (next[sym]) delete next[sym];
    else next[sym] = 1;
    onChange(next);
  };
  const setAmount = (sym, v) => {
    const n = parseFloat(v);
    const next = { ...counts };
    if (!n || n <= 0) delete next[sym];
    else next[sym] = n;
    onChange(next);
  };
  const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;

  return (
    <section className="editor">
      <h2>Composition editor</h2>
      <p className="muted">Click an element to add or remove it, then set its amount on the right.</p>
      <div className="editor-grid">
        <div className="ptable">
          {elements.map((e) => (
            <button
              key={e.z}
              className={`el cat-${e.cat} ${counts[e.symbol] ? "selected" : ""}`}
              style={{ gridRow: e.row, gridColumn: e.group }}
              title={`${e.name} (Z=${e.z}) – ${CAT_LABEL[e.cat]}${e.en ? `, χ=${e.en}` : ""}`}
              onClick={() => toggle(e.symbol)}
            >
              <span className="z">{e.z}</span>
              <span className="sym">{e.symbol}</span>
            </button>
          ))}
          <div className="gap" style={{ gridRow: 8, gridColumn: "1 / 19" }} />
        </div>
        <div className="amounts">
          <h3>Amounts</h3>
          {Object.keys(counts).length === 0 && <p className="muted">No element selected.</p>}
          {Object.entries(counts).map(([sym, n]) => (
            <div className="amount-row" key={sym}>
              <span className="sym-badge">{sym}</span>
              <input type="number" min="0" step="0.5" value={n} onChange={(e) => setAmount(sym, e.target.value)} />
              <div className="bar">
                <div style={{ width: `${(100 * n) / total}%` }} />
              </div>
              <span className="pct">{((100 * n) / total).toFixed(1)}%</span>
              <button className="x" onClick={() => toggle(sym)}>×</button>
            </div>
          ))}
          <button className="secondary small" onClick={() => onChange({})}>
            Clear
          </button>
          <div className="legend">
            {Object.entries(CAT_LABEL).map(([k, v]) => (
              <span key={k}>
                <i className={`dot cat-${k}`} />
                {v}
              </span>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
