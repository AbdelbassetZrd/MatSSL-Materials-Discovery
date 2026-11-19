// Minimal chemical-formula parser (supports parentheses, e.g. Ca3(PO4)2).
export function parseFormula(text) {
  const s = text.replace(/\s+/g, "");
  let i = 0;
  function group() {
    const counts = {};
    while (i < s.length && s[i] !== ")") {
      if (s[i] === "(") {
        i++;
        const inner = group();
        if (s[i] !== ")") throw new Error("missing )");
        i++;
        const n = number();
        for (const [el, c] of Object.entries(inner)) counts[el] = (counts[el] || 0) + c * n;
      } else {
        const m = /^[A-Z][a-z]?/.exec(s.slice(i));
        if (!m) throw new Error(`unexpected '${s[i]}'`);
        i += m[0].length;
        const n = number();
        counts[m[0]] = (counts[m[0]] || 0) + n;
      }
    }
    return counts;
  }
  function number() {
    const m = /^\d*\.?\d+/.exec(s.slice(i));
    if (!m) return 1;
    i += m[0].length;
    return parseFloat(m[0]);
  }
  if (!s) return {};
  const out = group();
  if (i !== s.length) throw new Error("unbalanced )");
  return out;
}

export function toFormula(counts) {
  return Object.entries(counts)
    .filter(([, n]) => n > 0)
    .map(([el, n]) => (n === 1 ? el : `${el}${+n.toFixed(3)}`))
    .join("");
}
