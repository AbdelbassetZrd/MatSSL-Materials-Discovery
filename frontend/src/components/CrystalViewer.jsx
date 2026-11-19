import { useEffect, useRef, useState } from "react";
import * as $3Dmol from "3dmol";
import { api } from "../api.js";

function drawCell(viewer, L) {
  const [a, b, c] = L;
  const add = (p, q) => [p[0] + q[0], p[1] + q[1], p[2] + q[2]];
  const o = [0, 0, 0];
  const corners = { o, a, b, c, ab: add(a, b), ac: add(a, c), bc: add(b, c), abc: add(add(a, b), c) };
  const edges = [
    ["o", "a"], ["o", "b"], ["o", "c"], ["a", "ab"], ["a", "ac"], ["b", "ab"],
    ["b", "bc"], ["c", "ac"], ["c", "bc"], ["ab", "abc"], ["ac", "abc"], ["bc", "abc"],
  ];
  for (const [p, q] of edges) {
    const P = corners[p], Q = corners[q];
    viewer.addCylinder({
      start: { x: P[0], y: P[1], z: P[2] }, end: { x: Q[0], y: Q[1], z: Q[2] },
      radius: 0.04, color: "#64748b", fromCap: true, toCap: true,
    });
  }
}

export default function CrystalViewer({ formula, counts }) {
  const box = useRef(null);
  const [info, setInfo] = useState(null);
  const [status, setStatus] = useState("idle");

  useEffect(() => {
    if (!formula) return;
    let cancelled = false;
    setStatus("loading");
    api
      .structure(formula)
      .then((s) => {
        if (cancelled) return;
        setInfo(s);
        setStatus("ok");
        const viewer = $3Dmol.createViewer(box.current, { backgroundColor: "white" });
        viewer.addModel(s.xyz, "xyz");
        viewer.setStyle({}, { sphere: { scale: 0.32 }, stick: { radius: 0.12 } });
        drawCell(viewer, s.lattice_vectors);
        viewer.zoomTo();
        viewer.spin("y", 0.4);
        viewer.render();
      })
      .catch((e) => !cancelled && (setStatus(e.message), setInfo(null)));
    return () => {
      cancelled = true;
      if (box.current) box.current.innerHTML = "";
    };
  }, [formula]);

  const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
  return (
    <section className="viewer-page">
      <h2>3D crystal view</h2>
      <p className="muted">
        Lowest-energy known structure for <code>{formula}</code> from the Alexandria database (OPTIMADE API).
      </p>
      <div className="viewer-grid">
        <div className="viewer" ref={box}>
          {status === "loading" && <div className="overlay">Looking up structure…</div>}
          {status !== "ok" && status !== "loading" && status !== "idle" && (
            <div className="overlay">
              No known crystal structure for this composition ({status}). Predictions still work from the
              composition alone.
            </div>
          )}
        </div>
        <div className="viewer-side">
          {info && (
            <>
              <p><b>Database ID:</b> {info.id}</p>
              <p><b>Atoms in cell:</b> {info.xyz.split("\n")[0]}</p>
              <p><b>E above hull:</b> {info.e_above_hull != null ? `${info.e_above_hull.toFixed(3)} eV/atom` : "—"}</p>
            </>
          )}
          <h3>Composition</h3>
          {Object.entries(counts).map(([el, n]) => (
            <div className="amount-row" key={el}>
              <span className="sym-badge">{el}</span>
              <div className="bar"><div style={{ width: `${(100 * n) / total}%` }} /></div>
              <span className="pct">{((100 * n) / total).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
