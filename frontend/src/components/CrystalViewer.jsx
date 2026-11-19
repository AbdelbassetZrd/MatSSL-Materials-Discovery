import { useEffect, useRef, useState } from "react";
import * as $3Dmol from "3dmol";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api.js";

const EDGES = [
  ["o", "a"], ["o", "b"], ["o", "c"], ["a", "ab"], ["a", "ac"], ["b", "ab"],
  ["b", "bc"], ["c", "ac"], ["c", "bc"], ["ab", "abc"], ["ac", "abc"], ["bc", "abc"],
];

function drawCell(viewer, L) {
  const [a, b, c] = L;
  const add = (p, q) => [p[0] + q[0], p[1] + q[1], p[2] + q[2]];
  const P = { o: [0, 0, 0], a, b, c, ab: add(a, b), ac: add(a, c), bc: add(b, c), abc: add(add(a, b), c) };
  EDGES.forEach(([p, q]) =>
    viewer.addCylinder({
      start: { x: P[p][0], y: P[p][1], z: P[p][2] },
      end: { x: P[q][0], y: P[q][1], z: P[q][2] },
      radius: 0.04,
      color: "#64748b",
    })
  );
}

export default function CrystalViewer({ formula, height = 360, spin = true }) {
  const box = useRef(null);
  const [err, setErr] = useState(null);
  const { data, isLoading, error } = useQuery({
    queryKey: ["structure", formula],
    queryFn: () => api.structure(formula),
    enabled: !!formula,
    retry: 0,
  });
  useEffect(() => {
    if (!data || !box.current) return;
    box.current.innerHTML = "";
    try {
      const v = $3Dmol.createViewer(box.current, { backgroundColor: "white" });
      v.addModel(data.xyz, "xyz");
      v.setStyle({}, { sphere: { scale: 0.32 }, stick: { radius: 0.12 } });
      drawCell(v, data.lattice_vectors);
      v.zoomTo();
      if (spin) v.spin("y", 0.4);
      v.render();
      setErr(null);
    } catch (e) {
      setErr(e.message);
    }
  }, [data, spin]);
  return (
    <div className="relative rounded-xl border border-line bg-white overflow-hidden" style={{ height }}>
      <div ref={box} className="absolute inset-0" />
      {(isLoading || error || err) && (
        <div className="absolute inset-0 grid place-items-center p-6 text-center text-sm text-ink-mute">
          {isLoading
            ? "Looking up the lowest-energy structure…"
            : `No known crystal structure for ${formula}. Predictions work from the composition alone.`}
        </div>
      )}
      {data && (
        <div className="absolute left-2 bottom-2 rounded-lg bg-white/90 px-2 py-1 text-[11px] text-ink-soft">
          Alexandria {data.id} · {data.xyz.split("\n")[0]} atoms · E<sub>hull</sub>{" "}
          {data.e_above_hull != null ? data.e_above_hull.toFixed(3) : "–"} eV/atom
        </div>
      )}
    </div>
  );
}
