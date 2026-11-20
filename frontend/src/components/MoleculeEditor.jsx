import { useEffect, useRef, useState } from "react";
import { Editor } from "ketcher-react";
import { StandaloneStructServiceProvider } from "ketcher-standalone";
import "ketcher-react/dist/index.css";

// one shared Indigo instance (WASM) for the whole session
const structServiceProvider = new StandaloneStructServiceProvider();

/**
 * Ketcher drawing canvas, two-way synced with a SMILES string - the molecular counterpart of
 * the periodic table / formula box pair in the Composition Studio.
 */
export default function MoleculeEditor({ smiles, onChange, height = 440 }) {
  const ketcher = useRef(null);
  const applied = useRef("");      // last SMILES we pushed into the canvas
  const [ready, setReady] = useState(false);

  // outside change (example library, URL, text box) -> canvas
  useEffect(() => {
    const k = ketcher.current;
    if (!ready || !k || smiles === applied.current) return;
    applied.current = smiles;
    (smiles ? k.setMolecule(smiles) : k.setMolecule("")).catch(() => {});
  }, [smiles, ready]);

  // canvas change -> SMILES, polled because Ketcher has no stable change event
  useEffect(() => {
    if (!ready) return undefined;
    const id = setInterval(async () => {
      try {
        const s = (await ketcher.current?.getSmiles()) ?? "";
        if (s !== applied.current) {
          applied.current = s;
          onChange(s);
        }
      } catch {
        /* the canvas is mid-edit */
      }
    }, 700);
    return () => clearInterval(id);
  }, [ready, onChange]);

  return (
    <div className="rounded-xl overflow-hidden border border-line bg-white" style={{ height }}>
      <Editor
        staticResourcesUrl={import.meta.env.BASE_URL}
        structServiceProvider={structServiceProvider}
        errorHandler={() => {}}
        onInit={(k) => {
          ketcher.current = k;
          setReady(true);
        }}
      />
    </div>
  );
}
