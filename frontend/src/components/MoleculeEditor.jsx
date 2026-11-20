import "../lib/ketcher-shim.js"; // must run before Ketcher is evaluated
import { useEffect, useRef, useState } from "react";
import { Editor } from "ketcher-react";
// the default entry hides its worker behind a dynamic require, so Vite never emits it;
// the binaryWasm entry builds the worker URL with `new URL(..., import.meta.url)`, which
// Vite understands and bundles together with the Indigo WebAssembly module
import { StandaloneStructServiceProvider } from "ketcher-standalone/dist/binaryWasm";
import "ketcher-react/dist/index.css";

// one shared Indigo instance (WASM) for the whole session, built on first use so that a
// failure to start it cannot take the rest of the application down with it
let provider = null;
const getProvider = () => (provider ??= new StandaloneStructServiceProvider());

/**
 * Ketcher drawing canvas, two-way synced with a SMILES string - the molecular counterpart of
 * the periodic table / formula box pair in the Composition Studio.
 */
export default function MoleculeEditor({ smiles, onChange, height = 440 }) {
  const ketcher = useRef(null);
  const applied = useRef("");      // last SMILES we pushed into the canvas
  const [ready, setReady] = useState(false);
  const [slow, setSlow] = useState(false);

  // the Indigo WebAssembly module is ~8 MB; say so instead of showing an empty frame
  useEffect(() => {
    if (ready) return undefined;
    const id = setTimeout(() => setSlow(true), 6000);
    return () => clearTimeout(id);
  }, [ready]);

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
    <div className="relative rounded-xl overflow-hidden border border-line bg-white"
         style={{ height }}>
      {!ready && (
        <div className="absolute inset-0 z-10 grid place-items-center bg-white/90 text-center text-sm text-ink-mute px-6">
          {slow
            ? "The drawing canvas is still starting (it loads an 8 MB chemistry engine). Use the builder or the SMILES box meanwhile."
            : "Starting the chemical editor…"}
        </div>
      )}
      <Editor
        staticResourcesUrl={import.meta.env.BASE_URL}
        structServiceProvider={getProvider()}
        errorHandler={() => {}}
        onInit={(k) => {
          ketcher.current = k;
          setReady(true);
        }}
      />
    </div>
  );
}
