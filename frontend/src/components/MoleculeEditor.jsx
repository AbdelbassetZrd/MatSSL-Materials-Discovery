import "../lib/ketcher-shim.js"; // must run before Ketcher is evaluated
import { useEffect, useRef, useState } from "react";
import { Editor } from "ketcher-react";
// the default entry hides its worker behind a dynamic require, so Vite never emits it;
// the binaryWasm entry builds the worker URL with `new URL(..., import.meta.url)`, which
// Vite understands and bundles together with the Indigo WebAssembly module
import { StandaloneStructServiceProvider } from "ketcher-standalone/dist/binaryWasm";
import { Download, Upload } from "lucide-react";
import "ketcher-react/dist/index.css";

// one shared Indigo instance (WASM) for the whole session, built on first use so that a
// failure to start it cannot take the rest of the application down with it
let provider = null;
const getProvider = () => (provider ??= new StandaloneStructServiceProvider());

/**
 * Ketcher drawing canvas.
 *
 * The canvas is the source of truth for what is drawn: "Use what I drew" reads it back as a
 * SMILES string. Pushing a structure *into* the canvas is explicit rather than automatic,
 * because Ketcher accepts structures only once its WebAssembly backend is fully up, and a
 * silent background write would otherwise wipe what the user has.
 */
export default function MoleculeEditor({ smiles, onChange, height = 440 }) {
  const ketcher = useRef(null);
  const [ready, setReady] = useState(false);
  const [slow, setSlow] = useState(false);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState(null);

  // the Indigo WebAssembly module is ~8 MB; say so instead of showing an empty frame
  useEffect(() => {
    if (ready) return undefined;
    const id = setTimeout(() => setSlow(true), 6000);
    return () => clearTimeout(id);
  }, [ready]);

  const read = async () => {
    setNote(null);
    try {
      const s = (await ketcher.current?.getSmiles()) ?? "";
      if (!s.trim()) return setNote("The canvas is empty — draw something first.");
      return onChange(s);
    } catch {
      return setNote("Could not read the canvas yet; give the editor a moment.");
    }
  };

  const load = async () => {
    setNote(null);
    setBusy(true);
    for (let attempt = 0; attempt < 20; attempt++) {
      try {
        await ketcher.current.setMolecule(smiles || "");
        setBusy(false);
        return;
      } catch {
        await new Promise((r) => setTimeout(r, 400));
      }
    }
    setBusy(false);
    setNote("The editor is still starting — try again in a few seconds.");
  };

  return (
    <div className="space-y-2">
      <div className="relative rounded-xl overflow-hidden border border-line bg-white"
           style={{ height }}>
        {!ready && (
          <div className="absolute inset-0 z-10 grid place-items-center bg-white/90 px-6 text-center text-sm text-ink-mute">
            {slow
              ? "Still starting: the editor loads an 8 MB chemistry engine. The builder and the SMILES box below work already."
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
      <div className="flex flex-wrap items-center gap-2">
        <button className="btn-primary text-sm" onClick={read} disabled={!ready}>
          <Download size={15} /> Use what I drew
        </button>
        <button className="btn-ghost text-sm" onClick={load} disabled={!ready || busy || !smiles}>
          <Upload size={15} /> {busy ? "Loading…" : "Load current molecule"}
        </button>
        {note && <span className="text-xs text-amber-700">{note}</span>}
      </div>
    </div>
  );
}
