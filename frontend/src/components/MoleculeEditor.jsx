import "../lib/ketcher-shim.js"; // must run before Ketcher is evaluated
import { useCallback, useEffect, useRef, useState } from "react";
import { Editor } from "ketcher-react";
// the default entry hides its worker behind a dynamic require, so Vite never emits it;
// the binaryWasm entry builds the worker URL with `new URL(..., import.meta.url)`, which
// Vite understands and bundles together with the Indigo WebAssembly module
import { StandaloneStructServiceProvider } from "ketcher-standalone/dist/binaryWasm";
// Ketcher's "3D" toolbar button renders through Miew and is inert unless Miew is handed to it.
// Indigo (already running here for everything else) supplies the 3D coordinates; Miew only draws
// them. It is a heavy module, but this whole component is lazy-loaded, so nothing is paid for it
// until the Molecule Studio page is opened.
import Miew from "miew";
import { Download, Upload } from "lucide-react";
import "ketcher-react/dist/index.css";
import "miew/dist/Miew.min.css";

// Ketcher looks for Miew on the window as well as on the prop, depending on how the viewer is
// opened, so it is registered in both places.
if (typeof window !== "undefined") window.Miew = Miew;

// one shared Indigo instance (WASM) for the whole session, built on first use so that a
// failure to start it cannot take the rest of the application down with it
let provider = null;
const getProvider = () => (provider ??= new StandaloneStructServiceProvider());

/**
 * Ketcher drawing canvas.
 *
 * The canvas and the SMILES box are kept in step in both directions. "Use what I drew" reads the
 * canvas back as SMILES; a SMILES arriving from anywhere else - typed, picked from a preset,
 * loaded from a link - is pushed into the canvas so it can be edited rather than retyped.
 *
 * The one thing that must not happen is an echo: writing a structure into the canvas must not
 * look like a fresh edit and bounce back out again, which would reformat the user's drawing on
 * every keystroke. `synced` records the SMILES the two sides last agreed on, and a write is
 * skipped whenever the incoming string already matches it.
 */
export default function MoleculeEditor({ smiles, onChange, height = 440 }) {
  const ketcher = useRef(null);
  const synced = useRef(null);
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
      synced.current = s; // what leaves the canvas must not be pushed straight back into it
      return onChange(s);
    } catch {
      return setNote("Could not read the canvas yet; give the editor a moment.");
    }
  };

  // Writing into Ketcher only works once its WebAssembly backend is up, and there is no event
  // for that beyond onInit, so a write is retried briefly rather than assumed to succeed.
  const push = useCallback(async (value, { announce } = {}) => {
    setBusy(true);
    for (let attempt = 0; attempt < 20; attempt++) {
      try {
        await ketcher.current.setMolecule(value || "");
        synced.current = value;
        setBusy(false);
        return true;
      } catch {
        await new Promise((r) => setTimeout(r, 400));
      }
    }
    setBusy(false);
    if (announce) setNote("The editor is still starting — try again in a few seconds.");
    return false;
  }, []);

  // Keep the canvas showing whatever molecule the page is working on. Skipped when the value is
  // already what the two sides agreed on, which is what stops "Use what I drew" from echoing.
  useEffect(() => {
    if (!ready) return;
    const value = smiles || "";
    if (value === synced.current) return;
    push(value);
  }, [ready, smiles, push]);

  const load = () => {
    setNote(null);
    synced.current = null; // force the write even if the canvas was drawn over
    push(smiles || "", { announce: true });
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
            // Ketcher's own demo publishes the instance this way; it costs nothing and makes the
            // editor inspectable from the browser console when something goes wrong here.
            if (typeof window !== "undefined") window.ketcher = k;
            setReady(true);
          }}
          // enables the 3D button in the toolbar; Indigo supplies the coordinates
          Miew={Miew}
          options={{ miewMode: "LN" }}
        />
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <button className="btn-primary text-sm" onClick={read} disabled={!ready}>
          <Download size={15} /> Use what I drew
        </button>
        <button className="btn-ghost text-sm" onClick={load} disabled={!ready || busy || !smiles}>
          <Upload size={15} /> {busy ? "Loading…" : "Reload current molecule"}
        </button>
        <span className="text-xs text-ink-mute">
          The SMILES box loads here automatically — edit the structure, then press “Use what I
          drew”. The <b>3D</b> toolbar button shows it in three dimensions.
        </span>
        {note && <span className="text-xs text-amber-700">{note}</span>}
      </div>
    </div>
  );
}
