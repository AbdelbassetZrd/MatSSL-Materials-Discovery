// RDKit.js ships a WebAssembly module that has to be served as a static file. It is copied
// out of node_modules on every build instead of being committed, so the binary always matches
// the installed @rdkit/rdkit version.
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const from = resolve(here, "..", "node_modules", "@rdkit", "rdkit", "dist");
const to = resolve(here, "..", "public", "rdkit");

mkdirSync(to, { recursive: true });
for (const f of ["RDKit_minimal.js", "RDKit_minimal.wasm"]) {
  copyFileSync(resolve(from, f), resolve(to, f));
}
console.log("copied RDKit.js to public/rdkit/");
