import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development, API calls to /materials are proxied to the FastAPI backend.
export default defineConfig({
  base: "/ui/", // served by FastAPI at /ui/ in production
  plugins: [react()],
  // Ketcher's bundled dependencies still reference Node globals; without these the chemical
  // editor dies with "global is not defined" and never draws its canvas.
  define: { global: "globalThis", "process.env": {} },
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      ["/materials", "/molecules", "/design", "/health"].map((p) => [p, "http://localhost:8000"]),
    ),
  },
});
