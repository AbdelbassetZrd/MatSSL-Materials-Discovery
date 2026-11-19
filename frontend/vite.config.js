import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development, API calls to /materials are proxied to the FastAPI backend.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/materials": "http://localhost:8000", "/health": "http://localhost:8000" },
  },
});
