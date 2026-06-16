import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Dev proxy: the SPA talks to the FastAPI brain on :8090 (INTERFACES.md §11/§12).
// In production the same app is served from web/dist by FastAPI (single origin),
// so these paths resolve relatively — no proxy needed there.
const API_TARGET = process.env.VITE_API_TARGET || "http://localhost:8090";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: API_TARGET, changeOrigin: true },
      "/video": { target: API_TARGET, changeOrigin: true },
      "/events": { target: API_TARGET, changeOrigin: true, ws: true },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    chunkSizeWarningLimit: 1200,
  },
});
