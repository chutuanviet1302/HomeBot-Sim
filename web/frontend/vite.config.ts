import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run build:static` (mode "static") reads public/data/*.json instead of the API and writes dist-static/.
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  base: "./",
  define: { __STATIC__: JSON.stringify(mode === "static") },
  build: { outDir: mode === "static" ? "dist-static" : "dist", emptyOutDir: true },
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000", "/media": "http://127.0.0.1:8000" },
  },
}));
