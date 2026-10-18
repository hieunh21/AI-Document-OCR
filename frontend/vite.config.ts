import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies API calls to FastAPI (python -m uvicorn app.main:app --app-dir backend)
const API_TARGET = process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": API_TARGET,
      "/health": API_TARGET,
    },
  },
});
