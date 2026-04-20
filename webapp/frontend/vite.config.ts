import react from "@vitejs/plugin-react";
import path from "path";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8001",
      "/u": "http://127.0.0.1:8001",
      "/o": "http://127.0.0.1:8001",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
