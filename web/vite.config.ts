import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const srcDir = new URL("./src", import.meta.url).pathname;

// Same-origin API endpoints served by Flask (src/web_app.py). In dev we proxy
// them to the running Flask server so `npm run dev` + Flask gives hot reload.
const API_PATHS = ["/state", "/gestures", "/voice", "/music", "/sound", "/camera.mjpg", "/face.mjpg"];

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": srcDir },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    proxy: Object.fromEntries(
      API_PATHS.map((p) => [p, { target: "http://127.0.0.1:8000", changeOrigin: true }]),
    ),
  },
});
