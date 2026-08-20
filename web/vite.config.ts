import { defineConfig } from "vite";

// base is "./" so the built bundle works from any path (GitHub Pages
// project sites, file://, a sub-route of a portfolio site).
export default defineConfig({
  base: "./",
  build: { target: "es2022", outDir: "dist" },
});
