import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";
export default defineConfig({ root: process.cwd(), base: "./", plugins: [react()],
  resolve: { alias: { "@shared": resolve(process.cwd(), "src/shared"), "@renderer": resolve(process.cwd(), "src/renderer/src") } }, build: {
  outDir: resolve(process.cwd(), ".runtime/device-allocation-ui"), emptyOutDir: true,
  rollupOptions: { input: resolve(process.cwd(), "tests/e2e/device-allocation-fixture.html") }
} });
