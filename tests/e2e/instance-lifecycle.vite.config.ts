import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";
export default defineConfig({ root: process.cwd(), base: "./", plugins: [react()], build: {
  outDir: resolve(process.cwd(), ".runtime/instance-lifecycle-ui"), emptyOutDir: true,
  rollupOptions: { input: resolve(process.cwd(), "tests/e2e/instance-lifecycle-fixture.html") }
} });
