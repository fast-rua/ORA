import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `ora web setup` builds the page into dist/; the run's web server (server/main.ts) serves it.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist", emptyOutDir: true, assetsInlineLimit: 0, chunkSizeWarningLimit: 1024 },
  test: {
    include: ["test/**/*.test.{ts,tsx}"],
    environment: "node",
    restoreMocks: true,
  },
} as never);
