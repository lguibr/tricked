/// <reference types="vitest" />
import path from "path"
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// const host = process.env.TAURI_DEV_HOST;

// https://vitejs.dev/config/
export default defineConfig(async () => ({
  plugins: [react()],

  server: {
    port: 5173,
    strictPort: false,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/setupTests.ts'],
    exclude: ['node_modules', 'dist', '.idea', '.git', '.cache', 'e2e/**'],
  },
}));
