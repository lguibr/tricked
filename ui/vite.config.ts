import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import path from 'path';

const backendHost = process.env.BACKEND_HOST || '127.0.0.1';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      '/api': `http://${backendHost}:8080`,
      '/ws': {
        target: `ws://${backendHost}:8080`,
        ws: true,
      },
    },
  },
});
