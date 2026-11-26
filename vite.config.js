import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { defineConfig } from 'vite';

const __dirname = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  server: {
    open: true,
  },
  build: {
    target: 'es2022',
    rollupOptions: {
      // The slice app and the yard volume are separate pages.
      input: {
        main: resolve(__dirname, 'index.html'),
        volume: resolve(__dirname, 'volume.html'),
      },
    },
  },
});
