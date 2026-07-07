import { resolve } from 'node:path';
import { defineConfig } from 'vite';

/** Bundle shared scene code into public/ for the legacy playground. */
export default defineConfig({
  build: {
    target: 'es2022',
    lib: {
      entry: resolve(__dirname, 'src/render/sceneShared.ts'),
      formats: ['es'],
      fileName: () => 'scene-shared.js',
    },
    outDir: 'public',
    emptyOutDir: false,
  },
});
