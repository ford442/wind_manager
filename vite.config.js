import { defineConfig } from 'vite';

export default defineConfig({
  // Default is fine for this project.
  // Shaders are imported with ?raw so they are inlined as strings.
  server: {
    open: true, // optional: auto-open browser on npm run dev
  },
  build: {
    target: 'es2022', // WebGPU is modern
  },
});