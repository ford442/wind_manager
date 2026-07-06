# wind_manager

An interactive simulation of **evaporative downdraft** — the real effect behind the garden-hose observation where a fine water spray evaporatively cools the air it rises through, that cooled air becomes negatively buoyant and sinks, and the sinking column spreads along the floor as a radial outflow gust (a backyard dry microburst).

## WebGPU simulation (Issue #1)

## Development

This project uses Vite for fast development and bundling.

```bash
npm install
npm run dev
# then visit the URL shown (usually http://localhost:5173)
```

- `npm run build` — produces a production build in `dist/`
- `npm run preview` — preview the production build locally

Legacy standalone prototype is still at `/prototype.html` when the dev server is running.

### Features

- Two-way coupled **air / droplet / temperature** solver on GPU (WebGPU)
- Vertical spray emitter with d²-law evaporation, virtual-temperature buoyancy, semi-implicit droplet drag
- **Latent heat toggle** — ON → downdraft + floor outflow; OFF → momentum-only updraft (proof-of-mechanism)
- Field overlays: velocity, temperature, humidity; plus droplet points and velocity arrows
- Stable-fluids pressure projection (collocated grid + Jacobi)

### Module layout (src/)

```
src/
  main.ts     WebGPU init + frame loop (TypeScript)
  sim/        params (with Params interface), fields, droplets, step orchestration
  render/     overlay renderer
  ui/         controls
  shaders/    WGSL compute + render passes (imported via ?raw)
index.html    app shell (references /src/main.ts)
public/       static assets (prototype.html legacy)
```

The project is now a Vite + TypeScript setup. Run `npm run dev` to develop.

## Legacy prototype

`prototype.html` is the earlier 2D canvas playground (grass, trees, draggable water/wind emitters). No build step.

## Roadmap

See the full list of open issues: https://github.com/ford442/wind_manager/issues

Key upcoming work includes:
- Direct canvas interaction for the emitter (point the hose)
- Velocity-driven grass, swaying trees, houses/flags, and clouds
- Multiple emitters + pure air gust sources
- Droplet-ground interaction and surface effects
- Presets, advanced controls, performance, visualization improvements, and more

The original garden-hose downdraft observation is the north star.

Let's make the invisible visible and play with the forces that move the world.
