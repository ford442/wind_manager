# wind_manager

An interactive simulation of **evaporative downdraft** — the real effect behind the garden-hose observation where a fine water spray evaporatively cools the air it rises through, that cooled air becomes negatively buoyant and sinks, and the sinking column spreads along the floor as a radial outflow gust (a backyard dry microburst).

## WebGPU simulation (Issue #1)

Open `index.html` via a local HTTP server (ES modules + shader fetch require it):

```bash
npm start
# then visit http://localhost:8080
```

Or: `python3 -m http.server 8080`

### Features

- Two-way coupled **air / droplet / temperature** solver on GPU (WebGPU)
- Vertical spray emitter with d²-law evaporation, virtual-temperature buoyancy, semi-implicit droplet drag
- **Latent heat toggle** — ON → downdraft + floor outflow; OFF → momentum-only updraft (proof-of-mechanism)
- Field overlays: velocity, temperature, humidity; plus droplet points and velocity arrows
- Stable-fluids pressure projection (collocated grid + Jacobi)

### Module layout

```
sim/          params, fields, droplets, step orchestration
shaders/      WGSL compute + render passes
render/       overlay renderer
ui/           controls
main.js       WebGPU init + frame loop
index.html    main app
prototype.html  legacy 2D canvas playground
```

## Legacy prototype

`prototype.html` is the earlier 2D canvas playground (grass, trees, draggable water/wind emitters). No build step.

## Roadmap

- Issue #2: fans, vent hoses, multiple emitters
- Issue #3: collection surfaces, wettability, radiative cooling
- Issue #4: day/night cycle, ambient presets

Let's make the invisible visible and play with the forces that move the world.
