# wind_manager

An interactive visualization of wind and water streams interacting with a landscape — grass, trees, houses, and clouds.

## The Idea

Inspired by real-world experiments: spraying water from a garden hose in specific ways can generate surprisingly strong, localized gusts capable of visibly shaking tall trees over 100 feet away. Wind appears and disappears dramatically depending on the arrangement of air and water streams.

Goal: build a playground to explore different configurations of:
- Water jets / hoses (particles with momentum)
- Air streams / gust sources (force fields)
- How they couple and propagate to move flexible elements at a distance

## Vision

- **Scene**: Field of grass, several trees (different heights/distances), houses, layered clouds.
- **Wind**: Combination of global wind + multiple user-placeable localized streams. Visualize the flow.
- **Water**: Ejected droplets affected by gravity, drag, and the local wind field. Water can "seed" or amplify visible air movement.
- **Interaction**: Drag to reposition emitters, tweak strength/angle/spread. Experiment quickly with different arrangements and see emergent effects (e.g. upward water jet → high air disturbance → distant tree sway).
- **Future**: Presets, recording, more realistic turbulence, 3D view, sound, parameter studies.

## Getting Started (Prototype)

Open `index.html` in a browser. No build step.

Current state: basic interactive 2D canvas prototype.

## Roadmap ideas

- [ ] Multiple draggable water + wind emitters
- [ ] Better wind field visualization (streamlines / tracers)
- [ ] Variable tree response (height, foliage density, distance damping)
- [ ] Water splashes, ground interaction, evaporation
- [ ] Turbulence / Perlin-style noise wind
- [ ] UI controls panel (strength, angle, pulse mode)
- [ ] Export/share specific configurations
- [ ] Performance: thousands of grass blades + particles
- [ ] Optional: p5.js or Three.js upgrade path

Let's make the invisible visible and play with the forces that move the world.
