/** Canonical backyard layout + palette — shared by physics sim and art playground. */

export const DOMAIN = { w: 32, h: 8 } as const;

export const SCENE_COLORS = {
  skyTop: '#0f172a',
  skyMid: '#1e3a5f',
  skyHorizon: '#334155',
  ground: '#166534',
  groundDark: '#14532d',
  grass: '#4ade80',
  grassStroke: '#4ade80',
  trunk: '#3f2a1f',
  foliage: '#166534',
  foliageMid: '#15803d',
  foliageHi: 'rgba(74, 222, 128, 0.16)',
  houseBody: '#334155',
  houseRoof: '#1e2937',
  cloud: 'rgba(226, 232, 240, 0.72)',
} as const;

/** Default wide yard: ~30 m propagation reach at 32 m width. */
export const LAYOUT = {
  emitterLeft: 0.032,
  emitterCenter: 0.5,
  emitterRight: 0.965,
  trees: [
    { xFrac: 0.06, height: 4.5, depth: 0.92, phase: 1.1, sampleYFrac: 0.62, trunkTau: 1.8, foliageTau: 0.55, windAmp: 1.05 },
    { xFrac: 0.19, height: 3.1, depth: 0.78, phase: 0.35, sampleYFrac: 0.58, trunkTau: 1.2, foliageTau: 0.42, windAmp: 0.88 },
    { xFrac: 0.36, height: 4.2, depth: 0.88, phase: 2.1, sampleYFrac: 0.64, trunkTau: 2.0, foliageTau: 0.58, windAmp: 1.15 },
    { xFrac: 0.56, height: 3.4, depth: 0.72, phase: 1.75, sampleYFrac: 0.56, trunkTau: 1.1, foliageTau: 0.45, windAmp: 0.95 },
    { xFrac: 0.91, height: 5.2, depth: 1.0, phase: 2.7, sampleYFrac: 0.68, trunkTau: 2.6, foliageTau: 0.72, windAmp: 1.55 },
  ],
  houses: [
    { xFrac: 0.28, wFrac: 0.032, hFrac: 0.11, chimney: false },
    { xFrac: 0.52, wFrac: 0.026, hFrac: 0.095, chimney: true },
    { xFrac: 0.74, wFrac: 0.028, hFrac: 0.1, chimney: false },
  ],
  clouds: [
    { xFrac: 0.06, yFrac: 0.88, s: 0.85, phase: 0 },
    { xFrac: 0.28, yFrac: 0.92, s: 1.05, phase: 1.4 },
    { xFrac: 0.52, yFrac: 0.86, s: 0.75, phase: 2.8 },
    { xFrac: 0.76, yFrac: 0.9, s: 0.95, phase: 4.2 },
    { xFrac: 0.93, yFrac: 0.87, s: 0.8, phase: 5.5 },
  ],
} as const;

/** Scale world tree height (m) to playground canvas pixels. */
export const PLAYGROUND_TREE_SCALE = 0.48;
