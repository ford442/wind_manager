import type { Params } from '../sim/params';

/** Horizontal fraction [0, 1] along the domain width. */
export function sceneX(p: Params, xFrac: number): number {
  return xFrac * p.domainW;
}

/** Vertical fraction [0, 1] from ground up. */
export function sceneY(p: Params, yFrac: number): number {
  return yFrac * p.domainH;
}

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

export function heroTreeX(p: Params): number {
  return sceneX(p, LAYOUT.trees[4].xFrac);
}

export function heroTreeReachFt(p: Params, emitterX = sceneX(p, LAYOUT.emitterLeft)): number {
  return (heroTreeX(p) - emitterX) * 3.28084;
}
