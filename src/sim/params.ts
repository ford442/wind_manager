import type { Emitter } from './emitters';
import { defaultEmitters } from './emitters';

export interface Params {
  nx: number;
  ny: number;
  domainW: number;
  domainH: number;
  dt: number;
  substeps: number;
  jacobiIters: number;

  tAmb: number;
  rhAmb: number;

  latentOn: boolean;

  emitters: Emitter[];
  selectedEmitterId: string | null;

  /** Wall-clock burst length when triggered (space / button). */
  burstDuration: number;
  /** Emission-rate multiplier during an active burst. */
  burstMultiplier: number;
  /** Seconds remaining in the current burst (runtime only). */
  burstRemaining: number;

  maxDroplets: number;

  relax: number;
  damp: number;

  paused: boolean;
  overlay: number;
  showArrows: boolean;
  showDroplets: boolean;

  showTracers: boolean;
  showTracerStreaks: boolean;
  tracerDensity: number;
  tracerLifetime: number;

  showGrass: boolean;
  grassDensity: number;
  grassStiffness: number;

  showTrees: boolean;

  showHouses: boolean;
  showClouds: boolean;
  showGroundMist: boolean;
  showWetGround: boolean;
  showGroundMoisture: boolean;

  /** Composite painterly backyard scene on the physics canvas. */
  stylizedView: boolean;
}

export function qSat(tC: number): number {
  if (tC <= -243.0) return 0;
  const es = 610.94 * Math.exp((17.625 * tC) / (tC + 243.04));
  const p = 101325.0;
  return (0.622 * es) / (p - 0.378 * es);
}

export function defaultParams(): Params {
  const domainW = 32.0;
  const emitters = defaultEmitters(domainW);
  return {
    nx: 512,
    ny: 128,
    domainW,
    domainH: 8.0,
    dt: 1 / 120,
    substeps: 2,
    jacobiIters: 30,

    tAmb: 30.0,
    rhAmb: 20,

    latentOn: true,

    emitters,
    selectedEmitterId: emitters[0]?.id ?? null,

    burstDuration: 0.5,
    burstMultiplier: 8,
    burstRemaining: 0,

    maxDroplets: 200000,

    relax: 1 / 120,
    damp: 0.02,

    paused: false,
    overlay: 2,
    showArrows: true,
    showDroplets: true,

    showTracers: true,
    showTracerStreaks: true,
    tracerDensity: 4500,
    tracerLifetime: 10,

    showGrass: true,
    grassDensity: 120,
    grassStiffness: 0.72,

    showTrees: true,

    showHouses: true,
    showClouds: true,
    showGroundMist: true,
    showWetGround: true,
    showGroundMoisture: true,
    stylizedView: true,
  };
}

export function cellSize(p: Params): number {
  return p.domainW / p.nx;
}

export function qAmb(p: Params): number {
  return (p.rhAmb / 100) * qSat(p.tAmb);
}
