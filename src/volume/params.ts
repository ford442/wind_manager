import { qSat } from '../sim/params';
import { MAX_EMITTERS } from '../sim/emitters';
import { VOL_PRESSURE_CYCLES } from './pressure';
import { LAYOUT } from '../render/sceneData';
import { DEFAULT_VOLUME_GRID, VOLUME_DOMAIN, VOLUME_MAX_DROPLETS, gridById, spacing, type VolumeGrid } from './grid';

export { MAX_EMITTERS };

export type VolumeEmitterType = 'water' | 'air';

/**
 * A source in the yard: a horizontal disk of radius `size` (m) at (x, y, z),
 * spraying a cone of half-angle `spreadDeg` around a direction tilted
 * `angleDeg` from vertical toward `azimuthDeg` (0 = +x, 90 = +z).
 */
export interface VolumeEmitter {
  type: VolumeEmitterType;
  x: number;
  y: number;
  z: number;
  size: number;
  angleDeg: number;
  azimuthDeg: number;
  spreadDeg: number;
  speed: number;
  /** Parcels (water) or impulses (air) per second for the whole source. */
  rate: number;
  /**
   * Water: real drops per computational parcel. The water flow is
   * `rate * dropsPerParcel` drops per second while the GPU tracks only `rate`
   * parcels, so the droplet cap does not limit how hard a hose sprays.
   */
  dropsPerParcel: number;
  rMinUm: number;
  rMaxUm: number;
}

export type ViewMode = 'yard' | 'side' | 'split';
export type CameraMode = 'orbit' | 'walk';
export type SideField = 'temp' | 'speed';

export interface VolumeParams {
  gridId: string;
  dt: number;
  substeps: number;
  pressureCycles: number;
  tAmb: number;
  rhAmb: number;
  latentOn: boolean;
  solidObstacles: boolean;
  relax: number;
  damp: number;
  maxDroplets: number;
  emitters: VolumeEmitter[];
  paused: boolean;

  viewMode: ViewMode;
  cameraMode: CameraMode;
  showVolume: boolean;
  showDroplets: boolean;
  showGrass: boolean;
  /** Side view: z of the slice (m) and the field it shows. */
  sideZ: number;
  sideField: SideField;
  /** Raymarch: temperature deficit (K) that reads fully opaque. */
  volumeContrast: number;
}

export function waterEmitter(x: number, z: number, partial: Partial<VolumeEmitter> = {}): VolumeEmitter {
  return {
    type: 'water',
    x, y: 0.3, z,
    size: 0.12,
    angleDeg: 0,
    azimuthDeg: 0,
    // A wider cone than the slice's 11°: the drops fall back around the
    // rising jet, so the cooled air sinks outside the updraft it drives.
    spreadDeg: 22,
    speed: 14,
    // 40k parcels/s × 40 drops ≈ 220 g/s (13 L/min, a garden hose on full)
    // of 40–650 µm spray. The slice's drop rate is per metre of depth; a point
    // source spreads its outflow over a circle, so it needs far more water
    // for a gust of the same speed.
    rate: 40000,
    dropsPerParcel: 40,
    rMinUm: 40,
    rMaxUm: 650,
    ...partial,
  };
}

export function airEmitter(x: number, z: number, partial: Partial<VolumeEmitter> = {}): VolumeEmitter {
  return {
    type: 'air',
    x, y: 0.4, z,
    size: 0.2,
    angleDeg: 90,
    azimuthDeg: 0,
    spreadDeg: 20,
    speed: 9,
    rate: 20000,
    dropsPerParcel: 1,
    rMinUm: 50,
    rMaxUm: 800,
    ...partial,
  };
}

/** Hero tree (index 4 in the shared layout) in metres. */
export function heroTree(): { x: number; z: number; index: number } {
  const i = LAYOUT.trees.length - 1;
  const t = LAYOUT.trees[i];
  return { x: t.xFrac * VOLUME_DOMAIN.w, z: t.zFrac * VOLUME_DOMAIN.d, index: i };
}

/** Default hose: vertical, 4 m in -x of the hero tree, mid-yard in z. */
export function defaultHose(): VolumeEmitter {
  const tree = heroTree();
  return waterEmitter(tree.x - 4.2, VOLUME_DOMAIN.d * 0.5);
}

export function defaultVolumeParams(): VolumeParams {
  return {
    gridId: DEFAULT_VOLUME_GRID,
    dt: 1 / 120,
    substeps: 2,
    pressureCycles: VOL_PRESSURE_CYCLES,
    tAmb: 32,
    rhAmb: 18,
    latentOn: true,
    solidObstacles: true,
    relax: 1 / 120,
    damp: 0.02,
    maxDroplets: VOLUME_MAX_DROPLETS,
    emitters: [defaultHose()],
    paused: false,

    viewMode: 'split',
    cameraMode: 'orbit',
    showVolume: true,
    showDroplets: true,
    showGrass: true,
    sideZ: VOLUME_DOMAIN.d * 0.5,
    sideField: 'temp',
    volumeContrast: 8,
  };
}

export function volumeGrid(p: VolumeParams): VolumeGrid {
  return gridById(p.gridId);
}

export function volumeSpacing(p: VolumeParams): [number, number, number] {
  return spacing(volumeGrid(p));
}

export function volumeQAmb(p: VolumeParams): number {
  return (p.rhAmb / 100) * qSat(p.tAmb);
}

/** Keep a source inside the yard and above the floor. */
export function clampVolumeEmitter(em: VolumeEmitter): void {
  const m = 0.4;
  em.x = Math.max(m, Math.min(VOLUME_DOMAIN.w - m, em.x));
  em.z = Math.max(m, Math.min(VOLUME_DOMAIN.d - m, em.z));
  em.y = Math.max(0.1, Math.min(VOLUME_DOMAIN.h - 1, em.y));
  em.size = Math.max(0.02, Math.min(1.5, em.size));
  em.dropsPerParcel = Math.max(1, Math.min(500, em.dropsPerParcel));
  em.rate = Math.max(0, Math.min(150000, em.rate));
  em.rMinUm = Math.max(5, Math.min(2000, em.rMinUm));
  em.rMaxUm = Math.max(em.rMinUm, Math.min(2000, em.rMaxUm));
}

export interface VolumePreset {
  id: string;
  label: string;
  apply: (p: VolumeParams) => void;
}

/** Hot, dry afternoon (the slice's "strong vertical hose" air). */
const base = (p: VolumeParams, emitters: VolumeEmitter[], latentOn = true): void => {
  p.emitters = emitters;
  p.latentOn = latentOn;
  p.tAmb = 32;
  p.rhAmb = 18;
  p.solidObstacles = true;
  p.sideZ = emitters[0]?.z ?? VOLUME_DOMAIN.d * 0.5;
};

export const VOLUME_PRESETS: readonly VolumePreset[] = [
  {
    id: 'hose-tree',
    label: 'Vertical hose by the hero tree',
    apply: (p) => base(p, [defaultHose()]),
  },
  {
    id: 'hose-tree-dry',
    label: 'Same hose, latent heat OFF',
    apply: (p) => base(p, [defaultHose()], false),
  },
  {
    id: 'house-spray',
    label: 'Hose aimed at the left house',
    apply: (p) => {
      const h = LAYOUT.houses[0];
      const hx = h.xFrac * VOLUME_DOMAIN.w;
      const hz = h.zFrac * VOLUME_DOMAIN.d;
      base(p, [waterEmitter(hx - 2.2, hz, { angleDeg: 35, azimuthDeg: 0 })]);
    },
  },
  {
    id: 'yard-centre',
    label: 'Hose in the middle of the yard',
    apply: (p) => base(p, [waterEmitter(VOLUME_DOMAIN.w * 0.5, VOLUME_DOMAIN.d * 0.5)]),
  },
  {
    id: 'hose-plus-fan',
    label: 'Hose plus a fan pushing toward the tree',
    apply: (p) => {
      const hose = defaultHose();
      base(p, [hose, airEmitter(hose.x - 1.5, hose.z, { y: 0.35, angleDeg: 88, azimuthDeg: 0 })]);
    },
  },
];

export const VOLUME_EMITTER_LIMIT = MAX_EMITTERS;

/** Mean drop mass (kg) for radii log-uniform in [rMin, rMax] (how `vol_emit` samples them). */
export function meanDropMass(em: VolumeEmitter): number {
  const a = em.rMinUm * 1e-6;
  const b = em.rMaxUm * 1e-6;
  const meanR3 = Math.abs(b - a) < 1e-12 ? a ** 3 : (b ** 3 - a ** 3) / (3 * Math.log(b / a));
  return 1000 * (4 / 3) * Math.PI * meanR3;
}

/** Water flow of a source in g/s (0 for air). */
export function waterFlowGs(em: VolumeEmitter): number {
  return em.type === 'water' ? em.rate * em.dropsPerParcel * meanDropMass(em) * 1000 : 0;
}
