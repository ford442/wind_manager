import type { Params } from './params';
import { defaultParams } from './params';
import { createEmitter, clampEmitterPosition, type Emitter } from './emitters';
import { LAYOUT, heroTreeReachFt, sceneX } from '../render/sceneLayout';

export interface SimPreset {
  id: string;
  label: string;
  hint: string;
  patch: Partial<Omit<Params, 'emitters' | 'selectedEmitterId'>>;
  emitters: (Partial<Emitter> & Pick<Emitter, 'type' | 'y'> & { x?: number; xFrac?: number })[];
  selectedEmitter?: number;
}

type EmitterDef = SimPreset['emitters'][number];

function em(
  type: Emitter['type'],
  xOrFrac: number,
  y: number,
  partial?: Partial<Emitter> & { useFrac?: boolean },
): EmitterDef {
  const useFrac = partial?.useFrac ?? xOrFrac <= 1.0;
  const { useFrac: _, ...rest } = partial ?? {};
  if (useFrac) {
    return { type, xFrac: xOrFrac, y, ...rest };
  }
  return { type, x: xOrFrac, y, ...rest };
}

export const SIM_PRESETS: SimPreset[] = [
  {
    id: 'default',
    label: 'Default — single center spray',
    hint: 'Baseline water emitter near mid-domain.',
    patch: {},
    emitters: [em('water', LAYOUT.emitterCenter, 0.3, { useFrac: true })],
  },
  {
    id: 'vertical-hose-left',
    label: 'Strong vertical hose (left) — downdraft + distant sway',
    hint: 'Tight column on the left; cold air sinks and fans out — watch grass and far trees.',
    patch: {
      latentOn: true,
      overlay: 2,
      tAmb: 32,
      rhAmb: 18,
      showTrees: true,
      showGrass: true,
      showTracers: true,
      showTracerStreaks: true,
      showArrows: false,
      showDroplets: true,
      grassStiffness: 0.55,
    },
    emitters: [
      em('water', LAYOUT.emitterLeft, 0.28, {
        useFrac: true,
        angleDeg: 0,
        spreadDeg: 11,
        speed: 14.5,
        rate: 12000,
        rMinUm: 40,
        rMaxUm: 650,
      }),
    ],
  },
  {
    id: 'angled-crossflow',
    label: 'Angled spray — cross flow along floor',
    hint: 'Jet tilted right creates a diagonal column and a strong horizontal gust at ground level.',
    patch: {
      latentOn: true,
      overlay: 1,
      tAmb: 30,
      rhAmb: 22,
      showTrees: true,
      showGrass: true,
      showWetGround: true,
      showTracers: true,
    },
    emitters: [
      em('water', 0.1, 0.34, {
        useFrac: true,
        angleDeg: 26,
        spreadDeg: 16,
        speed: 12,
        rate: 9500,
      }),
    ],
  },
  {
    id: 'water-plus-air',
    label: 'Multiple water + air sources',
    hint: 'Evaporative downdraft from the left plus a momentum-only air stream steering the outflow.',
    patch: {
      latentOn: true,
      overlay: 1,
      showTrees: true,
      showGrass: true,
      showHouses: true,
    },
    emitters: [
      em('water', LAYOUT.emitterLeft, 0.3, {
        useFrac: true,
        angleDeg: 4,
        spreadDeg: 13,
        speed: 13,
        rate: 10000,
      }),
      em('air', 0.58, 0.48, {
        useFrac: true,
        angleDeg: -18,
        spreadDeg: 24,
        speed: 11,
        rate: 7000,
      }),
    ],
    selectedEmitter: 0,
  },
  {
    id: 'latent-off',
    label: 'Latent heat OFF — momentum only',
    hint: 'Same spray geometry but no evaporative cooling — expect updraft instead of downdraft.',
    patch: {
      latentOn: false,
      overlay: 2,
      tAmb: 30,
      rhAmb: 20,
      showDroplets: true,
    },
    emitters: [
      em('water', LAYOUT.emitterLeft, 0.32, {
        useFrac: true,
        angleDeg: 0,
        spreadDeg: 14,
        speed: 12,
        rate: 9000,
      }),
    ],
  },
  {
    id: 'dry-hot',
    label: 'Low humidity / high temp contrast',
    hint: 'Hot dry ambient air maximizes evaporative cooling and downdraft strength.',
    patch: {
      latentOn: true,
      overlay: 2,
      tAmb: 42,
      rhAmb: 6,
      showTrees: true,
      showGrass: true,
      showWetGround: true,
    },
    emitters: [
      em('water', LAYOUT.emitterLeft, 0.3, {
        useFrac: true,
        angleDeg: 0,
        spreadDeg: 12,
        speed: 13.5,
        rate: 11000,
        rMinUm: 35,
        rMaxUm: 550,
      }),
    ],
  },
  {
    id: 'tree-shake',
    label: '100 ft tree shake — hero demo',
    hint: 'Hose at the left fence; outflow travels ~90+ ft to the tall canopy on the right.',
    patch: {
      latentOn: true,
      overlay: 1,
      tAmb: 34,
      rhAmb: 12,
      showTrees: true,
      showGrass: true,
      showHouses: true,
      showGroundMist: true,
      showWetGround: true,
      showTracers: true,
      showTracerStreaks: true,
      grassStiffness: 0.42,
      grassDensity: 140,
      showArrows: false,
      showDroplets: true,
    },
    emitters: [
      em('water', LAYOUT.emitterLeft, 0.22, {
        useFrac: true,
        angleDeg: 5,
        spreadDeg: 9,
        speed: 15,
        rate: 16500,
        rMinUm: 30,
        rMaxUm: 500,
      }),
    ],
  },
  {
    id: 'microburst-puddle',
    label: 'Microburst + puddling',
    hint: 'Heavy sustained spray builds wet ground and near-surface humidity for a microburst-like outflow.',
    patch: {
      latentOn: true,
      overlay: 3,
      tAmb: 33,
      rhAmb: 15,
      showWetGround: true,
      showGroundMist: true,
      showGrass: true,
      showTrees: true,
      showDroplets: true,
      showTracers: true,
    },
    emitters: [
      em('water', LAYOUT.emitterLeft, 0.26, {
        useFrac: true,
        angleDeg: 2,
        spreadDeg: 10,
        speed: 14,
        rate: 16500,
        rMinUm: 45,
        rMaxUm: 700,
      }),
    ],
  },
  {
    id: 'dual-hose',
    label: 'Twin hoses — converging gusts',
    hint: 'Two water columns from left and right; gusts collide and swirl near center.',
    patch: {
      latentOn: true,
      overlay: 1,
      showTrees: true,
      showGrass: true,
      showTracers: true,
    },
    emitters: [
      em('water', LAYOUT.emitterLeft, 0.32, {
        useFrac: true,
        angleDeg: 8,
        spreadDeg: 12,
        speed: 12,
        rate: 8000,
      }),
      em('water', LAYOUT.emitterRight, 0.32, {
        useFrac: true,
        angleDeg: -8,
        spreadDeg: 12,
        speed: 12,
        rate: 8000,
      }),
    ],
  },
];

type ScalarParamKey = {
  [K in keyof Params]: K extends 'emitters' | 'selectedEmitterId' ? never : K;
}[keyof Params];

function assignScalar<K extends ScalarParamKey>(
  target: Params,
  key: K,
  value: Params[K],
): void {
  target[key] = value;
}

export function applyPreset(target: Params, preset: SimPreset): void {
  const defaults = defaultParams();
  for (const key of Object.keys(defaults) as ScalarParamKey[]) {
    assignScalar(target, key, preset.patch[key] ?? defaults[key]);
  }

  target.emitters = preset.emitters.map((def) => {
    const x = def.x ?? sceneX(target, def.xFrac ?? LAYOUT.emitterCenter);
    const em = createEmitter(def.type, x, def.y, def);
    clampEmitterPosition(em, target);
    return em;
  });
  const sel = preset.selectedEmitter ?? 0;
  target.selectedEmitterId = target.emitters[sel]?.id ?? target.emitters[0]?.id ?? null;
}

export function presetHint(preset: SimPreset, params: Params): string {
  if (preset.id === 'tree-shake') {
    const ft = Math.round(heroTreeReachFt(params));
    return `${preset.hint} (hero tree ~${ft} ft downwind)`;
  }
  return preset.hint;
}

export function getPreset(id: string): SimPreset | undefined {
  return SIM_PRESETS.find((p) => p.id === id);
}
