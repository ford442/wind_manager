import { cellSize, qAmb, type Params } from './params';
import { MAX_EMITTERS, type Emitter } from './emitters';
import type { UniformWriter } from './uniformWriter';

/**
 * GPU uniform `Params` — global per-substep values shared by every sim kernel.
 * Must match `struct Params` in `src/shaders/common.wgsl`. Total **64 bytes**.
 *
 * | Offset | WGSL field     | Type | CPU source                          |
 * |-------:|----------------|------|-------------------------------------|
 * |      0 | nx             | u32  | `params.nx`                         |
 * |      4 | ny             | u32  | `params.ny`                         |
 * |      8 | h              | f32  | `cellSize(params)`                  |
 * |     12 | dt             | f32  | `params.dt`                         |
 * |     16 | t_amb          | f32  | `params.tAmb`                       |
 * |     20 | q_amb          | f32  | `qAmb(params)`                      |
 * |     24 | latent_on      | f32  | `params.latentOn ? 1 : 0`           |
 * |     28 | time           | f32  | simulation time (s)                 |
 * |     32 | max_droplets   | u32  | `params.maxDroplets`                |
 * |     36 | seed           | u32  | per-substep RNG seed                |
 * |     40 | relax          | f32  | `params.relax`                      |
 * |     44 | damp           | f32  | `params.damp`                       |
 * |     48 | emitter_count  | u32  | live entries in the emitter array   |
 * |  52–63 | _pad           |      |                                     |
 */
export const SIM_PARAMS_LAYOUT = {
  nx: 0,
  ny: 4,
  h: 8,
  dt: 12,
  tAmb: 16,
  qAmb: 20,
  latentOn: 24,
  time: 28,
  maxDroplets: 32,
  seed: 36,
  relax: 40,
  damp: 44,
  emitterCount: 48,
  BYTE_LENGTH: 64,
} as const;

/**
 * One element of `E : array<EmitterParams, MAX_EMITTERS>` in `droplets.wgsl`.
 * Stride **48 bytes** (uniform array stride must be a multiple of 16).
 *
 * | Offset | WGSL field | Type | CPU source                         |
 * |-------:|------------|------|------------------------------------|
 * |      0 | x          | f32  | `emitter.x`                        |
 * |      4 | y          | f32  | `emitter.y`                        |
 * |      8 | angle      | f32  | `emitter.angleDeg` (rad)           |
 * |     12 | spread     | f32  | `emitter.spreadDeg` (rad)          |
 * |     16 | speed      | f32  | `emitter.speed`                    |
 * |     20 | count      | u32  | droplets / impulses this substep   |
 * |     24 | kind       | u32  | `0` water, `1` air                 |
 * |     28 | r_min      | f32  | `emitter.rMinUm * 1e-6` (m)        |
 * |     32 | r_max      | f32  | `emitter.rMaxUm * 1e-6` (m)        |
 * |     36 | seed       | u32  | per-emitter RNG seed               |
 * |  40–47 | _pad       |      |                                    |
 */
export const EMITTER_PARAMS_LAYOUT = {
  x: 0,
  y: 4,
  angle: 8,
  spread: 12,
  speed: 16,
  count: 20,
  kind: 24,
  rMin: 28,
  rMax: 32,
  seed: 36,
  STRIDE: 48,
} as const;

export const EMITTER_ARRAY_BYTES = MAX_EMITTERS * EMITTER_PARAMS_LAYOUT.STRIDE;

/**
 * One substep's upload: `Params` followed by the emitter array. Both offsets
 * are 4-byte aligned for `copyBufferToBuffer`; the slot is padded to 256 bytes
 * so slots stay aligned if they are ever bound directly with dynamic offsets.
 */
export const SUBSTEP_UPLOAD_LAYOUT = {
  params: 0,
  emitters: 256,
  STRIDE: 256 + Math.ceil(EMITTER_ARRAY_BYTES / 256) * 256,
} as const;

export interface EmitterDispatch {
  emitter: Emitter;
  count: number;
  seed: number;
}

export interface SubstepPackInput {
  params: Params;
  simTime: number;
  seed: number;
  emitters: readonly EmitterDispatch[];
}

export function packSimParams(
  w: UniformWriter,
  base: number,
  params: Params,
  simTime: number,
  seed: number,
  emitterCount: number,
): void {
  const L = SIM_PARAMS_LAYOUT;
  w.u32(base + L.nx, params.nx);
  w.u32(base + L.ny, params.ny);
  w.f32(base + L.h, cellSize(params));
  w.f32(base + L.dt, params.dt);
  w.f32(base + L.tAmb, params.tAmb);
  w.f32(base + L.qAmb, qAmb(params));
  w.f32(base + L.latentOn, params.latentOn ? 1 : 0);
  w.f32(base + L.time, simTime);
  w.u32(base + L.maxDroplets, params.maxDroplets);
  w.u32(base + L.seed, seed);
  w.f32(base + L.relax, params.relax);
  w.f32(base + L.damp, params.damp);
  w.u32(base + L.emitterCount, emitterCount);
}

export function packEmitterParams(w: UniformWriter, base: number, d: EmitterDispatch): void {
  const L = EMITTER_PARAMS_LAYOUT;
  const em = d.emitter;
  w.f32(base + L.x, em.x);
  w.f32(base + L.y, em.y);
  w.f32(base + L.angle, (em.angleDeg * Math.PI) / 180);
  w.f32(base + L.spread, (em.spreadDeg * Math.PI) / 180);
  w.f32(base + L.speed, em.speed);
  w.u32(base + L.count, d.count);
  w.u32(base + L.kind, em.type === 'air' ? 1 : 0);
  w.f32(base + L.rMin, em.rMinUm * 1e-6);
  w.f32(base + L.rMax, em.rMaxUm * 1e-6);
  w.u32(base + L.seed, d.seed);
}

/** Pack one substep slot (params + emitter array) at `slotBase` in `w`. */
export function packSubstepUpload(w: UniformWriter, slotBase: number, input: SubstepPackInput): void {
  const n = Math.min(input.emitters.length, MAX_EMITTERS);
  packSimParams(w, slotBase + SUBSTEP_UPLOAD_LAYOUT.params, input.params, input.simTime, input.seed, n);
  const eBase = slotBase + SUBSTEP_UPLOAD_LAYOUT.emitters;
  for (let i = 0; i < n; i++) {
    packEmitterParams(w, eBase + i * EMITTER_PARAMS_LAYOUT.STRIDE, input.emitters[i]);
  }
}
