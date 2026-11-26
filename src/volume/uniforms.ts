import type { UniformWriter } from '../sim/uniformWriter';
import { MAX_EMITTERS } from '../sim/emitters';
import { volumeGrid, volumeQAmb, volumeSpacing, type VolumeEmitter, type VolumeParams } from './params';

/**
 * `VParams` in vol_common.wgsl — 64 bytes.
 *
 * | Offset | Field         | Type  |
 * |-------:|---------------|-------|
 * |      0 | nx, ny, nz    | u32×3 |
 * |     12 | max_droplets  | u32   |
 * |     16 | h             | vec3f |
 * |     28 | dt            | f32   |
 * |     32 | t_amb, q_amb, latent_on, time | f32×4 |
 * |     48 | seed, emitter_count | u32×2 |
 * |     56 | relax, damp   | f32×2 |
 */
export const VOL_PARAMS_LAYOUT = {
  nx: 0, ny: 4, nz: 8, maxDroplets: 12,
  h: 16, dt: 28,
  tAmb: 32, qAmb: 36, latentOn: 40, time: 44,
  seed: 48, emitterCount: 52, relax: 56, damp: 60,
  BYTE_LENGTH: 64,
} as const;

/**
 * `VEmitter` in vol_droplets.wgsl — 64-byte stride.
 *
 * | Offset | Field | | Offset | Field |
 * |---:|---|---|---:|---|
 * | 0 | pos (vec3f) | | 32 | count (u32) |
 * | 12 | size | | 36 | kind (u32: 0 water, 1 air) |
 * | 16 | angle (rad) | | 40 | r_min (m) |
 * | 20 | azimuth (rad) | | 44 | r_max (m) |
 * | 24 | spread (rad) | | 48 | seed (u32) |
 * | 28 | speed | | 52 | mult (drops per parcel) |
 * | | | | 56–63 | pad |
 */
export const VOL_EMITTER_LAYOUT = {
  pos: 0, size: 12, angle: 16, azimuth: 20, spread: 24, speed: 28,
  count: 32, kind: 36, rMin: 40, rMax: 44, seed: 48, mult: 52,
  STRIDE: 64,
} as const;

export const VOL_EMITTER_ARRAY_BYTES = MAX_EMITTERS * VOL_EMITTER_LAYOUT.STRIDE;

/** One substep's upload slot: params, then the emitter array (copied in the encoder). */
export const VOL_SUBSTEP_LAYOUT = {
  params: 0,
  emitters: 256,
  STRIDE: 256 + Math.ceil(VOL_EMITTER_ARRAY_BYTES / 256) * 256,
} as const;

export interface VolEmitterDispatch {
  emitter: VolumeEmitter;
  count: number;
  seed: number;
}

const rad = (d: number) => (d * Math.PI) / 180;

export function packVolParams(
  w: UniformWriter, base: number, p: VolumeParams, time: number, seed: number, emitterCount: number,
): void {
  const L = VOL_PARAMS_LAYOUT;
  const g = volumeGrid(p);
  const h = volumeSpacing(p);
  w.u32(base + L.nx, g.nx);
  w.u32(base + L.ny, g.ny);
  w.u32(base + L.nz, g.nz);
  w.u32(base + L.maxDroplets, p.maxDroplets);
  w.f32(base + L.h, h[0]);
  w.f32(base + L.h + 4, h[1]);
  w.f32(base + L.h + 8, h[2]);
  w.f32(base + L.dt, p.dt);
  w.f32(base + L.tAmb, p.tAmb);
  w.f32(base + L.qAmb, volumeQAmb(p));
  w.f32(base + L.latentOn, p.latentOn ? 1 : 0);
  w.f32(base + L.time, time);
  w.u32(base + L.seed, seed);
  w.u32(base + L.emitterCount, emitterCount);
  w.f32(base + L.relax, p.relax);
  w.f32(base + L.damp, p.damp);
}

export function packVolEmitter(w: UniformWriter, base: number, d: VolEmitterDispatch): void {
  const L = VOL_EMITTER_LAYOUT;
  const e = d.emitter;
  w.f32(base + L.pos, e.x);
  w.f32(base + L.pos + 4, e.y);
  w.f32(base + L.pos + 8, e.z);
  w.f32(base + L.size, e.size);
  w.f32(base + L.angle, rad(e.angleDeg));
  w.f32(base + L.azimuth, rad(e.azimuthDeg));
  w.f32(base + L.spread, rad(e.spreadDeg));
  w.f32(base + L.speed, e.speed);
  w.u32(base + L.count, d.count);
  w.u32(base + L.kind, e.type === 'air' ? 1 : 0);
  w.f32(base + L.rMin, e.rMinUm * 1e-6);
  w.f32(base + L.rMax, e.rMaxUm * 1e-6);
  w.u32(base + L.seed, d.seed);
  w.f32(base + L.mult, e.type === 'water' ? e.dropsPerParcel : 1);
}

export function packVolSubstep(
  w: UniformWriter, slotBase: number, p: VolumeParams, time: number, seed: number,
  emitters: readonly VolEmitterDispatch[],
): void {
  const n = Math.min(emitters.length, MAX_EMITTERS);
  packVolParams(w, slotBase + VOL_SUBSTEP_LAYOUT.params, p, time, seed, n);
  for (let i = 0; i < n; i++) {
    packVolEmitter(w, slotBase + VOL_SUBSTEP_LAYOUT.emitters + i * VOL_EMITTER_LAYOUT.STRIDE, emitters[i]);
  }
}
