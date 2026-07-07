import { cellSize, qAmb, type Params } from './params';
import type { Emitter } from './emitters';
import { UniformWriter } from './uniformWriter';

/**
 * GPU uniform `Params` — must match `struct Params` in `src/shaders/common.wgsl`.
 *
 * All fields are 4-byte aligned (u32 / f32). Total size **84 bytes**.
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
 * |     32 | emit_x         | f32  | `emitter.x`                         |
 * |     36 | emit_y         | f32  | `emitter.y`                         |
 * |     40 | emit_angle     | f32  | `emitter.angleDeg` (rad)            |
 * |     44 | emit_spread    | f32  | `emitter.spreadDeg` (rad)           |
 * |     48 | emit_speed     | f32  | `emitter.speed`                     |
 * |     52 | emit_count     | u32  | droplets / impulses this dispatch   |
 * |     56 | emit_type      | u32  | `0` water, `1` air                  |
 * |     60 | r_min          | f32  | `emitter.rMinUm * 1e-6` (m)         |
 * |     64 | r_max          | f32  | `emitter.rMaxUm * 1e-6` (m)         |
 * |     68 | max_droplets   | u32  | `params.maxDroplets`                |
 * |     72 | seed           | u32  | per-dispatch RNG seed               |
 * |     76 | relax          | f32  | `params.relax`                      |
 * |     80 | damp           | f32  | `params.damp`                       |
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
  emitX: 32,
  emitY: 36,
  emitAngle: 40,
  emitSpread: 44,
  emitSpeed: 48,
  emitCount: 52,
  emitType: 56,
  rMin: 60,
  rMax: 64,
  maxDroplets: 68,
  seed: 72,
  relax: 76,
  damp: 80,
  BYTE_LENGTH: 84,
} as const;

export type SimParamsLayout = typeof SIM_PARAMS_LAYOUT;

export interface SimParamsPackInput {
  params: Params;
  emitter: Emitter;
  emitCount: number;
  seed: number;
  simTime: number;
}

export function packSimParamsUniform(writer: UniformWriter, input: SimParamsPackInput): void {
  const { params, emitter, emitCount, seed, simTime } = input;
  const L = SIM_PARAMS_LAYOUT;

  writer.u32(L.nx, params.nx);
  writer.u32(L.ny, params.ny);
  writer.f32(L.h, cellSize(params));
  writer.f32(L.dt, params.dt);
  writer.f32(L.tAmb, params.tAmb);
  writer.f32(L.qAmb, qAmb(params));
  writer.f32(L.latentOn, params.latentOn ? 1 : 0);
  writer.f32(L.time, simTime);
  writer.f32(L.emitX, emitter.x);
  writer.f32(L.emitY, emitter.y);
  writer.f32(L.emitAngle, (emitter.angleDeg * Math.PI) / 180);
  writer.f32(L.emitSpread, (emitter.spreadDeg * Math.PI) / 180);
  writer.f32(L.emitSpeed, emitter.speed);
  writer.u32(L.emitCount, emitCount);
  writer.u32(L.emitType, emitter.type === 'air' ? 1 : 0);
  writer.f32(L.rMin, emitter.rMinUm * 1e-6);
  writer.f32(L.rMax, emitter.rMaxUm * 1e-6);
  writer.u32(L.maxDroplets, params.maxDroplets);
  writer.u32(L.seed, seed);
  writer.f32(L.relax, params.relax);
  writer.f32(L.damp, params.damp);
}

export class SimParamsUniform {
  private readonly writer = new UniformWriter(SIM_PARAMS_LAYOUT.BYTE_LENGTH);

  get buffer(): ArrayBuffer {
    return this.writer.buffer;
  }

  pack(input: SimParamsPackInput): ArrayBuffer {
    packSimParamsUniform(this.writer, input);
    return this.buffer;
  }
}
