import { MAX_EMITTERS } from '../sim/emitters';
import { UniformWriter } from '../sim/uniformWriter';
import { compileShaderModule, withValidationScope, workgroups1d } from '../render/webgpu';
import { volumeShaderParts } from './shaderParts';
import { VOL_PRESSURE_COARSE_SMOOTHS, VOL_PRESSURE_SMOOTHS } from './pressure';
import type { VolumeFields } from './fields';
import type { VolumeParams } from './params';
import {
  VOL_EMITTER_ARRAY_BYTES,
  VOL_PARAMS_LAYOUT,
  VOL_SUBSTEP_LAYOUT,
  packVolSubstep,
  type VolEmitterDispatch,
} from './uniforms';

import volDropletsSrc from '../shaders/vol_droplets.wgsl?raw';
import volScatterSrc from '../shaders/vol_scatter.wgsl?raw';
import volAdvectSrc from '../shaders/vol_advect.wgsl?raw';
import volBuoyancySrc from '../shaders/vol_buoyancy.wgsl?raw';
import volPressureSrc from '../shaders/vol_pressure.wgsl?raw';

export interface VolumeSim {
  time: number;
  frame: number;
  /** The uniform the solver kernels read (scene passes may bind it too). */
  params: GPUBuffer;
  setFields: (f: VolumeFields) => void;
  reset: (p: VolumeParams) => void;
  /**
   * Record one frame's substeps into `enc`; uploads are queued before return.
   * `timing` (optional) gets a timestamp at the start of the first substep
   * pass (`begin`) and the end of the last (`end`).
   */
  step: (p: VolumeParams, enc: GPUCommandEncoder, timing?: PassTiming) => void;
}

export interface PassTiming {
  querySet: GPUQuerySet;
  begin: number;
  end: number;
}

interface Parity {
  update: GPUBindGroup;
  apply: GPUBindGroup;
  applyWet: GPUBindGroup;
  advect: GPUBindGroup;
  buoy: GPUBindGroup;
  bound: GPUBindGroup;
  div: GPUBindGroup;
  proj: GPUBindGroup;
}

const wg3 = (x: number, y: number, z: number): [number, number, number] =>
  [Math.ceil(x / 4), Math.ceil(y / 4), Math.ceil(z / 4)];

/**
 * The yard-volume solver: the slice's steps in the slice's order (emit,
 * droplet update, scatter, advect, buoyancy, boundaries, divergence, V-cycles,
 * project) on an `nx × ny × nz` grid. Recording model as in `src/sim/step.ts`:
 * every substep's params + emitter array are packed into one staging slot,
 * the staging buffer is written once per frame, and each substep copies its
 * slot into the live uniforms inside the encoder.
 */
export async function createVolumeSim(device: GPUDevice, fields: VolumeFields): Promise<VolumeSim> {
  const mod = (name: string, src: string) => compileShaderModule(device, name, volumeShaderParts(name, src));
  const [mDrops, mScatter, mAdvect, mBuoy, mPressure] = await Promise.all([
    mod('vol_droplets', volDropletsSrc),
    mod('vol_scatter', volScatterSrc),
    mod('vol_advect', volAdvectSrc),
    mod('vol_buoyancy', volBuoyancySrc),
    mod('vol_pressure', volPressureSrc),
  ]);
  const pipe = (module: GPUShaderModule, entryPoint: string) =>
    device.createComputePipeline({ label: entryPoint, layout: 'auto', compute: { module, entryPoint } });
  const pipes = await withValidationScope(device, 'volume compute', () => ({
    emit: pipe(mDrops, 'vol_emit'),
    air: pipe(mDrops, 'vol_air_inject'),
    update: pipe(mDrops, 'vol_update'),
    apply: pipe(mScatter, 'vol_apply'),
    applyWet: pipe(mScatter, 'vol_apply_wet'),
    advect: pipe(mAdvect, 'vol_advect'),
    buoy: pipe(mBuoy, 'vol_buoyancy'),
    bound: pipe(mBuoy, 'vol_boundaries'),
    div: pipe(mPressure, 'vol_divergence'),
    smooth: pipe(mPressure, 'vol_pressure_smooth'),
    restrict: pipe(mPressure, 'vol_restrict_residual'),
    prolong: pipe(mPressure, 'vol_prolongate'),
    proj: pipe(mPressure, 'vol_project'),
  }));

  const paramsUniform = device.createBuffer({
    label: 'vol params', size: VOL_PARAMS_LAYOUT.BYTE_LENGTH,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const emitterUniform = device.createBuffer({
    label: 'vol emitters', size: VOL_EMITTER_ARRAY_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  let staging: GPUBuffer | null = null;
  let writer: UniformWriter | null = null;
  let slots = 0;
  function ensureStaging(n: number): { buffer: GPUBuffer; writer: UniformWriter } {
    if (!staging || !writer || n > slots) {
      staging?.destroy();
      slots = Math.max(n, 4);
      const size = slots * VOL_SUBSTEP_LAYOUT.STRIDE;
      staging = device.createBuffer({
        label: 'vol upload staging', size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      writer = new UniformWriter(size);
    }
    return { buffer: staging, writer };
  }

  const U: GPUBindGroupEntry = { binding: 0, resource: { buffer: paramsUniform } };
  const EP: GPUBindGroupEntry = { binding: 10, resource: { buffer: emitterUniform } };
  const e = (binding: number, buffer: GPUBuffer): GPUBindGroupEntry => ({ binding, resource: { buffer } });
  const bg = (p: GPUComputePipeline, label: string, entries: GPUBindGroupEntry[]) =>
    device.createBindGroup({ label, layout: p.getBindGroupLayout(0), entries });

  let emitBG: GPUBindGroup;
  let airBG: GPUBindGroup;
  let parity: [Parity, Parity];
  let parity0Vel: GPUBuffer;
  let multigrid: { smooth: [GPUBindGroup, GPUBindGroup]; restrict?: GPUBindGroup; prolong?: GPUBindGroup }[];

  function buildParity(
    vel0: GPUBuffer, vel1: GPUBuffer, T0: GPUBuffer, T1: GPUBuffer, q0: GPUBuffer, q1: GPUBuffer,
  ): Parity {
    const f = fields;
    return {
      update: bg(pipes.update, 'vol_update', [
        U, e(1, f.drops), e(3, vel0), e(4, T0), e(5, q0), e(6, f.accQT), e(7, f.solid), e(8, f.accM), e(9, f.accWet),
      ]),
      apply: bg(pipes.apply, 'vol_apply', [
        U, e(1, vel0), e(2, T0), e(3, q0), e(4, f.accQT), e(5, f.solid), e(6, f.accM),
      ]),
      applyWet: bg(pipes.applyWet, 'vol_apply_wet', [
        U, e(2, T0), e(3, q0), e(5, f.solid), e(7, f.wet), e(8, f.accWet),
      ]),
      advect: bg(pipes.advect, 'vol_advect', [
        U, e(1, vel0), e(2, T0), e(3, q0), e(4, vel1), e(5, T1), e(6, q1), e(7, f.solid),
      ]),
      // Buoyancy, boundaries and the projection act on the advected half.
      buoy: bg(pipes.buoy, 'vol_buoyancy', [U, e(1, vel1), e(2, T1), e(3, q1), e(4, f.solid)]),
      bound: bg(pipes.bound, 'vol_boundaries', [U, e(1, vel1), e(2, T1), e(3, q1), e(4, f.solid)]),
      div: bg(pipes.div, 'vol_divergence', [
        U, e(1, vel1), e(2, f.pressure[0].rhs), e(5, f.solid),
      ]),
      proj: bg(pipes.proj, 'vol_project', [U, e(1, vel1), e(3, f.pressure[0].p0), e(5, f.solid)]),
    };
  }

  function rebuild(): void {
    const f = fields;
    emitBG = bg(pipes.emit, 'vol_emit', [U, EP, e(1, f.drops), e(2, f.dropCounter)]);
    airBG = bg(pipes.air, 'vol_air_inject', [U, EP, e(7, f.solid), e(8, f.accM)]);
    multigrid = f.pressure.map((lv, i) => {
      const base = [e(6, lv.uniform), e(7, lv.graph)];
      const coarse = f.pressure[i + 1];
      return {
        smooth: [
          bg(pipes.smooth, `smooth L${i} a`, [...base, e(2, lv.rhs), e(3, lv.p0), e(4, lv.p1)]),
          bg(pipes.smooth, `smooth L${i} b`, [...base, e(2, lv.rhs), e(3, lv.p1), e(4, lv.p0)]),
        ],
        restrict: coarse && bg(pipes.restrict, `restrict L${i}`, [
          ...base, e(2, lv.rhs), e(3, lv.p0), e(9, coarse.rhs), e(10, coarse.p0),
        ]),
        prolong: coarse && bg(pipes.prolong, `prolong L${i}`, [...base, e(4, lv.p0), e(8, coarse.p0)]),
      };
    });
    parity = [
      buildParity(f.vel0, f.vel1, f.T0, f.T1, f.q0, f.q1),
      buildParity(f.vel1, f.vel0, f.T1, f.T0, f.q1, f.q0),
    ];
    parity0Vel = f.vel0;
  }
  rebuild();

  function swap(): void {
    const f = fields;
    [f.vel0, f.vel1] = [f.vel1, f.vel0];
    [f.T0, f.T1] = [f.T1, f.T0];
    [f.q0, f.q1] = [f.q1, f.q0];
  }

  function vCycle(pass: GPUComputePassEncoder, index: number): void {
    const lv = fields.pressure[index];
    const g = multigrid[index];
    const grid = wg3(lv.nx, lv.ny, lv.nz);
    const smooth = (count: number) => {
      pass.setPipeline(pipes.smooth);
      for (let i = 0; i < count; i++) {
        pass.setBindGroup(0, g.smooth[i % 2]);
        pass.dispatchWorkgroups(...grid);
      }
    };
    if (!g.restrict || !g.prolong) {
      smooth(VOL_PRESSURE_COARSE_SMOOTHS);
      return;
    }
    smooth(VOL_PRESSURE_SMOOTHS);
    pass.setPipeline(pipes.restrict);
    pass.setBindGroup(0, g.restrict);
    pass.dispatchWorkgroups(...wg3(lv.nx / 2, lv.ny / 2, lv.nz / 2));
    vCycle(pass, index + 1);
    pass.setPipeline(pipes.prolong);
    pass.setBindGroup(0, g.prolong);
    pass.dispatchWorkgroups(...grid);
    smooth(VOL_PRESSURE_SMOOTHS);
  }

  let time = 0;
  let frame = 0;
  let seed = 1;
  /** Fractional emission carried between substeps so low rates still emit. */
  const carry = new Map<VolumeParams['emitters'][number], number>();

  interface Plan { list: VolEmitterDispatch[]; water: number; air: number; maxWater: number; maxAir: number }
  function plan(p: VolumeParams): Plan {
    const out: Plan = { list: [], water: 0, air: 0, maxWater: 0, maxAir: 0 };
    for (const em of p.emitters) {
      if (out.list.length >= MAX_EMITTERS) break;
      const exact = em.rate * p.dt + (carry.get(em) ?? 0);
      const count = Math.max(0, Math.floor(exact));
      carry.set(em, exact - count);
      if (count <= 0) continue;
      out.list.push({ emitter: em, count, seed: seed++ });
      if (em.type === 'water') { out.water++; out.maxWater = Math.max(out.maxWater, count); }
      else { out.air++; out.maxAir = Math.max(out.maxAir, count); }
    }
    return out;
  }

  function recordSubstep(
    enc: GPUCommandEncoder, p: VolumeParams, buffer: GPUBuffer, slot: number, pl: Plan,
    timestampWrites?: GPUComputePassTimestampWrites,
  ): void {
    const base = slot * VOL_SUBSTEP_LAYOUT.STRIDE;
    enc.copyBufferToBuffer(buffer, base + VOL_SUBSTEP_LAYOUT.params, paramsUniform, 0, VOL_PARAMS_LAYOUT.BYTE_LENGTH);
    enc.copyBufferToBuffer(buffer, base + VOL_SUBSTEP_LAYOUT.emitters, emitterUniform, 0, VOL_EMITTER_ARRAY_BYTES);
    const f = fields;
    const b = f.vel0 === parity0Vel ? parity[0] : parity[1];
    const g = f.grid;
    const grid = wg3(g.nx, g.ny, g.nz);
    const pass = enc.beginComputePass({ label: 'vol substep', timestampWrites });
    if (pl.water > 0) {
      pass.setPipeline(pipes.emit);
      pass.setBindGroup(0, emitBG);
      pass.dispatchWorkgroups(workgroups1d(pl.maxWater, 64), pl.list.length);
    }
    if (pl.air > 0) {
      pass.setPipeline(pipes.air);
      pass.setBindGroup(0, airBG);
      pass.dispatchWorkgroups(workgroups1d(pl.maxAir, 64), pl.list.length);
    }
    pass.setPipeline(pipes.update);
    pass.setBindGroup(0, b.update);
    pass.dispatchWorkgroups(workgroups1d(p.maxDroplets, 128));
    pass.setPipeline(pipes.apply);
    pass.setBindGroup(0, b.apply);
    pass.dispatchWorkgroups(workgroups1d(f.n, 64));
    pass.setPipeline(pipes.applyWet);
    pass.setBindGroup(0, b.applyWet);
    pass.dispatchWorkgroups(workgroups1d(g.nx * g.nz, 64));
    for (const [pl2, group] of [
      [pipes.advect, b.advect], [pipes.buoy, b.buoy], [pipes.bound, b.bound], [pipes.div, b.div],
    ] as const) {
      pass.setPipeline(pl2);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(...grid);
    }
    for (let i = 0; i < p.pressureCycles; i++) vCycle(pass, 0);
    pass.setPipeline(pipes.proj);
    pass.setBindGroup(0, b.proj);
    pass.dispatchWorkgroups(...grid);
    pass.end();
  }

  const sim: VolumeSim = {
    time: 0,
    frame: 0,
    params: paramsUniform,
    setFields(f) {
      fields = f;
      rebuild();
    },
    reset(p) {
      time = 0;
      frame = 0;
      seed = 1;
      carry.clear();
      fields.reset(device.queue, p);
      sim.time = 0;
      sim.frame = 0;
    },
    step(p, enc, timing) {
      const n = p.substeps;
      const { buffer, writer: w } = ensureStaging(n);
      const plans: Plan[] = [];
      // Each substep advances dt; a frame covers substeps * dt of sim time.
      for (let s = 0; s < n; s++) {
        const pl = plan(p);
        packVolSubstep(w, s * VOL_SUBSTEP_LAYOUT.STRIDE, p, time + s * p.dt, seed++, pl.list);
        plans.push(pl);
      }
      device.queue.writeBuffer(buffer, 0, w.buffer, 0, n * VOL_SUBSTEP_LAYOUT.STRIDE);
      for (let s = 0; s < n; s++) {
        const writes: GPUComputePassTimestampWrites | undefined = timing && (s === 0 || s === n - 1)
          ? {
            querySet: timing.querySet,
            beginningOfPassWriteIndex: s === 0 ? timing.begin : undefined,
            endOfPassWriteIndex: s === n - 1 ? timing.end : undefined,
          }
          : undefined;
        recordSubstep(enc, p, buffer, s, plans[s], writes);
        swap();
      }
      time += n * p.dt;
      sim.time = time;
      sim.frame = ++frame;
    },
  };
  return sim;
}
