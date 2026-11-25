import { effectiveEmitRate } from './burst';
import type { Params } from './params';
import { MAX_EMITTERS } from './emitters';
import { SOLID_WET_SLOTS, type Fields } from './fields';
import type { Droplets } from './droplets';
import type { TracerSystem } from './tracers';
import {
  compileShaderModule,
  computePipeline,
  withValidationScope,
  workgroups1d,
  workgroups2d,
  type WgslPart,
  type Workgroup2D,
} from '../render/webgpu';
import {
  EMITTER_ARRAY_BYTES,
  SIM_PARAMS_LAYOUT,
  SUBSTEP_UPLOAD_LAYOUT,
  packSubstepUpload,
  type EmitterDispatch,
} from './simParamsUniform';
import { UniformWriter } from './uniformWriter';
import { PRESSURE_SMOOTHS, PRESSURE_COARSE_SMOOTHS } from './pressure';

// Static shader imports via Vite (inlined at build/dev time)
import commonSrc from '../shaders/common.wgsl?raw';
import dropletsSrc from '../shaders/droplets.wgsl?raw';
import applyScatterSrc from '../shaders/apply_scatter.wgsl?raw';
import advectSrc from '../shaders/advect.wgsl?raw';
import buoyancySrc from '../shaders/buoyancy.wgsl?raw';
import pressureSrc from '../shaders/pressure.wgsl?raw';

export interface Sim {
  time: number;
  frame: number;
  getFields: () => Fields;
  rebuildBindGroups: () => void;
  setFields: (f: Fields) => void;
  reset: (p: Params) => void;
  /**
   * Record every substep of one frame into `enc` (nothing is submitted here).
   * Uniform/staging uploads are issued with `queue.writeBuffer` before return,
   * so the caller just finishes and submits `enc` once.
   */
  step: (p: Params, enc: GPUCommandEncoder) => void;
}

/** Bind groups that depend on which half of the vel/T/q ping-pong is current. */
interface ParityBindGroups {
  update: GPUBindGroup;
  apply: GPUBindGroup;
  applyWet: GPUBindGroup;
  advect: GPUBindGroup;
  buoy: GPUBindGroup;
  bound: GPUBindGroup;
  div: GPUBindGroup;
  proj: GPUBindGroup;
}

/**
 * Build the GPU simulation: compute pipelines for droplets, scatter, advection,
 * buoyancy, and pressure projection.
 *
 * ## Recording model
 *
 * Uniform data is uploaded **inside the command encoder**, never with
 * `queue.writeBuffer` on a live binding while a pass is being recorded (queue
 * writes all land before `submit`, so every dispatch in the pass would see the
 * last value written). Each frame:
 *
 * 1. Every substep's global `Params` and its full emitter array are packed into
 *    one slot of a staging buffer, and the staging buffer is written once.
 * 2. Each substep starts with `copyBufferToBuffer` from its slot into the live
 *    `Params` uniform and the live `EmitterParams` array. All substeps (and the
 *    tracer passes) are recorded into the caller's encoder: one command buffer
 *    per frame, shared with the environment and field render passes.
 * 3. All water emitters emit in **one** `emit` dispatch and all air emitters in
 *    one `air_inject` dispatch; `workgroup_id.y` selects the emitter row.
 *
 * ## Bind groups
 *
 * vel/T/q ping-pong each substep and p0/p1 ping-pong inside each smoother. Scatter,
 * droplets and advection read the current half (`vel0`/`T0`/`q0`); advection
 * writes the other half, and buoyancy, the pressure solve and the boundary pass
 * work on that advected half, which the swap then makes current. Every pass
 * that touches velocity or scalars also binds the solid mask. Both
 * parities and multigrid bindings are built once when the
 * buffer set changes (`setFields`), and the step picks one by buffer identity.
 * No GPU resources are allocated inside the substep loop.
 */
export async function createSim(
  device: GPUDevice,
  fields: Fields,
  drops: Droplets,
  tracers: TracerSystem | null,
  _params: Params,
): Promise<Sim> {
  const generated: WgslPart = {
    name: 'generated',
    code: `const MAX_EMITTERS : u32 = ${MAX_EMITTERS}u;`,
  };
  const mod = (src: string, name: string): Promise<GPUShaderModule> =>
    compileShaderModule(device, name, [
      generated,
      { name: 'common.wgsl', code: commonSrc },
      { name: `${name}.wgsl`, code: src },
    ]);

  const [modDrops, modApply, modAdvect, modBuoy, modPressure] = await Promise.all([
    mod(dropletsSrc, 'droplets'),
    mod(applyScatterSrc, 'apply_scatter'),
    mod(advectSrc, 'advect'),
    mod(buoyancySrc, 'buoyancy'),
    mod(pressureSrc, 'pressure'),
  ]);

  const pipe = (module: GPUShaderModule, entryPoint: string): GPUComputePipeline =>
    computePipeline(device, {
      label: entryPoint,
      layout: 'auto',
      compute: { module, entryPoint },
    });

  const pipes = await withValidationScope(device, 'sim compute', () => ({
    emit: pipe(modDrops, 'emit'),
    air: pipe(modDrops, 'air_inject'),
    update: pipe(modDrops, 'update'),
    apply: pipe(modApply, 'apply'),
    applyWet: pipe(modApply, 'apply_wet'),
    advect: pipe(modAdvect, 'advect'),
    buoy: pipe(modBuoy, 'buoyancy'),
    bound: pipe(modBuoy, 'boundaries'),
    div: pipe(modPressure, 'divergence'),
    smooth: pipe(modPressure, 'pressure_smooth'),
    restrict: pipe(modPressure, 'restrict_residual'),
    prolong: pipe(modPressure, 'prolongate'),
    proj: pipe(modPressure, 'project'),
  }));

  const paramsUniform: GPUBuffer = device.createBuffer({
    label: 'sim params',
    size: SIM_PARAMS_LAYOUT.BYTE_LENGTH,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const emitterUniform: GPUBuffer = device.createBuffer({
    label: 'sim emitter params',
    size: EMITTER_ARRAY_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Staging ring: one slot per substep in the frame. Grows if substeps rises.
  let stagingSlots = 0;
  let staging: GPUBuffer | null = null;
  let stagingWriter: UniformWriter | null = null;
  function ensureStaging(slots: number): { buffer: GPUBuffer; writer: UniformWriter } {
    if (!staging || !stagingWriter || slots > stagingSlots) {
      staging?.destroy();
      stagingSlots = Math.max(slots, 4);
      const size = stagingSlots * SUBSTEP_UPLOAD_LAYOUT.STRIDE;
      staging = device.createBuffer({
        label: 'sim upload staging',
        size,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      stagingWriter = new UniformWriter(size);
    }
    return { buffer: staging, writer: stagingWriter };
  }

  let emitBG: GPUBindGroup;
  let airBG: GPUBindGroup;
  let multigrid: {
    smooth: [GPUBindGroup, GPUBindGroup];
    restrict?: GPUBindGroup;
    prolong?: GPUBindGroup;
  }[];
  /** [0]: built with the current `fields.vel0` as read side; [1]: swapped. */
  let parity: [ParityBindGroups, ParityBindGroups];
  /** Buffer that is `fields.vel0` when parity set 0 applies. */
  let parity0Vel: GPUBuffer;

  const U = { binding: 0, resource: { buffer: paramsUniform } };
  const EP = { binding: 10, resource: { buffer: emitterUniform } };

  function buildParity(
    vel0: GPUBuffer,
    vel1: GPUBuffer,
    T0: GPUBuffer,
    T1: GPUBuffer,
    q0: GPUBuffer,
    q1: GPUBuffer,
  ): ParityBindGroups {
    const bg = (p: GPUComputePipeline, label: string, entries: GPUBindGroupEntry[]) =>
      device.createBindGroup({ label, layout: p.getBindGroupLayout(0), entries });
    const projFor = (pFinal: GPUBuffer, label: string) =>
      bg(pipes.proj, label, [
        U,
        { binding: 1, resource: { buffer: vel1 } },
        { binding: 3, resource: { buffer: pFinal } },
        { binding: 5, resource: { buffer: fields.solid } },
      ]);
    return {
      update: bg(pipes.update, 'update', [
        U,
        { binding: 1, resource: { buffer: drops.pool } },
        { binding: 3, resource: { buffer: vel0 } },
        { binding: 4, resource: { buffer: T0 } },
        { binding: 5, resource: { buffer: q0 } },
        { binding: 6, resource: { buffer: fields.accQT } },
        { binding: 7, resource: { buffer: fields.solid } },
        { binding: 8, resource: { buffer: fields.accM } },
        { binding: 9, resource: { buffer: fields.accWet } },
      ]),
      apply: bg(pipes.apply, 'apply', [
        U,
        { binding: 1, resource: { buffer: vel0 } },
        { binding: 2, resource: { buffer: T0 } },
        { binding: 3, resource: { buffer: q0 } },
        { binding: 4, resource: { buffer: fields.accQT } },
        { binding: 5, resource: { buffer: fields.solid } },
        { binding: 6, resource: { buffer: fields.accM } },
      ]),
      applyWet: bg(pipes.applyWet, 'apply_wet', [
        U,
        { binding: 3, resource: { buffer: q0 } },
        { binding: 7, resource: { buffer: fields.wet } },
        { binding: 8, resource: { buffer: fields.accWet } },
        { binding: 9, resource: { buffer: fields.qDep } },
      ]),
      advect: bg(pipes.advect, 'advect', [
        U,
        { binding: 1, resource: { buffer: vel0 } },
        { binding: 2, resource: { buffer: T0 } },
        { binding: 3, resource: { buffer: q0 } },
        { binding: 4, resource: { buffer: vel1 } },
        { binding: 5, resource: { buffer: T1 } },
        { binding: 6, resource: { buffer: q1 } },
        { binding: 7, resource: { buffer: fields.solid } },
      ]),
      // Buoyancy, the pressure solve and the boundary pass act on the advected
      // half (vel1/T1/q1), which becomes current after the swap. (They used to
      // bind the advection source, so the next advection overwrote their work.)
      buoy: bg(pipes.buoy, 'buoyancy', [
        U,
        { binding: 1, resource: { buffer: vel1 } },
        { binding: 2, resource: { buffer: T1 } },
        { binding: 3, resource: { buffer: q1 } },
        { binding: 4, resource: { buffer: fields.solid } },
      ]),
      bound: bg(pipes.bound, 'boundaries', [
        U,
        { binding: 1, resource: { buffer: vel1 } },
        { binding: 2, resource: { buffer: T1 } },
        { binding: 3, resource: { buffer: q1 } },
        { binding: 4, resource: { buffer: fields.solid } },
      ]),
      div: bg(pipes.div, 'divergence', [
        U,
        { binding: 1, resource: { buffer: vel1 } },
        { binding: 2, resource: { buffer: fields.div } },
        { binding: 4, resource: { buffer: fields.p0 } },
        { binding: 5, resource: { buffer: fields.solid } },
      ]),
      proj: projFor(fields.p0, 'project p0'),
    };
  }

  function rebuildBindGroups(): void {
    emitBG = device.createBindGroup({
      label: 'emit',
      layout: pipes.emit.getBindGroupLayout(0),
      entries: [
        U,
        EP,
        { binding: 1, resource: { buffer: drops.pool } },
        { binding: 2, resource: { buffer: drops.counter } },
      ],
    });
    airBG = device.createBindGroup({
      label: 'air_inject',
      layout: pipes.air.getBindGroupLayout(0),
      entries: [
        U,
        EP,
        { binding: 7, resource: { buffer: fields.solid } },
        { binding: 8, resource: { buffer: fields.accM } },
      ],
    });
    const bufferEntry = (binding: number, buffer: GPUBuffer): GPUBindGroupEntry =>
      ({ binding, resource: { buffer } });
    const bind = (pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[]) =>
      device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    multigrid = fields.pressure.map((level, i) => {
      const base = [bufferEntry(6, level.uniform), bufferEntry(7, level.graph)];
      const coarse = fields.pressure[i + 1];
      return {
        smooth: [
          bind(pipes.smooth, [...base, bufferEntry(2, level.rhs),
            bufferEntry(3, level.p0), bufferEntry(4, level.p1)]),
          bind(pipes.smooth, [...base, bufferEntry(2, level.rhs),
            bufferEntry(3, level.p1), bufferEntry(4, level.p0)]),
        ],
        restrict: coarse && bind(pipes.restrict, [...base, bufferEntry(2, level.rhs),
          bufferEntry(3, level.p0), bufferEntry(9, coarse.rhs), bufferEntry(10, coarse.p0)]),
        prolong: coarse && bind(pipes.prolong, [...base, bufferEntry(4, level.p0),
          bufferEntry(8, coarse.p0)]),
      };
    });
    const { vel0, vel1, T0, T1, q0, q1 } = fields;
    parity = [
      buildParity(vel0, vel1, T0, T1, q0, q1),
      buildParity(vel1, vel0, T1, T0, q1, q0),
    ];
    parity0Vel = vel0;
  }

  rebuildBindGroups();

  /** Swap read/write halves. Bind groups for both halves already exist. */
  function swapAdvectBuffers(): void {
    const tv: GPUBuffer = fields.vel0;
    fields.vel0 = fields.vel1;
    fields.vel1 = tv;
    const tt: GPUBuffer = fields.T0;
    fields.T0 = fields.T1;
    fields.T1 = tt;
    const tq: GPUBuffer = fields.q0;
    fields.q0 = fields.q1;
    fields.q1 = tq;
  }

  let time = 0;
  let frame = 0;
  let seed = 1;

  const wg = workgroups1d;
  const wg2d = workgroups2d;

  function vCycle(pass: GPUComputePassEncoder, index: number): void {
    const level = fields.pressure[index];
    const bg = multigrid[index];
    const grid = wg2d(level.nx, level.ny, 8, 8);
    const smooth = (count: number) => {
      pass.setPipeline(pipes.smooth);
      for (let i = 0; i < count; i++) {
        pass.setBindGroup(0, bg.smooth[i % 2]);
        pass.dispatchWorkgroups(...grid);
      }
    };
    if (!bg.restrict || !bg.prolong) {
      smooth(PRESSURE_COARSE_SMOOTHS);
      return;
    }
    smooth(PRESSURE_SMOOTHS);
    pass.setPipeline(pipes.restrict);
    pass.setBindGroup(0, bg.restrict);
    pass.dispatchWorkgroups(...wg2d(level.nx / 2, level.ny / 2, 8, 8));
    vCycle(pass, index + 1);
    pass.setPipeline(pipes.prolong);
    pass.setBindGroup(0, bg.prolong);
    pass.dispatchWorkgroups(...grid);
    smooth(PRESSURE_SMOOTHS);
  }

  /** Per-substep dispatch plan, filled during packing. */
  interface SubstepPlan {
    waterCount: number;
    airCount: number;
    entries: number;
    maxWater: number;
    maxAir: number;
  }

  function planEmitters(p: Params): { list: EmitterDispatch[]; plan: SubstepPlan } {
    const list: EmitterDispatch[] = [];
    const plan: SubstepPlan = { waterCount: 0, airCount: 0, entries: 0, maxWater: 0, maxAir: 0 };
    for (const em of p.emitters) {
      if (list.length >= MAX_EMITTERS) break;
      const count = Math.max(0, Math.round((effectiveEmitRate(p, em) * p.dt) / p.substeps));
      if (count <= 0) continue;
      list.push({ emitter: em, count, seed: seed++ });
      if (em.type === 'water') {
        plan.waterCount++;
        plan.maxWater = Math.max(plan.maxWater, count);
      } else {
        plan.airCount++;
        plan.maxAir = Math.max(plan.maxAir, count);
      }
    }
    plan.entries = list.length;
    return { list, plan };
  }

  function recordSubstep(
    enc: GPUCommandEncoder,
    p: Params,
    stagingBuffer: GPUBuffer,
    slot: number,
    plan: SubstepPlan,
  ): void {
    const slotBase = slot * SUBSTEP_UPLOAD_LAYOUT.STRIDE;
    enc.copyBufferToBuffer(
      stagingBuffer,
      slotBase + SUBSTEP_UPLOAD_LAYOUT.params,
      paramsUniform,
      0,
      SIM_PARAMS_LAYOUT.BYTE_LENGTH,
    );
    enc.copyBufferToBuffer(
      stagingBuffer,
      slotBase + SUBSTEP_UPLOAD_LAYOUT.emitters,
      emitterUniform,
      0,
      EMITTER_ARRAY_BYTES,
    );

    const bg = fields.vel0 === parity0Vel ? parity[0] : parity[1];
    const pass: GPUComputePassEncoder = enc.beginComputePass({ label: 'sim substep' });

    if (plan.waterCount > 0) {
      pass.setPipeline(pipes.emit);
      pass.setBindGroup(0, emitBG);
      pass.dispatchWorkgroups(wg(plan.maxWater, 64), plan.entries);
    }
    if (plan.airCount > 0) {
      pass.setPipeline(pipes.air);
      pass.setBindGroup(0, airBG);
      pass.dispatchWorkgroups(wg(plan.maxAir, 64), plan.entries);
    }

    pass.setPipeline(pipes.update);
    pass.setBindGroup(0, bg.update);
    pass.dispatchWorkgroups(wg(p.maxDroplets, 128));

    pass.setPipeline(pipes.apply);
    pass.setBindGroup(0, bg.apply);
    pass.dispatchWorkgroups(wg(fields.n, 64));

    pass.setPipeline(pipes.applyWet);
    pass.setBindGroup(0, bg.applyWet);
    pass.dispatchWorkgroups(wg(p.nx + SOLID_WET_SLOTS, 64));

    const grid2d: Workgroup2D = wg2d(p.nx, p.ny, 8, 8);

    pass.setPipeline(pipes.advect);
    pass.setBindGroup(0, bg.advect);
    pass.dispatchWorkgroups(...grid2d);

    pass.setPipeline(pipes.buoy);
    pass.setBindGroup(0, bg.buoy);
    pass.dispatchWorkgroups(...grid2d);

    // Apply edge/wall conditions before projection, never after it.
    pass.setPipeline(pipes.bound);
    pass.setBindGroup(0, bg.bound);
    pass.dispatchWorkgroups(...grid2d);

    pass.setPipeline(pipes.div);
    pass.setBindGroup(0, bg.div);
    pass.dispatchWorkgroups(...grid2d);

    for (let i = 0; i < p.pressureCycles; i++) vCycle(pass, 0);

    pass.setPipeline(pipes.proj);
    pass.setBindGroup(0, bg.proj);
    pass.dispatchWorkgroups(...grid2d);

    pass.end();
  }

  const sim: Sim = {
    time: 0,
    frame: 0,
    getFields: () => fields,
    rebuildBindGroups,

    setFields(f: Fields): void {
      fields = f;
      rebuildBindGroups();
    },

    reset(p: Params): void {
      time = 0;
      frame = 0;
      seed = 1;
      p.burstRemaining = 0;
      fields.reset(device.queue, p);
      drops.reset(device.queue);
      tracers?.reset(device.queue);
      if (tracers && p.showTracers) {
        tracers.seedBurst(device, fields.vel0, fields.solid, p, seed++);
      }
    },

    step(p: Params, enc: GPUCommandEncoder): void {
      const substeps = p.substeps;
      const { buffer, writer } = ensureStaging(substeps);
      const plans: SubstepPlan[] = [];
      let t = time;
      for (let s = 0; s < substeps; s++) {
        const { list, plan } = planEmitters(p);
        packSubstepUpload(writer, s * SUBSTEP_UPLOAD_LAYOUT.STRIDE, {
          params: p,
          simTime: t,
          seed: seed++,
          emitters: list,
        });
        plans.push(plan);
        t += p.dt;
      }
      // One queue write for the whole frame, before the frame is submitted.
      // Each substep copies its own slot inside the encoder.
      device.queue.writeBuffer(buffer, 0, writer.buffer, 0, substeps * SUBSTEP_UPLOAD_LAYOUT.STRIDE);

      tracers?.begin();
      for (let s = 0; s < substeps; s++) {
        recordSubstep(enc, p, buffer, s, plans[s]);
        swapAdvectBuffers();
        // Tracers ride the velocity this substep produced.
        if (tracers && p.showTracers) {
          tracers.step(enc, fields.vel0, fields.solid, p, seed++);
        }
        time += p.dt;
      }
      tracers?.flush();
      sim.time = time;
      sim.frame = ++frame;
    },
  };

  return sim;
}
