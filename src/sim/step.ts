import { effectiveEmitRate } from './burst';
import type { Params } from './params';
import { MAX_EMITTERS } from './emitters';
import type { Fields } from './fields';
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
  step: (p: Params) => void;
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
  /** Indexed by `jacobiIters % 2`: final pressure in p0 (even) or p1 (odd). */
  proj: [GPUBindGroup, GPUBindGroup];
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
 * 2. Each substep's encoder starts with `copyBufferToBuffer` from its slot into
 *    the live `Params` uniform and the live `EmitterParams` array.
 * 3. All water emitters emit in **one** `emit` dispatch and all air emitters in
 *    one `air_inject` dispatch; `workgroup_id.y` selects the emitter row.
 *
 * ## Bind groups
 *
 * vel/T/q ping-pong each substep and p0/p1 ping-pong inside Jacobi. Both
 * parities (and both Jacobi endings for `project`) are built once when the
 * buffer set changes (`setFields`), and the step picks one by buffer identity.
 * Nothing is allocated inside the substep loop.
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
    jacobi: pipe(modPressure, 'jacobi'),
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
  let jacobiA: GPUBindGroup;
  let jacobiB: GPUBindGroup;
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
        { binding: 1, resource: { buffer: vel0 } },
        { binding: 3, resource: { buffer: pFinal } },
      ]);
    return {
      update: bg(pipes.update, 'update', [
        U,
        { binding: 1, resource: { buffer: drops.pool } },
        { binding: 3, resource: { buffer: vel0 } },
        { binding: 4, resource: { buffer: T0 } },
        { binding: 5, resource: { buffer: q0 } },
        { binding: 6, resource: { buffer: fields.accQ } },
        { binding: 7, resource: { buffer: fields.accT } },
        { binding: 8, resource: { buffer: fields.accM } },
        { binding: 9, resource: { buffer: fields.accWet } },
      ]),
      apply: bg(pipes.apply, 'apply', [
        U,
        { binding: 1, resource: { buffer: vel0 } },
        { binding: 2, resource: { buffer: T0 } },
        { binding: 3, resource: { buffer: q0 } },
        { binding: 4, resource: { buffer: fields.accQ } },
        { binding: 5, resource: { buffer: fields.accT } },
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
      ]),
      // NOTE: buoyancy / divergence / project / boundaries bind the advection
      // *source* (vel0/T0/q0), matching the bindings before the recording-model
      // change. After the swap the advected buffer is current, so this work does
      // not reach the next substep. Kept as-is here (no physics change); tracked
      // separately.
      buoy: bg(pipes.buoy, 'buoyancy', [
        U,
        { binding: 1, resource: { buffer: vel0 } },
        { binding: 2, resource: { buffer: T0 } },
        { binding: 3, resource: { buffer: q0 } },
      ]),
      bound: bg(pipes.bound, 'boundaries', [
        U,
        { binding: 1, resource: { buffer: vel0 } },
        { binding: 2, resource: { buffer: T0 } },
        { binding: 3, resource: { buffer: q0 } },
      ]),
      div: bg(pipes.div, 'divergence', [
        U,
        { binding: 1, resource: { buffer: vel0 } },
        { binding: 2, resource: { buffer: fields.div } },
        { binding: 4, resource: { buffer: fields.p0 } },
      ]),
      proj: [projFor(fields.p0, 'project p0'), projFor(fields.p1, 'project p1')],
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
      entries: [U, EP, { binding: 8, resource: { buffer: fields.accM } }],
    });
    jacobiA = device.createBindGroup({
      label: 'jacobi p0→p1',
      layout: pipes.jacobi.getBindGroupLayout(0),
      entries: [
        U,
        { binding: 2, resource: { buffer: fields.div } },
        { binding: 3, resource: { buffer: fields.p0 } },
        { binding: 4, resource: { buffer: fields.p1 } },
      ],
    });
    jacobiB = device.createBindGroup({
      label: 'jacobi p1→p0',
      layout: pipes.jacobi.getBindGroupLayout(0),
      entries: [
        U,
        { binding: 2, resource: { buffer: fields.div } },
        { binding: 3, resource: { buffer: fields.p1 } },
        { binding: 4, resource: { buffer: fields.p0 } },
      ],
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
    p: Params,
    stagingBuffer: GPUBuffer,
    slot: number,
    plan: SubstepPlan,
  ): GPUCommandBuffer {
    const enc: GPUCommandEncoder = device.createCommandEncoder({ label: 'sim substep' });
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
    pass.dispatchWorkgroups(wg(p.nx, 64));

    const grid2d: Workgroup2D = wg2d(p.nx, p.ny, 8, 8);

    pass.setPipeline(pipes.advect);
    pass.setBindGroup(0, bg.advect);
    pass.dispatchWorkgroups(...grid2d);

    pass.setPipeline(pipes.buoy);
    pass.setBindGroup(0, bg.buoy);
    pass.dispatchWorkgroups(...grid2d);

    pass.setPipeline(pipes.div);
    pass.setBindGroup(0, bg.div);
    pass.dispatchWorkgroups(...grid2d);

    pass.setPipeline(pipes.jacobi);
    for (let i = 0; i < p.jacobiIters; i++) {
      pass.setBindGroup(0, i % 2 === 0 ? jacobiA : jacobiB);
      pass.dispatchWorkgroups(...grid2d);
    }

    pass.setPipeline(pipes.proj);
    pass.setBindGroup(0, bg.proj[p.jacobiIters % 2]);
    pass.dispatchWorkgroups(...grid2d);

    pass.setPipeline(pipes.bound);
    pass.setBindGroup(0, bg.bound);
    pass.dispatchWorkgroups(...grid2d);

    pass.end();
    return enc.finish();
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
        tracers.seedBurst(device, fields.vel0, p, seed++);
      }
    },

    step(p: Params): void {
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
      // One queue write for the whole frame, before any substep is submitted.
      // Each substep copies its own slot inside its command buffer.
      device.queue.writeBuffer(buffer, 0, writer.buffer, 0, substeps * SUBSTEP_UPLOAD_LAYOUT.STRIDE);

      for (let s = 0; s < substeps; s++) {
        device.queue.submit([recordSubstep(p, buffer, s, plans[s])]);
        if (tracers && p.showTracers) {
          tracers.step(device, fields.vel0, p, seed++);
        }
        swapAdvectBuffers();
        time += p.dt;
      }
      sim.time = time;
      sim.frame = ++frame;
    },
  };

  return sim;
}
