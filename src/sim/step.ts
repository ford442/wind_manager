import { effectiveEmitRate } from './burst';
import type { Params } from './params';
import type { Emitter } from './emitters';
import type { Fields } from './fields';
import type { Droplets } from './droplets';
import type { TracerSystem } from './tracers';
import {
  computePipeline,
  shaderModule,
  workgroups1d,
  workgroups2d,
  type Workgroup2D,
} from '../render/webgpu';
import { concatWgsl } from '../shaders/wgsl';
import { SIM_PARAMS_LAYOUT, SimParamsUniform } from './simParamsUniform';

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

/**
 * Build the GPU simulation: compute pipelines for droplets, scatter, advection,
 * buoyancy, and pressure projection. Uniform params are packed each emitter
 * dispatch via {@link SimParamsUniform} (84-byte `Params` struct).
 *
 * `fields` vel/T/q buffers are ping-ponged each substep; bind groups are rebuilt
 * after every swap so bindings always reference the current read/write pair.
 */
export async function createSim(
  device: GPUDevice,
  fields: Fields,
  drops: Droplets,
  tracers: TracerSystem | null,
  _params: Params,
): Promise<Sim> {
  const common = commonSrc;

  const mod = (src: typeof dropletsSrc, label: string): GPUShaderModule =>
    shaderModule(device, label, concatWgsl(common, src));

  const modDrops = mod(dropletsSrc, 'droplets');
  const modApply = mod(applyScatterSrc, 'apply_scatter');
  const modAdvect = mod(advectSrc, 'advect');
  const modBuoy = mod(buoyancySrc, 'buoyancy');
  const modPressure = mod(pressureSrc, 'pressure');

  const pipe = (module: GPUShaderModule, entryPoint: string): GPUComputePipeline =>
    computePipeline(device, {
      label: entryPoint,
      layout: 'auto',
      compute: { module, entryPoint },
    });

  const emitPipe: GPUComputePipeline = pipe(modDrops, 'emit');
  const airPipe: GPUComputePipeline = pipe(modDrops, 'air_inject');
  const updatePipe: GPUComputePipeline = pipe(modDrops, 'update');
  const applyPipe: GPUComputePipeline = pipe(modApply, 'apply');
  const applyWetPipe: GPUComputePipeline = pipe(modApply, 'apply_wet');
  const advectPipe: GPUComputePipeline = pipe(modAdvect, 'advect');
  const buoyPipe: GPUComputePipeline = pipe(modBuoy, 'buoyancy');
  const boundPipe: GPUComputePipeline = pipe(modBuoy, 'boundaries');
  const divPipe: GPUComputePipeline = pipe(modPressure, 'divergence');
  const jacobiPipe: GPUComputePipeline = pipe(modPressure, 'jacobi');
  const projPipe: GPUComputePipeline = pipe(modPressure, 'project');

  const uniform: GPUBuffer = device.createBuffer({
    size: SIM_PARAMS_LAYOUT.BYTE_LENGTH,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const simParamsUniform = new SimParamsUniform();

  let emitBG: GPUBindGroup;
  let airBG: GPUBindGroup;
  let updateBG: GPUBindGroup;
  let applyBG: GPUBindGroup;
  let applyWetBG: GPUBindGroup;
  let advectBG: GPUBindGroup;
  let buoyBG: GPUBindGroup;
  let boundBG: GPUBindGroup;
  let divBG: GPUBindGroup;
  let jacobiA: GPUBindGroup;
  let jacobiB: GPUBindGroup;
  let projBG: GPUBindGroup;

  function rebuildBindGroups(): void {
    emitBG = device.createBindGroup({
      layout: emitPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: drops.pool } },
        { binding: 2, resource: { buffer: drops.counter } },
      ],
    });
    airBG = device.createBindGroup({
      layout: airPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 8, resource: { buffer: fields.accM } },
      ],
    });
    updateBG = device.createBindGroup({
      layout: updatePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: drops.pool } },
        { binding: 3, resource: { buffer: fields.vel0 } },
        { binding: 4, resource: { buffer: fields.T0 } },
        { binding: 5, resource: { buffer: fields.q0 } },
        { binding: 6, resource: { buffer: fields.accQ } },
        { binding: 7, resource: { buffer: fields.accT } },
        { binding: 8, resource: { buffer: fields.accM } },
        { binding: 9, resource: { buffer: fields.accWet } },
      ],
    });
    applyBG = device.createBindGroup({
      layout: applyPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: fields.vel0 } },
        { binding: 2, resource: { buffer: fields.T0 } },
        { binding: 3, resource: { buffer: fields.q0 } },
        { binding: 4, resource: { buffer: fields.accQ } },
        { binding: 5, resource: { buffer: fields.accT } },
        { binding: 6, resource: { buffer: fields.accM } },
      ],
    });
    applyWetBG = device.createBindGroup({
      layout: applyWetPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 3, resource: { buffer: fields.q0 } },
        { binding: 7, resource: { buffer: fields.wet } },
        { binding: 8, resource: { buffer: fields.accWet } },
        { binding: 9, resource: { buffer: fields.qDep } },
      ],
    });
    advectBG = device.createBindGroup({
      layout: advectPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: fields.vel0 } },
        { binding: 2, resource: { buffer: fields.T0 } },
        { binding: 3, resource: { buffer: fields.q0 } },
        { binding: 4, resource: { buffer: fields.vel1 } },
        { binding: 5, resource: { buffer: fields.T1 } },
        { binding: 6, resource: { buffer: fields.q1 } },
      ],
    });
    buoyBG = device.createBindGroup({
      layout: buoyPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: fields.vel0 } },
        { binding: 2, resource: { buffer: fields.T0 } },
        { binding: 3, resource: { buffer: fields.q0 } },
      ],
    });
    boundBG = device.createBindGroup({
      layout: boundPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: fields.vel0 } },
        { binding: 2, resource: { buffer: fields.T0 } },
        { binding: 3, resource: { buffer: fields.q0 } },
      ],
    });
    divBG = device.createBindGroup({
      layout: divPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: fields.vel0 } },
        { binding: 2, resource: { buffer: fields.div } },
        { binding: 4, resource: { buffer: fields.p0 } },
      ],
    });
    jacobiA = device.createBindGroup({
      layout: jacobiPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 2, resource: { buffer: fields.div } },
        { binding: 3, resource: { buffer: fields.p0 } },
        { binding: 4, resource: { buffer: fields.p1 } },
      ],
    });
    jacobiB = device.createBindGroup({
      layout: jacobiPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 2, resource: { buffer: fields.div } },
        { binding: 3, resource: { buffer: fields.p1 } },
        { binding: 4, resource: { buffer: fields.p0 } },
      ],
    });
    projBG = device.createBindGroup({
      layout: projPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: fields.vel0 } },
        { binding: 3, resource: { buffer: fields.p0 } },
      ],
    });
  }

  rebuildBindGroups();

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
    rebuildBindGroups();
  }

  function packParams(
    p: Params,
    em: Emitter,
    emitCount: number,
    seed: number,
    simTime: number,
  ): ArrayBuffer {
    return simParamsUniform.pack({ params: p, emitter: em, emitCount, seed, simTime });
  }

  let time = 0;
  let frame = 0;
  let seed = 1;

  const wg = workgroups1d;
  const wg2d = workgroups2d;

  function substep(p: Params): void {
    const enc: GPUCommandEncoder = device.createCommandEncoder();
    const pass: GPUComputePassEncoder = enc.beginComputePass();

    for (const em of p.emitters) {
      const count = Math.max(0, Math.round((effectiveEmitRate(p, em) * p.dt) / p.substeps));
      if (count <= 0) continue;
      const paramData = packParams(p, em, count, seed++, time);
      device.queue.writeBuffer(uniform, 0, paramData);
      if (em.type === 'water') {
        pass.setPipeline(emitPipe);
        pass.setBindGroup(0, emitBG);
        pass.dispatchWorkgroups(wg(count, 64));
      } else {
        pass.setPipeline(airPipe);
        pass.setBindGroup(0, airBG);
        pass.dispatchWorkgroups(wg(count, 64));
      }
    }

    pass.setPipeline(updatePipe);
    pass.setBindGroup(0, updateBG);
    pass.dispatchWorkgroups(wg(p.maxDroplets, 128));

    pass.setPipeline(applyPipe);
    pass.setBindGroup(0, applyBG);
    pass.dispatchWorkgroups(wg(fields.n, 64));

    pass.setPipeline(applyWetPipe);
    pass.setBindGroup(0, applyWetBG);
    pass.dispatchWorkgroups(wg(p.nx, 64));

    const grid2d: Workgroup2D = wg2d(p.nx, p.ny, 8, 8);

    pass.setPipeline(advectPipe);
    pass.setBindGroup(0, advectBG);
    pass.dispatchWorkgroups(...grid2d);

    pass.setPipeline(buoyPipe);
    pass.setBindGroup(0, buoyBG);
    pass.dispatchWorkgroups(...grid2d);

    pass.setPipeline(divPipe);
    pass.setBindGroup(0, divBG);
    pass.dispatchWorkgroups(...grid2d);

    for (let i = 0; i < p.jacobiIters; i++) {
      pass.setPipeline(jacobiPipe);
      pass.setBindGroup(0, i % 2 === 0 ? jacobiA : jacobiB);
      pass.dispatchWorkgroups(...grid2d);
    }

    const pFinal: GPUBuffer = p.jacobiIters % 2 === 0 ? fields.p0 : fields.p1;
    if (p.jacobiIters % 2 !== 0) {
      projBG = device.createBindGroup({
        layout: projPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: { buffer: fields.vel0 } },
          { binding: 3, resource: { buffer: pFinal } },
        ],
      });
    }

    pass.setPipeline(projPipe);
    pass.setBindGroup(0, projBG);
    pass.dispatchWorkgroups(...grid2d);

    pass.setPipeline(boundPipe);
    pass.setBindGroup(0, boundBG);
    pass.dispatchWorkgroups(...grid2d);

    pass.end();
    device.queue.submit([enc.finish()]);

    if (tracers && p.showTracers) {
      tracers.step(device, fields.vel0, p, seed++);
    }

    swapAdvectBuffers();
    time += p.dt;
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
      rebuildBindGroups();
      if (tracers && p.showTracers) {
        tracers.seedBurst(device, fields.vel0, p, seed++);
      }
    },

    step(p: Params): void {
      for (let s = 0; s < p.substeps; s++) {
        substep(p);
      }
      sim.time = time;
      sim.frame = ++frame;
    },
  };

  return sim;
}
