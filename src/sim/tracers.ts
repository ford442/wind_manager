import { cellSize, type Params } from './params';
import { compileShaderModule, computePipeline, withValidationScope } from '../render/webgpu';

import commonSrc from '../shaders/common.wgsl?raw';
import tracersSrc from '../shaders/tracers.wgsl?raw';

export const TRACER_STRIDE = 32;
export const TRACER_POOL_CAP = 65536;
const TRACER_PARAM_BYTES = 48;

export function tracerPoolSize(p: Params): number {
  return Math.min(
    TRACER_POOL_CAP,
    Math.max(4096, Math.ceil(p.tracerDensity * p.tracerLifetime * 1.2)),
  );
}

export interface TracerSystem {
  pool: GPUBuffer;
  counter: GPUBuffer;
  max: number;
  reset: (queue: GPUQueue) => void;
  destroy: () => void;
  step: (device: GPUDevice, vel: GPUBuffer, p: Params, seed: number) => void;
  seedBurst: (device: GPUDevice, vel: GPUBuffer, p: Params, seed: number) => void;
}

export async function createTracerSystem(device: GPUDevice): Promise<TracerSystem> {
  const usage =
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

  const pool = device.createBuffer({
    size: TRACER_POOL_CAP * TRACER_STRIDE,
    usage,
  });
  const counter = device.createBuffer({ size: 4, usage });

  const sys: TracerSystem = {
    pool,
    counter,
    max: TRACER_POOL_CAP,
    reset(queue) {
      queue.writeBuffer(pool, 0, new ArrayBuffer(TRACER_POOL_CAP * TRACER_STRIDE));
      queue.writeBuffer(counter, 0, new Uint32Array(1));
    },
    step(_device, _vel, _p, _seed) {},
    seedBurst(_device, _vel, _p, _seed) {},
    destroy() {
      pool.destroy();
      counter.destroy();
    },
  };

  await initTracerPipelines(device, sys);
  return sys;
}

async function initTracerPipelines(device: GPUDevice, tracers: TracerSystem): Promise<void> {
  const module = await compileShaderModule(device, 'tracers', [
    { name: 'common.wgsl', code: commonSrc },
    { name: 'tracers.wgsl', code: tracersSrc },
  ]);

  const { seedPipe, updatePipe } = await withValidationScope(device, 'tracers', () => ({
    seedPipe: computePipeline(device, {
      label: 'seed_tracers',
      layout: 'auto',
      compute: { module, entryPoint: 'seed_tracers' },
    }),
    updatePipe: computePipeline(device, {
      label: 'update_tracers',
      layout: 'auto',
      compute: { module, entryPoint: 'update_tracers' },
    }),
  }));

  const uniform = device.createBuffer({
    size: TRACER_PARAM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const paramData = new ArrayBuffer(TRACER_PARAM_BYTES);
  const dv = new DataView(paramData);

  // Auto layouts only contain bindings an entry point uses: `seed_tracers`
  // has no `vel` (3) and `update_tracers` has no `slot_counter` (2).
  const seedBG: GPUBindGroup = device.createBindGroup({
    label: 'seed_tracers',
    layout: seedPipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 1, resource: { buffer: tracers.pool } },
      { binding: 2, resource: { buffer: tracers.counter } },
    ],
  });
  // `update_tracers` reads whichever vel buffer is current; the sim ping-pongs
  // two, so cache one bind group per vel buffer instead of rebuilding per call.
  const updateBGs = new Map<GPUBuffer, GPUBindGroup>();
  function updateBindGroup(vel: GPUBuffer): GPUBindGroup {
    let bg = updateBGs.get(vel);
    if (!bg) {
      if (updateBGs.size >= 4) updateBGs.clear(); // grid rebuilds replace vel buffers
      bg = device.createBindGroup({
        label: 'update_tracers',
        layout: updatePipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: { buffer: tracers.pool } },
          { binding: 3, resource: { buffer: vel } },
        ],
      });
      updateBGs.set(vel, bg);
    }
    return bg;
  }

  function packTracerParams(p: Params, spawnCount: number, seed: number) {
    const h = cellSize(p);
    const pool = tracerPoolSize(p);
    let o = 0;
    const u32 = (v: number) => { dv.setUint32(o, v >>> 0, true); o += 4; };
    const f32 = (v: number) => { dv.setFloat32(o, v, true); o += 4; };
    u32(pool);
    u32(spawnCount);
    f32(p.tracerLifetime);
    f32(p.dt);
    u32(p.nx);
    u32(p.ny);
    f32(h);
    f32(p.domainW);
    f32(p.domainH);
    u32(seed);
    f32(0.1);
  }

  const wg = (n: number, size: number) => Math.ceil(n / size);

  function runPass(
    vel: GPUBuffer,
    p: Params,
    seed: number,
    spawnCount: number,
    update: boolean,
  ) {
    const pool = tracerPoolSize(p);
    packTracerParams(p, spawnCount, seed);
    device.queue.writeBuffer(uniform, 0, paramData);

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();

    if (spawnCount > 0) {
      pass.setPipeline(seedPipe);
      pass.setBindGroup(0, seedBG);
      pass.dispatchWorkgroups(wg(spawnCount, 64));
    }

    if (update) {
      pass.setPipeline(updatePipe);
      pass.setBindGroup(0, updateBindGroup(vel));
      pass.dispatchWorkgroups(wg(pool, 128));
    }

    pass.end();
    device.queue.submit([enc.finish()]);
  }

  tracers.step = (device, vel, p, seed) => {
    if (!p.showTracers) return;
    const spawnCount = Math.max(
      0,
      Math.min(
        tracerPoolSize(p),
        Math.round((p.tracerDensity * p.dt) / Math.max(1, p.substeps)),
      ),
    );
    runPass(vel, p, seed, spawnCount, true);
  };

  tracers.seedBurst = (device, vel, p, seed) => {
    const pool = tracerPoolSize(p);
    const burst = Math.min(pool, Math.ceil(p.tracerDensity * p.tracerLifetime * 0.85));
    runPass(vel, p, seed, burst, true);
  };
}
