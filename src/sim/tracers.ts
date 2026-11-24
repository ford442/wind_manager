import { cellSize, type Params } from './params';
import { compileShaderModule, computePipeline, withValidationScope } from '../render/webgpu';

import commonSrc from '../shaders/common.wgsl?raw';
import tracersSrc from '../shaders/tracers.wgsl?raw';

export const TRACER_STRIDE = 32;
export const TRACER_POOL_CAP = 65536;
const TRACER_PARAM_BYTES = 48;
/** Staging slot stride (bytes); copy offsets must be 4-byte aligned. */
const TRACER_SLOT_BYTES = 48;

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
  /** Start a frame's recording: staging slots are reused from slot 0. */
  begin: () => void;
  /**
   * Record one tracer substep into `enc`. Its uniform comes from a staging slot
   * copied inside the encoder, so several substeps in one command buffer each
   * see their own seed. Call {@link TracerSystem.flush} before submitting.
   */
  step: (enc: GPUCommandEncoder, vel: GPUBuffer, solid: GPUBuffer, p: Params, seed: number) => void;
  /** Upload the staging slots recorded since `begin`. Call before `submit`. */
  flush: () => void;
  /** Seed a burst and submit immediately (sim reset, outside the frame loop). */
  seedBurst: (device: GPUDevice, vel: GPUBuffer, solid: GPUBuffer, p: Params, seed: number) => void;
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
    begin() {},
    step(_enc, _vel, _solid, _p, _seed) {},
    flush() {},
    seedBurst(_device, _vel, _solid, _p, _seed) {},
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
  // Per-frame staging ring: one slot per recorded pass, uploaded once by
  // `flush` and copied into `uniform` inside the encoder before each pass.
  let slotCap = 0;
  let slotsUsed = 0;
  let staging: GPUBuffer | null = null;
  let stagingData = new ArrayBuffer(0);
  let dv = new DataView(stagingData);
  function ensureSlot(): number {
    if (slotsUsed >= slotCap) {
      const cap = Math.max(8, slotCap * 2);
      const grown = new ArrayBuffer(cap * TRACER_SLOT_BYTES);
      new Uint8Array(grown).set(new Uint8Array(stagingData));
      // Slots already recorded this frame point at the old buffer; upload them
      // there before swapping so their copies still read the right bytes.
      if (staging && slotsUsed > 0) {
        device.queue.writeBuffer(staging, 0, stagingData, 0, slotsUsed * TRACER_SLOT_BYTES);
      }
      const old = staging;
      staging = device.createBuffer({
        label: 'tracer upload staging',
        size: cap * TRACER_SLOT_BYTES,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      // Destroy after the frame's submit has been scheduled.
      if (old) queueMicrotask(() => old.destroy());
      stagingData = grown;
      dv = new DataView(stagingData);
      slotCap = cap;
    }
    return slotsUsed++;
  }

  // Auto layouts only contain bindings an entry point uses: `seed_tracers`
  // has no `vel` (3) and `update_tracers` has no `slot_counter` (2). Both read
  // the solid mask (4), which a grid rebuild replaces.
  let seedBG: GPUBindGroup | null = null;
  let seedSolid: GPUBuffer | null = null;
  function seedBindGroup(solid: GPUBuffer): GPUBindGroup {
    if (!seedBG || seedSolid !== solid) {
      seedSolid = solid;
      seedBG = device.createBindGroup({
        label: 'seed_tracers',
        layout: seedPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: { buffer: tracers.pool } },
          { binding: 2, resource: { buffer: tracers.counter } },
          { binding: 4, resource: { buffer: solid } },
        ],
      });
    }
    return seedBG;
  }
  // `update_tracers` reads whichever vel buffer is current; the sim ping-pongs
  // two, so cache one bind group per vel buffer instead of rebuilding per call.
  const updateBGs = new Map<GPUBuffer, { group: GPUBindGroup; solid: GPUBuffer }>();
  function updateBindGroup(vel: GPUBuffer, solid: GPUBuffer): GPUBindGroup {
    const hit = updateBGs.get(vel);
    if (hit && hit.solid === solid) return hit.group;
    if (updateBGs.size >= 4) updateBGs.clear(); // grid rebuilds replace vel buffers
    const group = device.createBindGroup({
      label: 'update_tracers',
      layout: updatePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: tracers.pool } },
        { binding: 3, resource: { buffer: vel } },
        { binding: 4, resource: { buffer: solid } },
      ],
    });
    updateBGs.set(vel, { group, solid });
    return group;
  }

  function packTracerParams(slot: number, p: Params, spawnCount: number, seed: number) {
    const h = cellSize(p);
    const pool = tracerPoolSize(p);
    let o = slot * TRACER_SLOT_BYTES;
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

  function recordPass(
    enc: GPUCommandEncoder,
    vel: GPUBuffer,
    solid: GPUBuffer,
    p: Params,
    seed: number,
    spawnCount: number,
    update: boolean,
  ) {
    const pool = tracerPoolSize(p);
    const slot = ensureSlot();
    packTracerParams(slot, p, spawnCount, seed);
    // A slot copied inside the encoder, not `queue.writeBuffer(uniform)`: queue
    // writes all land before submit, so every pass would see the last seed.
    enc.copyBufferToBuffer(staging!, slot * TRACER_SLOT_BYTES, uniform, 0, TRACER_PARAM_BYTES);

    const pass = enc.beginComputePass({ label: 'tracers' });

    if (spawnCount > 0) {
      pass.setPipeline(seedPipe);
      pass.setBindGroup(0, seedBindGroup(solid));
      pass.dispatchWorkgroups(wg(spawnCount, 64));
    }

    if (update) {
      pass.setPipeline(updatePipe);
      pass.setBindGroup(0, updateBindGroup(vel, solid));
      pass.dispatchWorkgroups(wg(pool, 128));
    }

    pass.end();
  }

  tracers.begin = () => {
    slotsUsed = 0;
  };

  tracers.flush = () => {
    if (staging && slotsUsed > 0) {
      device.queue.writeBuffer(staging, 0, stagingData, 0, slotsUsed * TRACER_SLOT_BYTES);
    }
  };

  tracers.step = (enc, vel, solid, p, seed) => {
    if (!p.showTracers) return;
    const spawnCount = Math.max(
      0,
      Math.min(
        tracerPoolSize(p),
        Math.round((p.tracerDensity * p.dt) / Math.max(1, p.substeps)),
      ),
    );
    recordPass(enc, vel, solid, p, seed, spawnCount, true);
  };

  tracers.seedBurst = (device, vel, solid, p, seed) => {
    const pool = tracerPoolSize(p);
    const burst = Math.min(pool, Math.ceil(p.tracerDensity * p.tracerLifetime * 0.85));
    const enc = device.createCommandEncoder({ label: 'tracer burst' });
    tracers.begin();
    recordPass(enc, vel, solid, p, seed, burst, true);
    tracers.flush();
    device.queue.submit([enc.finish()]);
    tracers.begin();
  };
}
