import type { Params } from '../sim/params';
import { DROPLET_STRIDE } from '../sim/droplets';
import { TRACER_POOL_CAP, TRACER_STRIDE } from '../sim/tracers';
import { GRID_MAX_NX, GRID_MAX_NY } from '../sim/advanced';

/**
 * Adapter + device setup with explicit limits.
 *
 * The shaders are written against specific limits (e.g. `droplets.wgsl`
 * `update` binds 8 storage buffers). We read `adapter.limits`, fail at startup
 * naming the first limit the adapter cannot meet, and request exactly what we
 * use so a device never silently runs on defaults that a later shader exceeds.
 *
 * Core WebGPU only — no `featureLevel: 'compatibility'`; the sim needs storage
 * atomics in compute and more storage buffers than compat guarantees.
 */

export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  /** True when the device was created with `timestamp-query`. */
  timestampQuery: boolean;
}

/** Thrown when the adapter cannot satisfy a limit the shaders depend on. */
export class GpuLimitError extends Error {
  constructor(
    readonly limit: keyof GPUSupportedLimits & string,
    readonly required: number,
    readonly available: number,
  ) {
    super(
      `GPU adapter does not meet required limit ${limit}: needs ${required}, adapter supports ${available}.`,
    );
    this.name = 'GpuLimitError';
  }
}

type NumericLimit =
  | 'maxStorageBuffersPerShaderStage'
  | 'maxStorageBufferBindingSize'
  | 'maxBufferSize'
  | 'maxUniformBuffersPerShaderStage'
  | 'maxBindingsPerBindGroup'
  | 'maxComputeWorkgroupsPerDimension'
  | 'maxComputeInvocationsPerWorkgroup'
  | 'maxComputeWorkgroupSizeX';

/** Largest single storage buffer the app allocates, in bytes. */
export function largestStorageBufferBytes(p: Params): number {
  const droplets = p.maxDroplets * DROPLET_STRIDE;
  const tracers = TRACER_POOL_CAP * TRACER_STRIDE;
  // vel / accM fields are vec2 (8 bytes per cell) at the largest allowed grid.
  const field = GRID_MAX_NX * GRID_MAX_NY * 8;
  return Math.max(droplets, tracers, field);
}

/** Limits the shaders and buffer sizes in this app depend on. */
export function requiredLimits(p: Params): Record<NumericLimit, number> {
  const bytes = largestStorageBufferBytes(p);
  return {
    // droplets.wgsl `update`: drops, vel, tf, qf, acc_qt, solid, acc_m, acc_wet.
    // This is the WebGPU default; keep it there (q/T scatter share one buffer
    // and wall wetness rides in acc_wet so the solid mask fits).
    maxStorageBuffersPerShaderStage: 8,
    maxStorageBufferBindingSize: bytes,
    maxBufferSize: bytes,
    // render.wgsl / sim shaders: Params + per-emitter uniforms.
    maxUniformBuffersPerShaderStage: 2,
    // Highest binding index used is 10 (emitter params in droplets.wgsl).
    maxBindingsPerBindGroup: 11,
    // `update` dispatch over the droplet pool with workgroup_size(128).
    maxComputeWorkgroupsPerDimension: Math.ceil(p.maxDroplets / 128),
    maxComputeInvocationsPerWorkgroup: 128,
    maxComputeWorkgroupSizeX: 128,
  };
}

export async function requestGpuDevice(p: Params): Promise<GpuContext> {
  if (!navigator.gpu) {
    throw new Error(
      'This browser does not expose WebGPU (navigator.gpu is missing). ' +
        'Use a recent Chrome/Edge, or enable WebGPU in your browser.',
    );
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    throw new Error('No WebGPU adapter available on this machine.');
  }

  const want = requiredLimits(p);
  const limits: Record<string, number> = {};
  for (const [name, required] of Object.entries(want) as [NumericLimit, number][]) {
    const available = adapter.limits[name];
    if (available < required) throw new GpuLimitError(name, required, available);
    limits[name] = required;
  }

  const requiredFeatures: GPUFeatureName[] = [];
  const timestampQuery = adapter.features.has('timestamp-query');
  if (timestampQuery) requiredFeatures.push('timestamp-query');

  const device = await adapter.requestDevice({
    label: 'wind_manager',
    requiredLimits: limits,
    requiredFeatures,
  });
  return { adapter, device, timestampQuery };
}
