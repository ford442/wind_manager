import type { Droplets } from '../sim/droplets';
import type { Fields } from '../sim/fields';
import { createReadbackBuffer } from './webgpu';

/**
 * Periodic CPU readback of field buffers for the stats panel (temperature range,
 * max speed, droplet counter). Runs on a throttled interval from `main.ts`.
 *
 * Flow: `copyBufferToBuffer` → `mapAsync(READ)` → typed array scan → `unmap`.
 * Buffers are resized when the grid (`nx×ny`) changes.
 */

export interface FieldDiagnosticsBuffers {
  cellCount: number;
  temperature: GPUBuffer;
  velocity: GPUBuffer;
  dropletCounter: GPUBuffer;
  resize: (device: GPUDevice, cellCount: number) => void;
  destroy: () => void;
}

export interface FieldDiagnosticsResult {
  minTemperature: number;
  maxTemperature: number;
  maxSpeedSquared: number;
  emitted: number;
  hasNaN: boolean;
}

export function createFieldDiagnosticsBuffers(
  device: GPUDevice,
  cellCount: number,
): FieldDiagnosticsBuffers {
  let n = cellCount;
  let temperature = createReadbackBuffer(device, n * 4);
  let velocity = createReadbackBuffer(device, n * 8);
  const dropletCounter = createReadbackBuffer(device, 4);

  return {
    get cellCount() {
      return n;
    },
    get temperature() {
      return temperature;
    },
    get velocity() {
      return velocity;
    },
    get dropletCounter() {
      return dropletCounter;
    },
    resize(device, cellCount) {
      if (cellCount === n) return;
      temperature.destroy();
      velocity.destroy();
      n = cellCount;
      temperature = createReadbackBuffer(device, n * 4);
      velocity = createReadbackBuffer(device, n * 8);
    },
    destroy() {
      temperature.destroy();
      velocity.destroy();
      dropletCounter.destroy();
    },
  };
}

function scanMappedFields(
  diagN: number,
  t: Float32Array,
  v: Float32Array,
): Pick<FieldDiagnosticsResult, 'minTemperature' | 'maxTemperature' | 'maxSpeedSquared' | 'hasNaN'> {
  let minTemperature = Infinity;
  let maxTemperature = -Infinity;
  let maxSpeedSquared = 0;
  let hasNaN = false;

  for (let i = 0; i < diagN; i++) {
    const ti = t[i];
    if (Number.isNaN(ti)) {
      hasNaN = true;
      break;
    }
    if (ti < minTemperature) minTemperature = ti;
    if (ti > maxTemperature) maxTemperature = ti;
    const vx = v[2 * i];
    const vy = v[2 * i + 1];
    if (Number.isNaN(vx) || Number.isNaN(vy)) {
      hasNaN = true;
      break;
    }
    const speedSquared = vx * vx + vy * vy;
    if (speedSquared > maxSpeedSquared) maxSpeedSquared = speedSquared;
  }

  return { minTemperature, maxTemperature, maxSpeedSquared, hasNaN };
}

/**
 * Copy temperature, velocity, and droplet-counter GPU buffers to staging memory,
 * scan for NaNs / extrema, and return summary stats for the HUD.
 */
export async function sampleFieldDiagnostics(
  device: GPUDevice,
  buffers: FieldDiagnosticsBuffers,
  fields: Fields,
  drops: Droplets,
): Promise<FieldDiagnosticsResult> {
  const diagN = buffers.cellCount;
  const encoder: GPUCommandEncoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(fields.T0, 0, buffers.temperature, 0, diagN * 4);
  encoder.copyBufferToBuffer(fields.vel0, 0, buffers.velocity, 0, diagN * 8);
  encoder.copyBufferToBuffer(drops.counter, 0, buffers.dropletCounter, 0, 4);
  device.queue.submit([encoder.finish()]);

  await Promise.all([
    buffers.temperature.mapAsync(GPUMapMode.READ),
    buffers.velocity.mapAsync(GPUMapMode.READ),
    buffers.dropletCounter.mapAsync(GPUMapMode.READ),
  ]);

  try {
    const temperatureView = new Float32Array(buffers.temperature.getMappedRange());
    const velocityView = new Float32Array(buffers.velocity.getMappedRange());
    const counterView = new Uint32Array(buffers.dropletCounter.getMappedRange());
    const emitted = counterView[0] ?? 0;
    const scan = scanMappedFields(diagN, temperatureView, velocityView);
    return { ...scan, emitted };
  } finally {
    buffers.temperature.unmap();
    buffers.velocity.unmap();
    buffers.dropletCounter.unmap();
  }
}
