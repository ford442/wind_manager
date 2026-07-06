import { cellSize, type Params } from '../sim/params';
import type { Fields } from '../sim/fields';

export interface VelSampler {
  scheduleRead: (device: GPUDevice, f: Fields, p: Params, enabled: boolean) => void;
  sampleVx: (worldX: number, worldY: number, p: Params) => number;
  sampleWind: (worldX: number, worldY: number, p: Params) => { vx: number; vy: number };
  sampleWet: (worldX: number, p: Params) => number;
  resize: (device: GPUDevice, p: Params) => void;
}

export function createVelSampler(device: GPUDevice, p: Params): VelSampler {
  let bandRows = p.ny;
  let bandBytes = p.nx * bandRows * 8;
  let wetBytes = p.nx * 4;
  let velBand: Float32Array | null = null;
  let wetBand: Float32Array | null = null;
  let readPending = false;

  let stag = device.createBuffer({
    size: bandBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let wetStag = device.createBuffer({
    size: wetBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  function rowIndex(worldY: number, params: Params): number {
    const h = cellSize(params);
    return Math.max(0, Math.min(bandRows - 1, Math.floor(worldY / h - 0.5)));
  }

  function sampleAtRowCol(row: number, worldX: number, params: Params): { vx: number; vy: number } {
    if (!velBand) return { vx: 0, vy: 0 };
    const h = cellSize(params);
    const gx = worldX / h - 0.5;
    const i0 = Math.max(0, Math.min(params.nx - 2, Math.floor(gx)));
    const fx = gx - i0;
    const base = 2 * (row * params.nx + i0);
    const v00x = velBand[base];
    const v00y = velBand[base + 1];
    const v10x = velBand[base + 2];
    const v10y = velBand[base + 3];
    return {
      vx: (1 - fx) * v00x + fx * v10x,
      vy: (1 - fx) * v00y + fx * v10y,
    };
  }

  return {
    scheduleRead(gpu, f, params, enabled) {
      if (!enabled || readPending) return;
      readPending = true;
      const enc = gpu.createCommandEncoder();
      enc.copyBufferToBuffer(f.vel0, 0, stag, 0, bandBytes);
      enc.copyBufferToBuffer(f.wet, 0, wetStag, 0, wetBytes);
      gpu.queue.submit([enc.finish()]);
      Promise.all([stag.mapAsync(GPUMapMode.READ), wetStag.mapAsync(GPUMapMode.READ)])
        .then(() => {
          const mapped = new Float32Array(stag.getMappedRange());
          velBand = new Float32Array(mapped);
          stag.unmap();
          const wetMapped = new Float32Array(wetStag.getMappedRange());
          wetBand = new Float32Array(wetMapped);
          wetStag.unmap();
          readPending = false;
        })
        .catch(() => {
          readPending = false;
        });
    },

    sampleVx(worldX, worldY, params) {
      const row = rowIndex(worldY, params);
      return sampleAtRowCol(row, worldX, params).vx;
    },

    sampleWind(worldX, worldY, params) {
      const row = rowIndex(worldY, params);
      return sampleAtRowCol(row, worldX, params);
    },

    sampleWet(worldX, params) {
      if (!wetBand) return 0;
      const h = cellSize(params);
      const gx = worldX / h - 0.5;
      const i0 = Math.max(0, Math.min(params.nx - 2, Math.floor(gx)));
      const fx = gx - i0;
      return (1 - fx) * wetBand[i0] + fx * wetBand[i0 + 1];
    },

    resize(gpu, params) {
      bandRows = params.ny;
      bandBytes = params.nx * bandRows * 8;
      wetBytes = params.nx * 4;
      stag.destroy();
      wetStag.destroy();
      stag = gpu.createBuffer({
        size: bandBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      wetStag = gpu.createBuffer({
        size: wetBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      velBand = null;
      wetBand = null;
      readPending = false;
    },
  };
}
