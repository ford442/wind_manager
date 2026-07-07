import { cellSize, qSat, type Params } from '../sim/params';
import type { Fields } from '../sim/fields';

export interface VelSampler {
  scheduleRead: (device: GPUDevice, f: Fields, p: Params, enabled: boolean) => void;
  sampleVx: (worldX: number, worldY: number, p: Params) => number;
  sampleWind: (worldX: number, worldY: number, p: Params) => { vx: number; vy: number };
  sampleWet: (worldX: number, p: Params) => number;
  sampleSurfaceRh: (worldX: number, p: Params) => number;
  sampleQDep: (worldX: number, p: Params) => number;
  sampleGroundMoisture: (worldX: number, p: Params) => number;
  resize: (device: GPUDevice, p: Params) => void;
}

const SURFACE_ROWS = 2;

export function createVelSampler(device: GPUDevice, p: Params): VelSampler {
  let bandRows = p.ny;
  let bandBytes = p.nx * bandRows * 8;
  let wetBytes = p.nx * 4;
  let qDepBytes = p.nx * 4;
  let surfBytes = p.nx * SURFACE_ROWS * 4;
  let velBand: Float32Array | null = null;
  let wetBand: Float32Array | null = null;
  let qDepBand: Float32Array | null = null;
  let qSurfBand: Float32Array | null = null;
  let tSurfBand: Float32Array | null = null;
  let readPending = false;

  let stag = device.createBuffer({
    size: bandBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let wetStag = device.createBuffer({
    size: wetBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let qDepStag = device.createBuffer({
    size: qDepBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let qStag = device.createBuffer({
    size: surfBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let tStag = device.createBuffer({
    size: surfBytes,
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

  function sampleSurfField(
    band: Float32Array | null,
    worldX: number,
    params: Params,
  ): number {
    if (!band) return 0;
    const h = cellSize(params);
    const gx = worldX / h - 0.5;
    const i0 = Math.max(0, Math.min(params.nx - 2, Math.floor(gx)));
    const fx = gx - i0;
    const nx = params.nx;
    const v00 = band[i0];
    const v10 = band[i0 + 1];
    const v01 = band[nx + i0];
    const v11 = band[nx + i0 + 1];
    return (1 - fx) * ((1 - 0.55) * v00 + 0.55 * v01) + fx * ((1 - 0.55) * v10 + 0.55 * v11);
  }

  return {
    scheduleRead(gpu, f, params, enabled) {
      if (!enabled || readPending) return;
      readPending = true;
      const enc = gpu.createCommandEncoder();
      enc.copyBufferToBuffer(f.vel0, 0, stag, 0, bandBytes);
      enc.copyBufferToBuffer(f.wet, 0, wetStag, 0, wetBytes);
      enc.copyBufferToBuffer(f.qDep, 0, qDepStag, 0, qDepBytes);
      enc.copyBufferToBuffer(f.q0, 0, qStag, 0, surfBytes);
      enc.copyBufferToBuffer(f.T0, 0, tStag, 0, surfBytes);
      gpu.queue.submit([enc.finish()]);
      Promise.all([
        stag.mapAsync(GPUMapMode.READ),
        wetStag.mapAsync(GPUMapMode.READ),
        qDepStag.mapAsync(GPUMapMode.READ),
        qStag.mapAsync(GPUMapMode.READ),
        tStag.mapAsync(GPUMapMode.READ),
      ])
        .then(() => {
          const mapped = new Float32Array(stag.getMappedRange());
          velBand = new Float32Array(mapped);
          stag.unmap();
          const wetMapped = new Float32Array(wetStag.getMappedRange());
          wetBand = new Float32Array(wetMapped);
          wetStag.unmap();
          const qDepMapped = new Float32Array(qDepStag.getMappedRange());
          qDepBand = new Float32Array(qDepMapped);
          qDepStag.unmap();
          const qMapped = new Float32Array(qStag.getMappedRange());
          qSurfBand = new Float32Array(qMapped);
          qStag.unmap();
          const tMapped = new Float32Array(tStag.getMappedRange());
          tSurfBand = new Float32Array(tMapped);
          tStag.unmap();
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

    sampleQDep(worldX, params) {
      if (!qDepBand) return 0;
      const h = cellSize(params);
      const gx = worldX / h - 0.5;
      const i0 = Math.max(0, Math.min(params.nx - 2, Math.floor(gx)));
      const fx = gx - i0;
      return (1 - fx) * qDepBand[i0] + fx * qDepBand[i0 + 1];
    },

    sampleSurfaceRh(worldX, params) {
      const q = sampleSurfField(qSurfBand, worldX, params);
      const t = sampleSurfField(tSurfBand, worldX, params);
      const qs = qSat(t);
      return qs > 1e-6 ? Math.max(0, Math.min(1, q / qs)) : 0;
    },

    sampleGroundMoisture(worldX, params) {
      const wv = this.sampleWet(worldX, params);
      const qd = this.sampleQDep(worldX, params);
      const rhS = this.sampleSurfaceRh(worldX, params);
      const rhA = params.rhAmb / 100;
      const vapor = Math.max(0, (rhS - rhA) / Math.max(1 - rhA, 0.06));
      return Math.min(1, Math.max(wv, Math.max(qd * 0.95, vapor * 0.68)));
    },

    resize(gpu, params) {
      bandRows = params.ny;
      bandBytes = params.nx * bandRows * 8;
      wetBytes = params.nx * 4;
      qDepBytes = params.nx * 4;
      surfBytes = params.nx * SURFACE_ROWS * 4;
      stag.destroy();
      wetStag.destroy();
      qDepStag.destroy();
      qStag.destroy();
      tStag.destroy();
      stag = gpu.createBuffer({
        size: bandBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      wetStag = gpu.createBuffer({
        size: wetBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      qDepStag = gpu.createBuffer({
        size: qDepBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      qStag = gpu.createBuffer({
        size: surfBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      tStag = gpu.createBuffer({
        size: surfBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      velBand = null;
      wetBand = null;
      qDepBand = null;
      qSurfBand = null;
      tSurfBand = null;
      readPending = false;
    },
  };
}
