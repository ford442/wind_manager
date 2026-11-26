import { volumeGrid, volumeQAmb, volumeSpacing, type VolumeParams } from './params';
import { cellCount, type VolumeGrid, VOL_BYTES } from './grid';
import { rasterizeVolumeSolids, type VolumeSolids } from './solids';
import { volPressureHierarchy } from './pressure';

export interface VolPressureLevel {
  nx: number;
  ny: number;
  nz: number;
  p0: GPUBuffer;
  p1: GPUBuffer;
  rhs: GPUBuffer;
  graph: GPUBuffer;
  uniform: GPUBuffer;
}

/**
 * GPU buffers of the yard volume. Sizes are exactly the rows of
 * `volumeBufferTable` (src/volume/grid.ts), which `main.ts` checks against the
 * adapter before calling this.
 */
export interface VolumeFields {
  grid: VolumeGrid;
  n: number;
  vel0: GPUBuffer;
  vel1: GPUBuffer;
  T0: GPUBuffer;
  T1: GPUBuffer;
  q0: GPUBuffer;
  q1: GPUBuffer;
  accQT: GPUBuffer;
  accM: GPUBuffer;
  /** Ground wetness per floor cell (`k * nx + i`) and its accumulator. */
  wet: GPUBuffer;
  accWet: GPUBuffer;
  solid: GPUBuffer;
  solids: VolumeSolids;
  pressure: VolPressureLevel[];
  drops: GPUBuffer;
  dropCounter: GPUBuffer;
  reset: (queue: GPUQueue, p: VolumeParams) => void;
  writeSolids: (queue: GPUQueue, p: VolumeParams) => void;
  destroy: () => void;
}

export function createVolumeFields(device: GPUDevice, p: VolumeParams): VolumeFields {
  const grid = volumeGrid(p);
  const n = cellCount(grid);
  const floor = grid.nx * grid.nz;
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const all: GPUBuffer[] = [];
  const mk = (label: string, size: number) => {
    const b = device.createBuffer({ label, size, usage });
    all.push(b);
    return b;
  };
  let solids = rasterizeVolumeSolids(grid, p.solidObstacles);
  const h = volumeSpacing(p);
  const pressure = volPressureHierarchy(grid.nx, grid.ny, grid.nz, h, solids.mask).map((lv, i) => {
    const c = lv.nx * lv.ny * lv.nz;
    const graph = mk(`vol graph L${i}`, lv.nodes.byteLength);
    const uniform = device.createBuffer({
      label: `vol level L${i}`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    all.push(uniform);
    device.queue.writeBuffer(graph, 0, lv.nodes);
    device.queue.writeBuffer(uniform, 0, new Uint32Array([lv.nx, lv.ny, lv.nz, 0]));
    return {
      nx: lv.nx, ny: lv.ny, nz: lv.nz,
      p0: mk(`vol p0 L${i}`, c * 4), p1: mk(`vol p1 L${i}`, c * 4), rhs: mk(`vol rhs L${i}`, c * 4),
      graph, uniform,
    };
  });

  const f: VolumeFields = {
    grid,
    n,
    vel0: mk('vol vel0', n * VOL_BYTES.vel),
    vel1: mk('vol vel1', n * VOL_BYTES.vel),
    T0: mk('vol T0', n * 4),
    T1: mk('vol T1', n * 4),
    q0: mk('vol q0', n * 4),
    q1: mk('vol q1', n * 4),
    accQT: mk('vol accQT', n * VOL_BYTES.accQT),
    accM: mk('vol accM', n * VOL_BYTES.accM),
    wet: mk('vol wet', floor * 4),
    accWet: mk('vol accWet', floor * 4),
    solid: mk('vol solid', n * 4),
    solids,
    pressure,
    drops: mk('vol droplets', p.maxDroplets * VOL_BYTES.droplet),
    dropCounter: mk('vol drop counter', 4),
    reset,
    writeSolids,
    destroy: () => { for (const b of all) b.destroy(); },
  };

  function writeSolids(queue: GPUQueue, params: VolumeParams): void {
    if (volumeGrid(params) !== grid) throw new Error('writeSolids: grid changed; rebuild the fields');
    solids = rasterizeVolumeSolids(grid, params.solidObstacles);
    f.solids = solids;
    queue.writeBuffer(f.solid, 0, solids.mask);
    volPressureHierarchy(grid.nx, grid.ny, grid.nz, h, solids.mask).forEach((lv, i) => {
      queue.writeBuffer(pressure[i].graph, 0, lv.nodes);
    });
  }

  function reset(queue: GPUQueue, params: VolumeParams): void {
    const t = new Float32Array(n).fill(params.tAmb);
    const q = new Float32Array(n).fill(volumeQAmb(params));
    for (const b of [f.vel0, f.vel1]) queue.writeBuffer(b, 0, new Float32Array(n * 4));
    for (const b of [f.T0, f.T1]) queue.writeBuffer(b, 0, t);
    for (const b of [f.q0, f.q1]) queue.writeBuffer(b, 0, q);
    queue.writeBuffer(f.accQT, 0, new Int32Array(n * 2));
    queue.writeBuffer(f.accM, 0, new Int32Array(n * 3));
    queue.writeBuffer(f.wet, 0, new Float32Array(floor));
    queue.writeBuffer(f.accWet, 0, new Int32Array(floor));
    for (const lv of pressure) {
      const z = new Float32Array(lv.nx * lv.ny * lv.nz);
      queue.writeBuffer(lv.p0, 0, z);
      queue.writeBuffer(lv.p1, 0, z);
      queue.writeBuffer(lv.rhs, 0, z);
    }
    queue.writeBuffer(f.drops, 0, new ArrayBuffer(params.maxDroplets * VOL_BYTES.droplet));
    queue.writeBuffer(f.dropCounter, 0, new Uint32Array(1));
    writeSolids(queue, params);
  }

  return f;
}
