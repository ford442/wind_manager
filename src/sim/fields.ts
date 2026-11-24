import { cellSize, qAmb, type Params } from './params';
import { N_SOLIDS, N_SOLID_FACES, rasterizeSolids, type SolidLayout } from './solids';

/** Wetness slots after the `nx` ground columns: one per solid face. */
export const SOLID_WET_SLOTS = N_SOLIDS * N_SOLID_FACES;

export interface Fields {
  n: number;
  vel0: GPUBuffer;
  vel1: GPUBuffer;
  T0: GPUBuffer;
  T1: GPUBuffer;
  q0: GPUBuffer;
  q1: GPUBuffer;
  p0: GPUBuffer;
  p1: GPUBuffer;
  div: GPUBuffer;
  /** Scatter accumulator for humidity and temperature, interleaved `[q, T]` per cell. */
  accQT: GPUBuffer;
  accM: GPUBuffer;
  /**
   * Surface wetness: `nx` ground columns, then {@link SOLID_WET_SLOTS} solid
   * faces (`nx + id * 2 + face`, face 0 = left wall, 1 = right wall).
   */
  wet: GPUBuffer;
  /** Fixed-point accumulator with the same layout as {@link Fields.wet}. */
  accWet: GPUBuffer;
  /** Slow-decay integral of surface humidity + splash deposition per ground column. */
  qDep: GPUBuffer;
  /** Solid mask, one `u32` per cell: 0 fluid, `id + 1` inside solid `id` (see `sim/solids.ts`). */
  solid: GPUBuffer;
  /**
   * CPU copy of what `solid` holds. Every {@link Fields.writeSolids} replaces
   * the object, so the scene can compare identity to re-upload the boxes it draws.
   */
  solids: SolidLayout;
  h: number;
  reset: (queue: GPUQueue, params: Params) => void;
  /** Re-rasterize the mask from the layout and `params` (domain, grid, toggle). */
  writeSolids: (queue: GPUQueue, params: Params) => void;
  destroy: () => void;
}

export function createFields(device: GPUDevice, p: Params): Fields {
  const n = p.nx * p.ny;
  const usage =
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const mk = (size: number) => device.createBuffer({ size, usage });
  const wetLen = p.nx + SOLID_WET_SLOTS;

  const fields: Fields = {
    n,
    vel0: mk(n * 8),
    vel1: mk(n * 8),
    T0: mk(n * 4),
    T1: mk(n * 4),
    q0: mk(n * 4),
    q1: mk(n * 4),
    p0: mk(n * 4),
    p1: mk(n * 4),
    div: mk(n * 4),
    accQT: mk(n * 8),
    accM: mk(n * 8),
    wet: mk(wetLen * 4),
    accWet: mk(wetLen * 4),
    qDep: mk(p.nx * 4),
    solid: mk(n * 4),
    solids: rasterizeSolids(p),
    h: cellSize(p),
    reset,
    writeSolids,
    destroy,
  };

  function writeSolids(queue: GPUQueue, params: Params): void {
    if (params.nx * params.ny !== n) {
      throw new Error('writeSolids: grid size changed; rebuild Fields first');
    }
    fields.solids = rasterizeSolids(params);
    queue.writeBuffer(fields.solid, 0, fields.solids.mask);
  }

  function reset(queue: GPUQueue, params: Params): void {
    const f = fields;
    const zeros2 = new Float32Array(n * 2);
    const tInit = new Float32Array(n).fill(params.tAmb);
    const qInit = new Float32Array(n).fill(qAmb(params));
    const zeros = new Float32Array(n);
    queue.writeBuffer(f.vel0, 0, zeros2);
    queue.writeBuffer(f.vel1, 0, zeros2);
    queue.writeBuffer(f.T0, 0, tInit);
    queue.writeBuffer(f.T1, 0, tInit);
    queue.writeBuffer(f.q0, 0, qInit);
    queue.writeBuffer(f.q1, 0, qInit);
    queue.writeBuffer(f.p0, 0, zeros);
    queue.writeBuffer(f.p1, 0, zeros);
    queue.writeBuffer(f.div, 0, zeros);
    queue.writeBuffer(f.accQT, 0, new Int32Array(n * 2));
    queue.writeBuffer(f.accM, 0, new Int32Array(n * 2));
    queue.writeBuffer(f.wet, 0, new Float32Array(wetLen));
    queue.writeBuffer(f.accWet, 0, new Int32Array(wetLen));
    queue.writeBuffer(f.qDep, 0, new Float32Array(params.nx));
    // Domain size and the obstacle toggle can change without a grid rebuild.
    writeSolids(queue, params);
  }

  function destroy(): void {
    const f = fields;
    for (const b of [
      f.vel0, f.vel1, f.T0, f.T1, f.q0, f.q1, f.p0, f.p1, f.div,
      f.accQT, f.accM, f.wet, f.accWet, f.qDep, f.solid,
    ]) {
      b.destroy();
    }
  }

  return fields;
}
