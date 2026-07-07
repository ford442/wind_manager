import { cellSize, qAmb, type Params } from './params';

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
  accQ: GPUBuffer;
  accT: GPUBuffer;
  accM: GPUBuffer;
  wet: GPUBuffer;
  accWet: GPUBuffer;
  /** Slow-decay integral of surface humidity + splash deposition per ground column. */
  qDep: GPUBuffer;
  h: number;
  reset: (queue: GPUQueue, params: Params) => void;
}

export function createFields(device: GPUDevice, p: Params): Fields {
  const n = p.nx * p.ny;
  const usage =
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const mk = (size: number) => device.createBuffer({ size, usage });

  const vel0 = mk(n * 8);
  const vel1 = mk(n * 8);
  const T0 = mk(n * 4);
  const T1 = mk(n * 4);
  const q0 = mk(n * 4);
  const q1 = mk(n * 4);
  const p0 = mk(n * 4);
  const p1 = mk(n * 4);
  const div = mk(n * 4);
  const accQ = mk(n * 4);
  const accT = mk(n * 4);
  const accM = mk(n * 8);
  const wet = mk(p.nx * 4);
  const accWet = mk(p.nx * 4);
  const qDep = mk(p.nx * 4);
  const h = cellSize(p);

  const reset = (queue: GPUQueue, params: Params): void => {
    const zeros2 = new Float32Array(n * 2);
    const tInit = new Float32Array(n).fill(params.tAmb);
    const qInit = new Float32Array(n).fill(qAmb(params));
    const zeros = new Float32Array(n);
    queue.writeBuffer(vel0, 0, zeros2);
    queue.writeBuffer(vel1, 0, zeros2);
    queue.writeBuffer(T0, 0, tInit);
    queue.writeBuffer(T1, 0, tInit);
    queue.writeBuffer(q0, 0, qInit);
    queue.writeBuffer(q1, 0, qInit);
    queue.writeBuffer(p0, 0, zeros);
    queue.writeBuffer(p1, 0, zeros);
    queue.writeBuffer(div, 0, zeros);
    queue.writeBuffer(accQ, 0, new Int32Array(n));
    queue.writeBuffer(accT, 0, new Int32Array(n));
    queue.writeBuffer(accM, 0, new Int32Array(n * 2));
    queue.writeBuffer(wet, 0, new Float32Array(p.nx));
    queue.writeBuffer(accWet, 0, new Int32Array(p.nx));
    queue.writeBuffer(qDep, 0, new Float32Array(p.nx));
  };

  return {
    n,
    vel0,
    vel1,
    T0,
    T1,
    q0,
    q1,
    p0,
    p1,
    div,
    accQ,
    accT,
    accM,
    wet,
    accWet,
    qDep,
    h,
    reset,
  };
}
