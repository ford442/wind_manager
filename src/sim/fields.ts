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
  h: number;
  reset: (queue: GPUQueue, params: Params) => void;
}

export function createFields(device: GPUDevice, p: Params): Fields {
  const n = p.nx * p.ny;
  const usage =
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const mk = (size) => device.createBuffer({ size, usage });

  const f: any = {
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
    accQ: mk(n * 4),
    accT: mk(n * 4),
    accM: mk(n * 8),
  };

  f.reset = (queue: GPUQueue, params: Params) => {
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
    queue.writeBuffer(f.accQ, 0, new Int32Array(n));
    queue.writeBuffer(f.accT, 0, new Int32Array(n));
    queue.writeBuffer(f.accM, 0, new Int32Array(n * 2));
  };

  f.h = cellSize(p);
  return f as Fields;
}
