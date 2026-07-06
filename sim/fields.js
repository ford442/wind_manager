import { cellSize, qAmb } from './params.js';

export function createFields(device, p) {
  const n = p.nx * p.ny;
  const usage =
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const mk = (size) => device.createBuffer({ size, usage });

  const f = {
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

  f.reset = (queue, params) => {
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
  return f;
}
