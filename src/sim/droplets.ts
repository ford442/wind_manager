import type { Params } from './params';

export const DROPLET_STRIDE = 24;

export interface Droplets {
  max: number;
  pool: GPUBuffer;
  counter: GPUBuffer;
  reset: (queue: GPUQueue) => void;
  destroy: () => void;
}

export function createDroplets(device: GPUDevice, p: Params): Droplets {
  const usage =
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const max = p.maxDroplets;
  const pool = device.createBuffer({ size: max * DROPLET_STRIDE, usage });
  const counter = device.createBuffer({ size: 4, usage });

  const reset = (queue: GPUQueue): void => {
    queue.writeBuffer(pool, 0, new ArrayBuffer(max * DROPLET_STRIDE));
    queue.writeBuffer(counter, 0, new Uint32Array(1));
  };

  return {
    max,
    pool,
    counter,
    reset,
    destroy() {
      pool.destroy();
      counter.destroy();
    },
  };
}
