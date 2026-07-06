import type { Params } from './params';

export const DROPLET_STRIDE = 24;

export interface Droplets {
  max: number;
  pool: GPUBuffer;
  counter: GPUBuffer;
  reset: (queue: GPUQueue) => void;
}

export function createDroplets(device: GPUDevice, p: Params): Droplets {
  const usage =
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

  const d: any = {
    max: p.maxDroplets,
    pool: device.createBuffer({ size: p.maxDroplets * DROPLET_STRIDE, usage }),
    counter: device.createBuffer({ size: 4, usage }),
  };

  d.reset = (queue: GPUQueue) => {
    queue.writeBuffer(d.pool, 0, new ArrayBuffer(p.maxDroplets * DROPLET_STRIDE));
    queue.writeBuffer(d.counter, 0, new Uint32Array(1));
  };

  return d as unknown as Droplets;
}
