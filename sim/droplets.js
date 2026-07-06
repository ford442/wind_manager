export const DROPLET_STRIDE = 24;

export function createDroplets(device, p) {
  const usage =
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

  const d = {
    max: p.maxDroplets,
    pool: device.createBuffer({ size: p.maxDroplets * DROPLET_STRIDE, usage }),
    counter: device.createBuffer({ size: 4, usage }),
  };

  d.reset = (queue) => {
    queue.writeBuffer(d.pool, 0, new ArrayBuffer(p.maxDroplets * DROPLET_STRIDE));
    queue.writeBuffer(d.counter, 0, new Uint32Array(1));
  };

  return d;
}
