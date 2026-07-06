// droplets.js — fixed droplet pool (ring-buffer emission, alive flag, no
// compaction in v1) plus the atomic emission counter.

export const DROPLET_STRIDE = 24; // pos vec2f + vel vec2f + radius f32 + alive u32

export function createDroplets(device, p) {
  const usage =
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

  const d = {
    max: p.maxDroplets,
    pool: device.createBuffer({ size: p.maxDroplets * DROPLET_STRIDE, usage }),
    counter: device.createBuffer({ size: 4, usage }),
  };

  d.reset = (queue) => {
    // alive = 0 everywhere; zero-fill does it.
    queue.writeBuffer(d.pool, 0, new ArrayBuffer(p.maxDroplets * DROPLET_STRIDE));
    queue.writeBuffer(d.counter, 0, new Uint32Array(1));
  };

  return d;
}
