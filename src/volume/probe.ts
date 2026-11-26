import { createReadbackBuffer, readMappedBuffer } from '../render/webgpu';
import type { VolumeFields } from './fields';
import { heroTree, volumeSpacing, type VolumeParams } from './params';
import { volProjectedDivergence } from './pressure';

/**
 * On-demand field readback for the HUD (about once a second) and the headless
 * acceptance run. Never every frame: the renderer reads the GPU buffers
 * directly. Everything here is measured, nothing is drawn from it.
 */
export interface VolumeProbe {
  time: number;
  finite: boolean;
  minT: number;
  maxSpeed: number;
  maxDivergence: number;
  /**
   * Outflow front from the first water emitter along the ground (mean of the
   * two bottom rows): furthest distance (m) in each direction where the
   * outward velocity component exceeds {@link FRONT_SPEED}.
   */
  front: { px: number; mx: number; pz: number; mz: number };
  /** Outward ground speed (m/s) 2 m from the source in +x and +z. */
  outflow2m: { px: number; pz: number };
  /** Coldest ground deficit (K below ambient) within 1.5 m of the source. */
  coolingAtSource: number;
  /**
   * Hero tree, along the unit vector from the source to the tree: ground wind
   * (m/s, mean of the two bottom rows) and wind at 1 m beside the trunk, and
   * crown sway (m). Positive = pushed away from the source (the outflow);
   * negative = drawn toward it (entrainment).
   */
  tree: { ground: number; wind: number; sway: number };
  /** |u| inside solid cells, and the largest velocity into a solid face from its fluid neighbour. */
  solid: { cells: number; maxInside: number; maxIntoFace: number };
  /** Live parcels in the pool. */
  liveDrops: number;
}

export const FRONT_SPEED = 0.15;

export interface ProbeBuffers {
  vel: GPUBuffer;
  t: GPUBuffer;
  solid: GPUBuffer;
  state: GPUBuffer;
  drops: GPUBuffer;
  destroy: () => void;
}

export function createProbeBuffers(device: GPUDevice, f: VolumeFields, maxDroplets: number): ProbeBuffers {
  const b = {
    vel: createReadbackBuffer(device, f.n * 16),
    t: createReadbackBuffer(device, f.n * 4),
    solid: createReadbackBuffer(device, f.n * 4),
    state: createReadbackBuffer(device, 5 * 4 * 4),
    drops: createReadbackBuffer(device, maxDroplets * 32),
  };
  return { ...b, destroy: () => Object.values(b).forEach((x) => x.destroy()) };
}

export async function probeVolume(
  device: GPUDevice, bufs: ProbeBuffers, f: VolumeFields, treeState: GPUBuffer, p: VolumeParams, time: number,
): Promise<VolumeProbe> {
  const enc = device.createCommandEncoder({ label: 'vol probe' });
  enc.copyBufferToBuffer(f.vel0, 0, bufs.vel, 0, f.n * 16);
  enc.copyBufferToBuffer(f.T0, 0, bufs.t, 0, f.n * 4);
  enc.copyBufferToBuffer(f.solid, 0, bufs.solid, 0, f.n * 4);
  enc.copyBufferToBuffer(treeState, 0, bufs.state, 0, 5 * 16);
  enc.copyBufferToBuffer(f.drops, 0, bufs.drops, 0, p.maxDroplets * 32);
  device.queue.submit([enc.finish()]);

  const [vel, T, solid, state, drops] = await Promise.all([
    readMappedBuffer(bufs.vel, (m) => new Float32Array(m.slice(0))),
    readMappedBuffer(bufs.t, (m) => new Float32Array(m.slice(0))),
    readMappedBuffer(bufs.solid, (m) => new Uint32Array(m.slice(0))),
    readMappedBuffer(bufs.state, (m) => new Float32Array(m.slice(0))),
    readMappedBuffer(bufs.drops, (m) => new Float32Array(m.slice(0))),
  ]);
  return analyse(vel, T, solid, state, drops, f, p, time);
}

export function analyse(
  vel: Float32Array, T: Float32Array, solid: Uint32Array, state: Float32Array, drops: Float32Array,
  f: VolumeFields, p: VolumeParams, time: number,
): VolumeProbe {
  const { nx, ny, nz } = f.grid;
  const h = volumeSpacing(p);
  const idx = (i: number, j: number, k: number) => (k * ny + j) * nx + i;
  let finite = true;
  let minT = Infinity;
  let maxSpeed = 0;
  for (let c = 0; c < f.n; c++) {
    const vx = vel[c * 4], vy = vel[c * 4 + 1], vz = vel[c * 4 + 2];
    if (!Number.isFinite(vx + vy + vz + T[c])) { finite = false; continue; }
    minT = Math.min(minT, T[c]);
    maxSpeed = Math.max(maxSpeed, Math.hypot(vx, vy, vz));
  }

  const src = p.emitters.find((e) => e.type === 'water') ?? p.emitters[0];
  const si = src ? Math.min(nx - 1, Math.floor(src.x / h[0])) : Math.floor(nx / 2);
  const sk = src ? Math.min(nz - 1, Math.floor(src.z / h[2])) : Math.floor(nz / 2);
  /** Ground-layer velocity component `axis` at (i, k): mean of rows 0 and 1. */
  const ground = (i: number, k: number, axis: number) =>
    0.5 * (vel[idx(i, 0, k) * 4 + axis] + vel[idx(i, 1, k) * 4 + axis]);
  const isSolid = (i: number, j: number, k: number) =>
    i >= 0 && j >= 0 && k >= 0 && i < nx && j < ny && k < nz && solid[idx(i, j, k)] !== 0;

  /** Furthest ground cell along a ray from the source whose outward speed exceeds FRONT_SPEED. */
  function front(di: number, dk: number): number {
    const axis = di !== 0 ? 0 : 2;
    const sign = di !== 0 ? di : dk;
    const step = di !== 0 ? h[0] : h[2];
    let last = 0;
    for (let s = 1; ; s++) {
      const i = si + di * s;
      const k = sk + dk * s;
      if (i < 0 || k < 0 || i >= nx || k >= nz) break;
      if (isSolid(i, 0, k)) continue;
      if (sign * ground(i, k, axis) > FRONT_SPEED) last = s;
    }
    return last * step;
  }
  const at = (dist: number, di: number, dk: number) => {
    const s = Math.round(dist / (di !== 0 ? h[0] : h[2]));
    const i = Math.max(0, Math.min(nx - 1, si + di * s));
    const k = Math.max(0, Math.min(nz - 1, sk + dk * s));
    return (di !== 0 ? di : dk) * ground(i, k, di !== 0 ? 0 : 2);
  };

  let cooling = 0;
  const r = 1.5;
  for (let k = 0; k < nz; k++) for (let i = 0; i < nx; i++) {
    if (Math.hypot((i - si) * h[0], (k - sk) * h[2]) > r) continue;
    cooling = Math.max(cooling, p.tAmb - T[idx(i, 0, k)]);
  }

  const tree = heroTree();
  const tj = Math.min(ny - 1, Math.floor(1 / h[1]));
  const tk = Math.min(nz - 1, Math.floor(tree.z / h[2]));
  // Upstream of the trunk (−x, toward the default hose), two cells clear of
  // the bark: the cell touching it has its wall-normal velocity clamped.
  const besideTrunk = (j: number) => {
    let i = Math.min(nx - 1, Math.floor(tree.x / h[0]));
    while (i > 0 && isSolid(i, j, tk)) i--;
    return Math.max(0, i - 1);
  };
  const ti = besideTrunk(tj);
  const twx = vel[idx(ti, tj, tk) * 4];
  const twz = vel[idx(ti, tj, tk) * 4 + 2];
  const gi = besideTrunk(0);
  const hs = tree.index * 4;
  const rx = tree.x - (src?.x ?? tree.x - 1);
  const rz = tree.z - (src?.z ?? tree.z);
  const rl = Math.hypot(rx, rz) || 1;
  const radial = (x: number, z: number) => (x * rx + z * rz) / rl;

  let cells = 0, maxInside = 0, maxIntoFace = 0;
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const c = idx(i, j, k);
    if (solid[c]) {
      cells++;
      maxInside = Math.max(maxInside, Math.hypot(vel[c * 4], vel[c * 4 + 1], vel[c * 4 + 2]));
      continue;
    }
    const v = [vel[c * 4], vel[c * 4 + 1], vel[c * 4 + 2]];
    if (isSolid(i + 1, j, k)) maxIntoFace = Math.max(maxIntoFace, v[0]);
    if (isSolid(i - 1, j, k)) maxIntoFace = Math.max(maxIntoFace, -v[0]);
    if (isSolid(i, j + 1, k)) maxIntoFace = Math.max(maxIntoFace, v[1]);
    if (isSolid(i, j - 1, k)) maxIntoFace = Math.max(maxIntoFace, -v[1]);
    if (isSolid(i, j, k + 1)) maxIntoFace = Math.max(maxIntoFace, v[2]);
    if (isSolid(i, j, k - 1)) maxIntoFace = Math.max(maxIntoFace, -v[2]);
  }

  let liveDrops = 0;
  for (let d = 7; d < drops.length; d += 8) if (drops[d] > 0) liveDrops++;

  return {
    time,
    finite,
    minT,
    maxSpeed,
    maxDivergence: volProjectedDivergence(vel, solid, nx, ny, nz, h),
    front: { px: front(1, 0), mx: front(-1, 0), pz: front(0, 1), mz: front(0, -1) },
    outflow2m: { px: at(2, 1, 0), pz: at(2, 0, 1) },
    coolingAtSource: cooling,
    tree: {
      ground: radial(ground(gi, tk, 0), ground(gi, tk, 2)),
      wind: radial(twx, twz),
      sway: radial(state[hs + 2], state[hs + 3]),
    },
    solid: { cells, maxInside, maxIntoFace },
    liveDrops,
  };
}
