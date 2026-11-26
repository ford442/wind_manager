/**
 * Pressure topology for the yard volume.
 *
 * The slice (`src/sim/pressure.ts`) solves A = GᵀG for the centred collocated
 * gradient G. Its stride-two stencil splits the grid into independent parity
 * subgrids (four in 2D, eight in 3D), each conserving its own mass. In the
 * volume that shows up as a ground outflow in row 0 returned through rows
 * 2, 4, 6 and a cold layer stuck one cell deep. So the volume uses the
 * standard approximate projection instead:
 *
 * - face velocities are cell averages (zero on a solid face or the floor,
 *   the cell's own value on the open sides and top);
 * - A is the compact 7-point face Laplacian D_f G_f (Neumann at walls, floor
 *   and the open x/z sides, zero ghost pressure above the top), which couples
 *   every cell to its neighbours;
 * - each cell velocity is corrected by the average of its two face gradients
 *   (`vol_project`), and wall-normal components next to a solid are clamped.
 * - coarse levels are 2×2×2 aggregates with Galerkin operators.
 *
 * The face divergence D_f(u_f − G_f p) is then zero to solver tolerance; the
 * cell-centred velocity is divergence-free up to the averaging. Weights are
 * in units of `h.x`: a face normal to axis a weighs `(h.x / h_a)²`.
 *
 * Mirrors `VPressureNode` in `vol_pressure.wgsl`: 8 words per node,
 * `[x+1, y+1, z+1, unused ×3, diagonal, fluid]`.
 */
export const VOL_PRESSURE_NODE_BYTES = 32;
const W = 8;
const DIAG = 6;
const FLUID = 7;
export const VOL_PRESSURE_LEVELS = 4;
/**
 * V-cycle settings, shared with vol_pressure.wgsl / step.ts and pinned by
 * scripts/check-volume-pressure.mjs. Piecewise-constant aggregates with a
 * Galerkin operator are stable but slow on the 7-point Laplacian; weighted
 * Jacobi at ω = 6/7 (the 3D optimum), four sweeps a side and a 1.8× coarse
 * correction bring it to ~0.3–0.4 residual reduction per cycle. p is
 * warm-started from the previous substep.
 */
export const VOL_PRESSURE_OMEGA = 6 / 7;
/** Sweeps before and after each coarse correction (even: ends in p0). */
export const VOL_PRESSURE_SMOOTHS = 4;
/** Sweeps on the coarsest level (≤ 16 × 4 × 8 for the offered grids; even). */
export const VOL_PRESSURE_COARSE_SMOOTHS = 32;
/** Scale on each prolongated coarse correction. */
export const VOL_PRESSURE_OVERCORRECT = 1.8;
/** V-cycles per substep. */
export const VOL_PRESSURE_CYCLES = 2;

export interface VolPressureGrid {
  nx: number;
  ny: number;
  nz: number;
  nodes: ArrayBuffer;
}

export type Spacing = readonly [hx: number, hy: number, hz: number];

function makeGrid(nx: number, ny: number, nz: number): VolPressureGrid {
  return { nx, ny, nz, nodes: new ArrayBuffer(nx * ny * nz * VOL_PRESSURE_NODE_BYTES) };
}

/** Fine-level operator plus Galerkin coarse levels (2×2×2 aggregates, R = Pᵀ). */
export function volPressureHierarchy(
  nx: number, ny: number, nz: number, h: Spacing, solid: Uint32Array,
): VolPressureGrid[] {
  const fine = makeGrid(nx, ny, nz);
  const f = new Float32Array(fine.nodes);
  const flags = new Uint32Array(fine.nodes);
  const fluid = (x: number, y: number, z: number) =>
    x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz && solid[(z * ny + y) * nx + x] === 0;
  const weight = h.map((ha) => (h[0] / ha) ** 2);
  const stride = [1, nx, nx * ny];
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const k = (z * ny + y) * nx + x;
        if (!fluid(x, y, z)) continue;
        flags[k * W + FLUID] = 1;
        const c = [x, y, z];
        for (let a = 0; a < 3; a++) {
          const n = [...c];
          n[a]++;
          if (fluid(n[0], n[1], n[2])) {
            f[k * W + a] += weight[a];
            f[k * W + DIAG] += weight[a];
            f[(k + stride[a]) * W + DIAG] += weight[a];
          }
        }
        // Open top: zero ghost pressure above the last row.
        if (y === ny - 1) f[k * W + DIAG] += weight[1];
      }
    }
  }
  const levels = [fine];
  while (levels.length < VOL_PRESSURE_LEVELS) {
    const prev = levels[levels.length - 1];
    if (prev.nx % 2 || prev.ny % 2 || prev.nz % 2 || prev.ny < 4) break;
    levels.push(coarsen(prev));
  }
  return levels;
}

function coarsen(prev: VolPressureGrid): VolPressureGrid {
  const coarse = makeGrid(prev.nx / 2, prev.ny / 2, prev.nz / 2);
  const a = new Float32Array(prev.nodes);
  const active = new Uint32Array(prev.nodes);
  const c = new Float32Array(coarse.nodes);
  const cActive = new Uint32Array(coarse.nodes);
  const fs = [1, prev.nx, prev.nx * prev.ny];
  const cs = [1, coarse.nx, coarse.nx * coarse.ny];
  const parent = (k: number) => {
    const x = k % prev.nx;
    const y = Math.floor(k / prev.nx) % prev.ny;
    const z = Math.floor(k / (prev.nx * prev.ny));
    return ((z >> 1) * coarse.ny + (y >> 1)) * coarse.nx + (x >> 1);
  };
  // An aggregate is active only if its fluid children connect through faces:
  // a coarse correction must not bridge a wall.
  for (let z = 0; z < coarse.nz; z++) {
    for (let y = 0; y < coarse.ny; y++) {
      for (let x = 0; x < coarse.nx; x++) {
        const base = ((2 * z) * prev.ny + 2 * y) * prev.nx + 2 * x;
        const child = (b: number) => base + (b & 1) * fs[0] + ((b >> 1) & 1) * fs[1] + ((b >> 2) & 1) * fs[2];
        let members = 0;
        for (let b = 0; b < 8; b++) if (active[child(b) * W + FLUID]) members |= 1 << b;
        let reached = members & -members;
        for (let it = 0; it < 8; it++) {
          for (let b = 0; b < 8; b++) {
            for (let axis = 0; axis < 3; axis++) {
              const bit = 1 << axis;
              if (b & bit) continue;
              const to = b | bit;
              if (a[child(b) * W + axis] > 0 && (reached & ((1 << b) | (1 << to)))) reached |= (1 << b) | (1 << to);
            }
          }
        }
        cActive[((z * coarse.ny + y) * coarse.nx + x) * W + FLUID] = members !== 0 && reached === members ? 1 : 0;
      }
    }
  }
  const n = prev.nx * prev.ny * prev.nz;
  for (let k = 0; k < n; k++) {
    const pk = parent(k);
    if (!cActive[pk * W + FLUID]) continue;
    c[pk * W + DIAG] += a[k * W + DIAG];
    for (let d = 0; d < 3; d++) {
      const w = a[k * W + d];
      if (w === 0) continue;
      const pn = parent(k + fs[d]);
      if (!cActive[pn * W + FLUID]) continue;
      if (pk === pn) c[pk * W + DIAG] -= 2 * w;
      else if (pn - pk === cs[d]) c[pk * W + d] += w;
      else throw new Error(`coarse edge ${pk}->${pn} is not a face along axis ${d}`);
    }
  }
  return coarse;
}

/**
 * Largest |D_f u_f| (1/s): the face divergence the solve drives to zero, with
 * the same face rules as `vol_divergence`. What the HUD and the probe report.
 */
export function volProjectedDivergence(
  vel: Float32Array, solid: Uint32Array, nx: number, ny: number, nz: number, h: Spacing,
  velStride = 4,
): number {
  let max = 0;
  const n = [nx, ny, nz];
  const stride = [1, nx, nx * ny];
  const fluidAt = (c: number[]) =>
    c[0] >= 0 && c[1] >= 0 && c[2] >= 0 && c[0] < nx && c[1] < ny && c[2] < nz &&
    solid[(c[2] * ny + c[1]) * nx + c[0]] === 0;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const k = (z * ny + y) * nx + x;
    if (solid[k]) continue;
    const c = [x, y, z];
    let d = 0;
    for (let a = 0; a < 3; a++) {
      const u = vel[k * velStride + a];
      const face = (dir: 1 | -1) => {
        const o = [...c];
        o[a] += dir;
        if (o[a] < 0 || o[a] >= n[a]) {
          // Floor: wall. Open sides and top: the cell's own value.
          return a === 1 && dir < 0 ? 0 : u;
        }
        if (!fluidAt(o)) return 0;
        return 0.5 * (u + vel[(k + dir * stride[a]) * velStride + a]);
      };
      d += (face(1) - face(-1)) / h[a];
    }
    max = Math.max(max, Math.abs(d));
  }
  return max;
}
