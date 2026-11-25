/** Positive-direction graph edges: x+1, y+1, x+2, y+2. */
export const PRESSURE_NODE_BYTES = 32;
export const PRESSURE_CYCLES_DEFAULT = 2;
export const PRESSURE_SMOOTHS = 2;
export const PRESSURE_COARSE_SMOOTHS = 12;

export interface PressureGrid {
  nx: number;
  ny: number;
  /** vec4 edge weights, diagonal, fluid flag, two padding words. */
  nodes: ArrayBuffer;
}

/**
 * G is the centred, collocated pressure gradient with zero wall-normal
 * components. Build A = GᵀG, including its stride-two interior stencil.
 * The top ghost pressure is zero; side ghosts copy the centre pressure.
 * Units are cell widths (the RHS is h Gᵀu).
 */
export function pressureHierarchy(nx: number, ny: number, solid: Uint32Array): PressureGrid[] {
  const make = (nx: number, ny: number): PressureGrid => ({
    nx, ny, nodes: new ArrayBuffer(nx * ny * PRESSURE_NODE_BYTES),
  });
  const fine = make(nx, ny);
  const f = new Float32Array(fine.nodes);
  const flags = new Uint32Array(fine.nodes);
  for (let k = 0; k < nx * ny; k++) flags[k * 8 + 5] = solid[k] === 0 ? 1 : 0;
  const fluid = (x: number, y: number) =>
    x >= 0 && x < nx && y >= 0 && y < ny && solid[y * nx + x] === 0;
  function edge(a: number, b: number, axis: number): void {
    f[a * 8 + axis] += 0.25;
    f[a * 8 + 4] += 0.25;
    f[b * 8 + 4] += 0.25;
  }
  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      if (!fluid(x, y)) continue;
      const k = y * nx + x;
      if ((x === 0 || fluid(x - 1, y)) && (x === nx - 1 || fluid(x + 1, y))) {
        const a = y * nx + Math.max(0, x - 1);
        const b = y * nx + Math.min(nx - 1, x + 1);
        edge(a, b, b - a === 1 ? 0 : 2);
      }
      if (y > 0 && fluid(x, y - 1) && (y === ny - 1 || fluid(x, y + 1))) {
        const a = k - nx;
        if (y === ny - 1) f[a * 8 + 4] += 0.25;
        else edge(a, k + nx, 3);
      }
    }
  }

  const levels = [fine];
  // Grid controls snap to multiples of eight; never create an odd-sized level.
  while (levels.length < 3) {
    const prev = levels[levels.length - 1];
    if (prev.nx % 4 !== 0 || prev.ny % 4 !== 0) break;
    const coarse = make(prev.nx / 2, prev.ny / 2);
    const a = new Float32Array(prev.nodes);
    const active = new Uint32Array(prev.nodes);
    const c = new Float32Array(coarse.nodes);
    const cActive = new Uint32Array(coarse.nodes);
    // Keep the four parity subgrids of the centred stencil separate.
    const parentAxis = (x: number) => 2 * Math.floor(x / 4) + x % 2;
    const parent = (k: number) =>
      parentAxis(Math.floor(k / prev.nx)) * coarse.nx + parentAxis(k % prev.nx);
    // Conservative aggregates cannot span a solid face or a thin obstacle.
    for (let y = 0; y < coarse.ny; y++) {
      for (let x = 0; x < coarse.nx; x++) {
        const fx = 4 * Math.floor(x / 2) + x % 2;
        const fy = 4 * Math.floor(y / 2) + y % 2;
        const k = fy * prev.nx + fx;
        const children = [k, k + 2, k + 2 * prev.nx, k + 2 * prev.nx + 2];
        let members = 0;
        for (let j = 0; j < 4; j++) if (active[children[j] * 8 + 5]) members |= 1 << j;
        let reached = members & -members;
        const links = [[0, 1, 2], [2, 3, 2], [0, 2, 3], [1, 3, 3]];
        for (let it = 0; it < 4; it++) {
          for (const [from, to, axis] of links) {
            if (a[children[from] * 8 + axis] > 0 && (reached & ((1 << from) | (1 << to)))) {
              reached |= (1 << from) | (1 << to);
            }
          }
        }
        cActive[(y * coarse.nx + x) * 8 + 5] = members !== 0 && reached === members ? 1 : 0;
      }
    }
    // Piecewise-constant P, sum restriction R=Pᵀ: A_coarse = Pᵀ A P.
    // An inactive aggregate has zero correction, not an artificial connection.
    const offsets = [1, prev.nx, 2, 2 * prev.nx];
    for (let k = 0; k < prev.nx * prev.ny; k++) {
      const pk = parent(k);
      if (cActive[pk * 8 + 5]) c[pk * 8 + 4] += a[k * 8 + 4];
      for (let d = 0; d < 4; d++) {
        const w = a[k * 8 + d];
        if (w === 0) continue;
        const pn = parent(k + offsets[d]);
        if (!cActive[pk * 8 + 5] || !cActive[pn * 8 + 5]) continue;
        if (pk === pn) c[pk * 8 + 4] -= 2 * w;
        else {
          const first = Math.min(pk, pn);
          const distance = Math.abs(pn - pk);
          const axis = distance === 1 ? 0 : distance === coarse.nx ? 1 :
            distance === 2 ? 2 : 3;
          c[first * 8 + axis] += w;
        }
      }
    }
    levels.push(coarse);
  }
  return levels;
}

/** -div(u), using exactly the adjoint of the shader's collocated gradient. */
export function projectedDivergence(
  velocity: Float32Array, solid: Uint32Array, nx: number, ny: number, h: number,
): number {
  const fluid = (x: number, y: number) =>
    x >= 0 && y >= 0 && x < nx && y < ny && solid[y * nx + x] === 0;
  const vx = (x: number, y: number) =>
    fluid(x, y) && (x === 0 || fluid(x - 1, y)) && (x === nx - 1 || fluid(x + 1, y))
      ? velocity[(y * nx + x) * 2] : 0;
  const vy = (x: number, y: number) =>
    fluid(x, y) && y > 0 && fluid(x, y - 1) && (y === ny - 1 || fluid(x, y + 1))
      ? velocity[(y * nx + x) * 2 + 1] : 0;
  let max = 0;
  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      if (!fluid(x, y)) continue;
      const d = (vx(x - 1, y) - vx(x + 1, y) +
        (x === nx - 1 ? vx(x, y) : 0) - (x === 0 ? vx(x, y) : 0) +
        vy(x, y - 1) - vy(x, y + 1)) / (2 * h);
      max = Math.max(max, Math.abs(d));
    }
  }
  return max;
}
