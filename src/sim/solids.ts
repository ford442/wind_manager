import { LAYOUT, SOLID_GEOMETRY } from '../render/sceneData';
import { cellSize, type Params } from './params';

/**
 * Static solid obstacles rasterized from the backyard layout.
 *
 * One `u32` per cell: `0` is fluid, `id + 1` is inside solid `id`. Houses come
 * first (ids `0 … N_HOUSES-1`, body box + stepped roof), then tree trunks (narrow
 * columns). The same {@link SolidBox} list is uploaded to the environment pass,
 * so a drawn house sits exactly on the cells that block the wind.
 *
 * Mirrors: `N_SOLIDS` / `N_SOLID_FACES` in `common.wgsl`; `SolidDef` in
 * `environment.wgsl` (packed by {@link packSolidDefs}).
 */
export const N_SOLID_HOUSES = LAYOUT.houses.length;
export const N_SOLID_TRUNKS = LAYOUT.trees.length;
export const N_SOLIDS = N_SOLID_HOUSES + N_SOLID_TRUNKS;
/** Wettable faces per solid: [left wall, right wall]. */
export const N_SOLID_FACES = 2;
/** Floats per solid in the environment scene buffer (two vec4f). */
export const SOLID_DEF_FLOATS = 8;

export type SolidKind = 'house' | 'trunk';

export interface SolidBox {
  kind: SolidKind;
  /** Solid id: index into the mask value (`id + 1`) and the face-wetness slots. */
  id: number;
  /** Index into `LAYOUT.houses` or `LAYOUT.trees`. */
  layoutIndex: number;
  /** Snapped body extent in cells: columns `[i0, i1)`, rows `[0, j1)`. */
  i0: number;
  i1: number;
  j1: number;
  /** The same extent in metres. */
  x0: number;
  x1: number;
  top: number;
  /** Roof triangle above `top` (houses; zero for trunks): apex at `cx`. */
  cx: number;
  roofHalf: number;
  roofH: number;
}

export interface SolidLayout {
  nx: number;
  ny: number;
  h: number;
  enabled: boolean;
  boxes: SolidBox[];
  /** Per-cell mask, `nx * ny`, row-major from the floor (`j * nx + i`). */
  mask: Uint32Array<ArrayBuffer>;
  /** Number of solid cells (0 when obstacles are off). */
  solidCells: number;
}

function trunkWidthM(depth: number): number {
  const w = SOLID_GEOMETRY.trunkWidth;
  const base = depth > 0.85 ? w.near : depth > 0.7 ? w.mid : w.far;
  return base * depth;
}

/** Snapped cell boxes for every house and trunk (independent of `solidObstacles`). */
export function solidBoxes(p: Params): SolidBox[] {
  const h = cellSize(p);
  const { nx, ny } = p;
  const W = p.domainW;
  // Keep obstacles two cells clear of the open sides and top, so the edge BCs
  // still apply and `boundaries` (buoyancy.wgsl) never reads a cell it rewrites.
  const snapI = (x: number) => Math.max(2, Math.min(nx - 2, Math.round(x / h)));
  const maxJ = ny - 3;
  const boxes: SolidBox[] = [];

  LAYOUT.houses.forEach((hd, idx) => {
    const cx = hd.xFrac * W;
    const halfW = (hd.wFrac * W) / 2;
    const i0 = snapI(cx - halfW);
    const i1 = Math.max(i0 + 1, snapI(cx + halfW));
    const j1 = Math.max(1, Math.min(maxJ, Math.round((hd.hFrac * p.domainH) / h)));
    const x0 = i0 * h;
    const x1 = i1 * h;
    const top = j1 * h;
    boxes.push({
      kind: 'house',
      id: boxes.length,
      layoutIndex: idx,
      i0, i1, j1, x0, x1, top,
      cx: (x0 + x1) / 2,
      roofHalf: (x1 - x0) / 2 + SOLID_GEOMETRY.roofOverhang,
      roofH: top * SOLID_GEOMETRY.roofHeightFrac,
    });
  });

  LAYOUT.trees.forEach((td, idx) => {
    const cx = td.xFrac * W;
    const cells = Math.max(1, Math.round(trunkWidthM(td.depth) / h));
    const i0 = Math.max(2, Math.min(nx - 2 - cells, Math.round(cx / h - cells / 2)));
    const i1 = i0 + cells;
    const solidH = td.height * td.depth * SOLID_GEOMETRY.trunkSolidFrac;
    const j1 = Math.max(1, Math.min(maxJ, Math.round(solidH / h)));
    const x0 = i0 * h;
    const x1 = i1 * h;
    boxes.push({
      kind: 'trunk',
      id: boxes.length,
      layoutIndex: idx,
      i0, i1, j1, x0, x1,
      top: j1 * h,
      cx: (x0 + x1) / 2,
      roofHalf: 0,
      roofH: 0,
    });
  });

  return boxes;
}

/** True if the cell centre `(xc, yc)` lies under `b`'s roof line. */
function inRoof(b: SolidBox, xc: number, yc: number): boolean {
  if (b.roofH <= 0 || yc < b.top) return false;
  const t = 1 - Math.abs(xc - b.cx) / b.roofHalf;
  return t > 0 && yc - b.top < b.roofH * t;
}

/**
 * Rasterize the solid mask for `p`'s grid and domain. With `p.solidObstacles`
 * off the boxes are still returned (the scene draws them) but the mask is empty.
 */
export function rasterizeSolids(p: Params): SolidLayout {
  const { nx, ny } = p;
  const h = cellSize(p);
  const boxes = solidBoxes(p);
  const mask = new Uint32Array(nx * ny);
  let solidCells = 0;
  if (p.solidObstacles) {
    for (const b of boxes) {
      const roofTopJ = Math.min(ny - 2, Math.ceil((b.top + b.roofH) / h));
      const roofI0 = Math.max(2, Math.floor((b.cx - b.roofHalf) / h));
      const roofI1 = Math.min(nx - 2, Math.ceil((b.cx + b.roofHalf) / h));
      for (let j = 0; j < roofTopJ || j < b.j1; j++) {
        const yc = (j + 0.5) * h;
        const lo = j < b.j1 ? b.i0 : roofI0;
        const hi = j < b.j1 ? b.i1 : roofI1;
        for (let i = lo; i < hi; i++) {
          if (j >= b.j1 && !inRoof(b, (i + 0.5) * h, yc)) continue;
          const k = j * nx + i;
          if (mask[k] === 0) solidCells++;
          mask[k] = b.id + 1;
        }
      }
    }
  }
  return { nx, ny, h, enabled: p.solidObstacles, boxes, mask, solidCells };
}

/**
 * Pack boxes for `environment.wgsl` `SolidDef`:
 * `a = (x0, x1, top, kind)`, `b = (cx, roofHalf, roofH, layoutIndex)`, metres.
 */
export function packSolidDefs(boxes: readonly SolidBox[]): Float32Array<ArrayBuffer> {
  if (boxes.length !== N_SOLIDS) {
    throw new Error(`expected ${N_SOLIDS} solids, got ${boxes.length}`);
  }
  const out = new Float32Array(N_SOLIDS * SOLID_DEF_FLOATS);
  boxes.forEach((b, s) => {
    out.set(
      [b.x0, b.x1, b.top, b.kind === 'house' ? 0 : 1, b.cx, b.roofHalf, b.roofH, b.layoutIndex],
      s * SOLID_DEF_FLOATS,
    );
  });
  return out;
}
