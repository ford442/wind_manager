import { LAYOUT, SOLID_GEOMETRY } from '../render/sceneData';
import { VOLUME_DOMAIN, type VolumeGrid, spacing } from './grid';

/**
 * Solid obstacles in the yard volume, rasterized from the same backyard layout
 * as the slice (`src/sim/solids.ts`), now with a z extent: houses are boxes
 * `[x0, x1] × [0, top] × [z0, z1]` with a gable roof running along z, trunks
 * are square columns. Mask: 0 fluid, `id + 1` inside solid `id` (houses
 * first). The renderer draws these boxes, so the houses in the yard are the
 * houses in the grid.
 */
export interface VolumeSolidBox {
  kind: 'house' | 'trunk';
  id: number;
  layoutIndex: number;
  /** Snapped cell extent: `[i0, i1) × [0, j1) × [k0, k1)`. */
  i0: number; i1: number; j1: number; k0: number; k1: number;
  /** The same extent in metres. */
  x0: number; x1: number; top: number; z0: number; z1: number;
  /** Gable roof over the box (houses): apex at x = cx, eaves `roofHalf` from it. */
  cx: number; roofHalf: number; roofH: number;
}

export interface VolumeSolids {
  boxes: VolumeSolidBox[];
  mask: Uint32Array<ArrayBuffer>;
  solidCells: number;
  enabled: boolean;
}

/** Floats per box for `vol_scene.wgsl` `VSolid` (three vec4f). */
export const VOL_SOLID_FLOATS = 12;
/** Cells kept clear of the open sides and top (see `vol_boundaries`). */
const EDGE_CLEAR = 3;

function trunkWidthM(depth: number): number {
  const w = SOLID_GEOMETRY.trunkWidth;
  return (depth > 0.85 ? w.near : depth > 0.7 ? w.mid : w.far) * depth;
}

export function volumeSolidBoxes(g: VolumeGrid): VolumeSolidBox[] {
  const [hx, hy, hz] = spacing(g);
  const { w: W, h: H, d: D } = VOLUME_DOMAIN;
  const snap = (v: number, hh: number, n: number) =>
    Math.max(EDGE_CLEAR, Math.min(n - EDGE_CLEAR, Math.round(v / hh)));
  const maxJ = g.ny - EDGE_CLEAR - 1;
  const boxes: VolumeSolidBox[] = [];

  LAYOUT.houses.forEach((hd, idx) => {
    const cx = hd.xFrac * W;
    const cz = hd.zFrac * D;
    const halfW = (hd.wFrac * W) / 2;
    const halfD = (hd.dFrac * D) / 2;
    const i0 = snap(cx - halfW, hx, g.nx);
    const i1 = Math.max(i0 + 1, snap(cx + halfW, hx, g.nx));
    const k0 = snap(cz - halfD, hz, g.nz);
    const k1 = Math.max(k0 + 1, snap(cz + halfD, hz, g.nz));
    const j1 = Math.max(1, Math.min(maxJ, Math.round((hd.hFrac * H) / hy)));
    const x0 = i0 * hx, x1 = i1 * hx, top = j1 * hy;
    boxes.push({
      kind: 'house', id: boxes.length, layoutIndex: idx,
      i0, i1, j1, k0, k1, x0, x1, top, z0: k0 * hz, z1: k1 * hz,
      cx: (x0 + x1) / 2,
      roofHalf: (x1 - x0) / 2 + SOLID_GEOMETRY.roofOverhang,
      roofH: top * SOLID_GEOMETRY.roofHeightFrac,
    });
  });

  LAYOUT.trees.forEach((td, idx) => {
    const width = trunkWidthM(td.depth);
    const cellsX = Math.max(1, Math.round(width / hx));
    const cellsZ = Math.max(1, Math.round(width / hz));
    const i0 = Math.max(EDGE_CLEAR, Math.min(g.nx - EDGE_CLEAR - cellsX, Math.round((td.xFrac * W) / hx - cellsX / 2)));
    const k0 = Math.max(EDGE_CLEAR, Math.min(g.nz - EDGE_CLEAR - cellsZ, Math.round((td.zFrac * D) / hz - cellsZ / 2)));
    const j1 = Math.max(1, Math.min(maxJ, Math.round((td.height * td.depth * SOLID_GEOMETRY.trunkSolidFrac) / hy)));
    const x0 = i0 * hx, x1 = (i0 + cellsX) * hx;
    boxes.push({
      kind: 'trunk', id: boxes.length, layoutIndex: idx,
      i0, i1: i0 + cellsX, j1, k0, k1: k0 + cellsZ,
      x0, x1, top: j1 * hy, z0: k0 * hz, z1: (k0 + cellsZ) * hz,
      cx: (x0 + x1) / 2, roofHalf: 0, roofH: 0,
    });
  });
  return boxes;
}

export function rasterizeVolumeSolids(g: VolumeGrid, enabled: boolean): VolumeSolids {
  const [hx, hy] = spacing(g);
  const boxes = volumeSolidBoxes(g);
  const mask = new Uint32Array(g.nx * g.ny * g.nz);
  let solidCells = 0;
  if (enabled) {
    for (const b of boxes) {
      const roofTopJ = Math.min(g.ny - EDGE_CLEAR - 1, Math.ceil((b.top + b.roofH) / hy));
      const roofI0 = Math.max(EDGE_CLEAR, Math.floor((b.cx - b.roofHalf) / hx));
      const roofI1 = Math.min(g.nx - EDGE_CLEAR, Math.ceil((b.cx + b.roofHalf) / hx));
      for (let k = b.k0; k < b.k1; k++) {
        for (let j = 0; j < Math.max(b.j1, roofTopJ); j++) {
          const body = j < b.j1;
          const lo = body ? b.i0 : roofI0;
          const hi = body ? b.i1 : roofI1;
          for (let i = lo; i < hi; i++) {
            if (!body) {
              const t = 1 - Math.abs((i + 0.5) * hx - b.cx) / b.roofHalf;
              if (b.roofH <= 0 || t <= 0 || (j + 0.5) * hy - b.top >= b.roofH * t) continue;
            }
            const c = (k * g.ny + j) * g.nx + i;
            if (mask[c] === 0) solidCells++;
            mask[c] = b.id + 1;
          }
        }
      }
    }
  }
  return { boxes, mask, solidCells, enabled };
}

/**
 * `VSolid` in vol_scene.wgsl: `a = (x0, top, z0, kind)`, `b = (x1, roofH, z1,
 * layoutIndex)`, `c = (cx, roofHalf, 0, 0)`, metres.
 */
export function packVolumeSolids(boxes: readonly VolumeSolidBox[]): Float32Array<ArrayBuffer> {
  const out = new Float32Array(boxes.length * VOL_SOLID_FLOATS);
  boxes.forEach((b, s) => {
    out.set([
      b.x0, b.top, b.z0, b.kind === 'house' ? 0 : 1,
      b.x1, b.roofH, b.z1, b.layoutIndex,
      b.cx, b.roofHalf, 0, 0,
    ], s * VOL_SOLID_FLOATS);
  });
  return out;
}
