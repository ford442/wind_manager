/**
 * Yard-volume grid options and the byte size of every buffer they allocate.
 *
 * No imports: `scripts/check-volume-pressure.mjs` and the README table load
 * this file on its own.
 *
 * The issue's budget is "on the order of 256 × 48 × 8" cells over
 * 32 m × 8 m × 16 m. Taken literally that is 125 mm × 167 mm × 2 m cells: the
 * whole 16 m depth gets eight cells, too few to carry a front in +z. The
 * default keeps the same cell count (~100k) with cubic 1/3 m cells instead.
 * The literal slab is still offered and the solver handles its spacing.
 */
export interface VolumeGrid {
  id: string;
  label: string;
  nx: number;
  ny: number;
  nz: number;
}

/** Domain extent (m): x along the yard, y up, z across the yard. */
export const VOLUME_DOMAIN = { w: 32, h: 8, d: 16 } as const;

export const VOLUME_GRIDS: readonly VolumeGrid[] = [
  { id: 'yard', label: '96 × 24 × 48 (⅓ m cubes)', nx: 96, ny: 24, nz: 48 },
  { id: 'fine', label: '128 × 32 × 64 (¼ m cubes)', nx: 128, ny: 32, nz: 64 },
  { id: 'slab', label: '256 × 48 × 8 (2 m deep cells)', nx: 256, ny: 48, nz: 8 },
];

export const DEFAULT_VOLUME_GRID = 'yard';

/** Same droplet cap as the slice. Raise only after the limit check passes on real adapters. */
export const VOLUME_MAX_DROPLETS = 200_000;

/** Bytes per element; mirrors the WGSL structs. */
export const VOL_BYTES = {
  /** `vec4f` velocity (xyz + pad). */
  vel: 16,
  scalar: 4,
  /** `[q, T]` fixed-point scatter. */
  accQT: 8,
  /** `[vx, vy, vz]` fixed-point scatter. */
  accM: 12,
  /** `VDroplet`: two `vec4f`. */
  droplet: 32,
  /** `VPressureNode` (src/volume/pressure.ts). */
  pressureNode: 32,
} as const;

export function gridById(id: string): VolumeGrid {
  return VOLUME_GRIDS.find((g) => g.id === id) ?? VOLUME_GRIDS[0];
}

export function cellCount(g: VolumeGrid): number {
  return g.nx * g.ny * g.nz;
}

/** Cell spacing (m) per axis. */
export function spacing(g: VolumeGrid): [number, number, number] {
  return [VOLUME_DOMAIN.w / g.nx, VOLUME_DOMAIN.h / g.ny, VOLUME_DOMAIN.d / g.nz];
}

export interface BufferRow {
  name: string;
  /** Element count and size, for the table. */
  count: number;
  stride: number;
  bytes: number;
  /** Bound as a storage buffer (checked against maxStorageBufferBindingSize). */
  storage: boolean;
}

/**
 * Pressure levels exactly as `volPressureHierarchy` builds them (halving while
 * every axis is even and ny ≥ 4, at most `levels`). check-volume-pressure.mjs
 * asserts the two agree for every offered grid.
 */
export function levelDims(g: VolumeGrid, levels = 4): [number, number, number][] {
  const out: [number, number, number][] = [[g.nx, g.ny, g.nz]];
  while (out.length < levels) {
    const [x, y, z] = out[out.length - 1];
    if (x % 2 || y % 2 || z % 2 || y < 4) break;
    out.push([x / 2, y / 2, z / 2]);
  }
  return out;
}

/** Every buffer the volume allocates for grid `g`, before any is created. */
export function volumeBufferTable(g: VolumeGrid, maxDroplets = VOLUME_MAX_DROPLETS): BufferRow[] {
  const n = cellCount(g);
  const floor = g.nx * g.nz;
  const row = (name: string, count: number, stride: number, storage = true): BufferRow =>
    ({ name, count, stride, bytes: count * stride, storage });
  const rows: BufferRow[] = [
    row('vel0, vel1 (each)', n, VOL_BYTES.vel),
    row('T0, T1, q0, q1 (each)', n, VOL_BYTES.scalar),
    row('accQT', n, VOL_BYTES.accQT),
    row('accM', n, VOL_BYTES.accM),
    row('solid mask', n, VOL_BYTES.scalar),
    row('wet, accWet (each)', floor, VOL_BYTES.scalar),
    row('droplets', maxDroplets, VOL_BYTES.droplet),
  ];
  levelDims(g).forEach(([x, y, z], i) => {
    const c = x * y * z;
    rows.push(row(`pressure L${i} p0, p1, rhs (each)`, c, VOL_BYTES.scalar));
    rows.push(row(`pressure L${i} graph`, c, VOL_BYTES.pressureNode));
  });
  // On-demand probe readback (MAP_READ, never bound; about once a second).
  rows.push(row('probe readback: vel', n, VOL_BYTES.vel, false));
  rows.push(row('probe readback: T, solid (each)', n, VOL_BYTES.scalar, false));
  rows.push(row('probe readback: droplets', maxDroplets, VOL_BYTES.droplet, false));
  return rows;
}

export interface BufferLimitProblem {
  limit: 'maxStorageBufferBindingSize' | 'maxBufferSize';
  row: BufferRow;
  available: number;
}

/** First buffer that does not fit the adapter, or null. Run before allocating. */
export function checkVolumeBuffers(
  rows: readonly BufferRow[],
  limits: { maxStorageBufferBindingSize: number; maxBufferSize: number },
): BufferLimitProblem | null {
  for (const r of rows) {
    if (r.bytes > limits.maxBufferSize) return { limit: 'maxBufferSize', row: r, available: limits.maxBufferSize };
    if (r.storage && r.bytes > limits.maxStorageBufferBindingSize) {
      return { limit: 'maxStorageBufferBindingSize', row: r, available: limits.maxStorageBufferBindingSize };
    }
  }
  return null;
}

/** Largest single buffer across every offered grid (the device is requested once). */
export function largestVolumeBuffer(maxDroplets = VOLUME_MAX_DROPLETS): number {
  let max = 0;
  for (const g of VOLUME_GRIDS) for (const r of volumeBufferTable(g, maxDroplets)) max = Math.max(max, r.bytes);
  return max;
}

export function formatBytes(b: number): string {
  if (b >= 1 << 20) return `${(b / (1 << 20)).toFixed(2)} MiB`;
  if (b >= 1 << 10) return `${(b / (1 << 10)).toFixed(1)} KiB`;
  return `${b} B`;
}
