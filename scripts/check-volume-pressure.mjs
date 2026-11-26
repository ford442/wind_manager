#!/usr/bin/env node
// CPU checks of the yard-volume pressure topology (src/volume/pressure.ts)
// that vol_pressure.wgsl consumes: the operator is the compact face
// Laplacian -D_f G_f with the shader's boundary rules, the coarse levels are
// Galerkin, and the V-cycle the shader runs converges. No solve runs on the
// CPU in the app.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import ts from 'typescript';

const load = async (rel) => {
  const source = readFileSync(new URL(rel, import.meta.url), 'utf8');
  const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ES2022 } }).outputText;
  return import(`data:text/javascript;base64,${Buffer.from(js).toString('base64')}`);
};
const {
  volPressureHierarchy, VOL_PRESSURE_LEVELS, VOL_PRESSURE_SMOOTHS, VOL_PRESSURE_COARSE_SMOOTHS, VOL_PRESSURE_CYCLES,
  VOL_PRESSURE_OMEGA, VOL_PRESSURE_OVERCORRECT,
} = await load('../src/volume/pressure.ts');

// The shader hard-codes the same numbers.
const wgsl = readFileSync(new URL('../src/shaders/vol_pressure.wgsl', import.meta.url), 'utf8');
const wgslConst = (name) => Number(new RegExp(`const ${name} : f32 = ([0-9.]+);`).exec(wgsl)?.[1]);
assert.ok(Math.abs(wgslConst('VOL_OMEGA') - VOL_PRESSURE_OMEGA) < 1e-6, 'VOL_OMEGA in vol_pressure.wgsl must match VOL_PRESSURE_OMEGA');
assert.equal(wgslConst('VOL_OVERCORRECT'), VOL_PRESSURE_OVERCORRECT, 'VOL_OVERCORRECT in vol_pressure.wgsl must match');
const { VOLUME_GRIDS, levelDims } = await load('../src/volume/grid.ts');

const W = 8, DIAG = 6, FLUID = 7;
assert.equal(VOL_PRESSURE_SMOOTHS % 2, 0, 'pre/post smoothing must finish in p0');
assert.equal(VOL_PRESSURE_COARSE_SMOOTHS % 2, 0, 'coarse smoothing must finish in p0');

/** A p, as vol_pressure_smooth applies it (positive-direction edges on the lower node). */
function matrix(level, p) {
  const a = new Float32Array(level.nodes);
  const n = level.nx * level.ny * level.nz;
  const offsets = [1, level.nx, level.nx * level.ny];
  const out = Float64Array.from(p, (v, k) => a[k * W + DIAG] * v);
  for (let k = 0; k < n; k++) {
    for (let d = 0; d < 3; d++) {
      const w = a[k * W + d];
      if (!w) continue;
      out[k] -= w * p[k + offsets[d]];
      out[k + offsets[d]] -= w * p[k];
    }
  }
  return out;
}

function smooth(level, b, p, count) {
  const a = new Float32Array(level.nodes);
  const flags = new Uint32Array(level.nodes);
  for (let it = 0; it < count; it++) {
    const ap = matrix(level, p);
    p = Float64Array.from(p, (v, k) =>
      flags[k * W + FLUID] && a[k * W + DIAG] > 0 ? v + VOL_PRESSURE_OMEGA * (b[k] - ap[k]) / a[k * W + DIAG] : 0);
  }
  return p;
}

const parentOf = (fine, coarse) => (k) => {
  const x = k % fine.nx, y = Math.floor(k / fine.nx) % fine.ny, z = Math.floor(k / (fine.nx * fine.ny));
  return ((z >> 1) * coarse.ny + (y >> 1)) * coarse.nx + (x >> 1);
};

/** The V-cycle vol_pressure runs (vCycle in src/volume/step.ts). */
function cycle(levels, b, p, index = 0) {
  const level = levels[index];
  if (index === levels.length - 1) return smooth(level, b, p, VOL_PRESSURE_COARSE_SMOOTHS);
  p = smooth(level, b, p, VOL_PRESSURE_SMOOTHS);
  const ap = matrix(level, p);
  const coarse = levels[index + 1];
  const parent = parentOf(level, coarse);
  const flags = new Uint32Array(level.nodes);
  const r = new Float64Array(coarse.nx * coarse.ny * coarse.nz);
  for (let k = 0; k < b.length; k++) if (flags[k * W + FLUID]) r[parent(k)] += b[k] - ap[k];
  const corr = cycle(levels, r, new Float64Array(r.length), index + 1);
  for (let k = 0; k < p.length; k++) if (flags[k * W + FLUID]) p[k] += VOL_PRESSURE_OVERCORRECT * corr[parent(k)];
  return smooth(level, b, p, VOL_PRESSURE_SMOOTHS);
}

/** h.x² |G_f p|² over every face vol_project uses (interior fluid faces + the open top). */
function faceEnergy(p, solid, nx, ny, nz, h) {
  const idx = (x, y, z) => (z * ny + y) * nx + x;
  const fluid = (x, y, z) => x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz && !solid[idx(x, y, z)];
  let e = 0;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    if (!fluid(x, y, z)) continue;
    const k = idx(x, y, z);
    if (fluid(x + 1, y, z)) e += ((p[idx(x + 1, y, z)] - p[k]) / h[0]) ** 2;
    if (fluid(x, y + 1, z)) e += ((p[idx(x, y + 1, z)] - p[k]) / h[1]) ** 2;
    if (fluid(x, y, z + 1)) e += ((p[idx(x, y, z + 1)] - p[k]) / h[2]) ** 2;
    if (y === ny - 1) e += (p[k] / h[1]) ** 2;
  }
  return h[0] ** 2 * e;
}

/** A flat solve: 30 Jacobi sweeps of the same operator. */
function flatJacobi(level, b) {
  const a = new Float32Array(level.nodes);
  const flags = new Uint32Array(level.nodes);
  let p = new Float64Array(b.length);
  for (let it = 0; it < 30; it++) {
    const ap = matrix(level, p);
    p = Float64Array.from(p, (v, k) => flags[k * W + FLUID] ? v + (b[k] - ap[k]) / a[k * W + DIAG] : 0);
  }
  return p;
}

const norm = (v) => v.reduce((m, x) => Math.max(m, Math.abs(x)), 0);
const l2 = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));

const cases = [
  { name: 'cubic, all fluid', n: [32, 16, 24], h: [1, 1, 1], obstacles: false },
  { name: 'cubic, house + trunk', n: [32, 16, 24], h: [1, 1, 1], obstacles: true },
  // Strongly anisotropic cells (the literal 256 × 48 × 8 budget): isotropic
  // coarsening with a point smoother converges more slowly here, which is
  // why the default grid uses cubic cells.
  { name: 'slab cells (hz = 4 hx), house + trunk', n: [32, 16, 8], h: [1, 1.5, 4], obstacles: true, slow: true },
  { name: 'default yard grid, house + trunk', n: [96, 24, 48], h: [1, 1, 1], obstacles: true },
];

for (const c of cases) {
  const [nx, ny, nz] = c.n;
  const n = nx * ny * nz;
  const idx = (x, y, z) => (z * ny + y) * nx + x;
  const solid = new Uint32Array(n);
  if (c.obstacles) {
    for (let z = 3; z < Math.min(nz - 3, 12); z++) for (let y = 0; y < 6; y++) for (let x = 6; x < 13; x++) solid[idx(x, y, z)] = 1;
    for (let y = 0; y < 8; y++) solid[idx(21, y, Math.floor(nz / 2))] = 2;
  }
  const levels = volPressureHierarchy(nx, ny, nz, c.h, solid);
  assert.ok(levels.length >= 3, `${c.name}: at least three levels`);
  const flags = new Uint32Array(levels[0].nodes);

  // pᵀ A p = h.x² |G_f p|²: A is the compact face Laplacian with the shader's walls.
  const p = Float64Array.from({ length: n }, (_, k) => flags[k * W + FLUID] ? Math.sin(k * 1.7) + Math.cos(k * 0.3) : 0);
  const ap = matrix(levels[0], p);
  const energy = p.reduce((s, v, k) => s + v * ap[k], 0);
  const grad = faceEnergy(p, solid, nx, ny, nz, c.h);
  assert.ok(Math.abs(energy - grad) < 1e-4 * Math.max(1, grad), `${c.name}: A must equal -D_f G_f (${energy} vs ${grad})`);

  for (let i = 0; i < levels.length - 1; i++) {
    const fine = levels[i], coarse = levels[i + 1];
    const active = new Uint32Array(coarse.nodes);
    const parent = parentOf(fine, coarse);
    const cp = Float64Array.from({ length: coarse.nx * coarse.ny * coarse.nz }, (_, k) => active[k * W + FLUID] ? Math.sin(k * 0.7) : 0);
    const afp = matrix(fine, Float64Array.from({ length: fine.nx * fine.ny * fine.nz }, (_, k) => cp[parent(k)]));
    const rap = new Float64Array(cp.length);
    for (let k = 0; k < afp.length; k++) if (active[parent(k) * W + FLUID]) rap[parent(k)] += afp[k];
    assert.ok(norm(matrix(coarse, cp).map((v, k) => v - rap[k])) < 1e-4, `${c.name}: Galerkin operator, level ${i + 1}`);
  }

  // A gust-like right-hand side: yard-scale smooth part plus cell-scale noise.
  const rhs = Float64Array.from({ length: n }, (_, k) => {
    if (!flags[k * W + FLUID]) return 0;
    const x = k % nx, y = Math.floor(k / nx) % ny, z = Math.floor(k / (nx * ny));
    return Math.cos(x * Math.PI / nx) * Math.cos(z * Math.PI / nz) * Math.cos(y * Math.PI / (2 * ny)) + 0.3 * Math.sin(k * 1.3);
  });
  const residual = (q) => l2(matrix(levels[0], q).map((v, k) => rhs[k] - v)) / l2(rhs);
  // Three substeps' worth of warm-started V-cycles on a fixed right-hand side.
  let mg = new Float64Array(n);
  const history = [];
  for (let k = 0; k < 3 * VOL_PRESSURE_CYCLES; k++) { mg = cycle(levels, rhs, mg); history.push(residual(mg)); }
  const jac = residual(flatJacobi(levels[0], rhs));
  assert.ok(history[VOL_PRESSURE_CYCLES - 1] < jac, `${c.name}: one substep of V-cycles must beat 30 Jacobi sweeps`);
  const [oneStep, threeSteps] = c.slow ? [0.7, 0.2] : [0.35, 0.05];
  assert.ok(history[VOL_PRESSURE_CYCLES - 1] < oneStep, `${c.name}: one substep left ${history[VOL_PRESSURE_CYCLES - 1]} of the residual`);
  assert.ok(history.at(-1) < threeSteps, `${c.name}: three warm-started substeps left ${history.at(-1)} of the residual`);
  for (let k = 0; k < n; k++) if (solid[k]) assert.equal(mg[k], 0, `${c.name}: pressure inside a solid`);
  console.log(`ok ${c.name}: ${levels.length} levels, relative residual per V-cycle ` +
    `${history.map((r) => r.toExponential(1)).join(' → ')} (30 Jacobi sweeps: ${jac.toExponential(1)})`);
}

for (const g of VOLUME_GRIDS) {
  const levels = volPressureHierarchy(g.nx, g.ny, g.nz, [1, 1, 1], new Uint32Array(g.nx * g.ny * g.nz));
  assert.ok(levels.length >= 3 && levels.length <= VOL_PRESSURE_LEVELS, `${g.id}: ${levels.length} levels`);
  assert.deepEqual(levelDims(g, VOL_PRESSURE_LEVELS), levels.map((l) => [l.nx, l.ny, l.nz]),
    `${g.id}: the buffer table (levelDims) must list the levels the hierarchy builds`);
}
console.log('Volume pressure operator, Galerkin levels and V-cycle convergence agree with vol_pressure.wgsl.');
