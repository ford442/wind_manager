#!/usr/bin/env node
// CPU numerical checks of the topology uploaded to WGSL; no solve runs on the CPU in the app.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import ts from 'typescript';

const source = readFileSync(new URL('../src/sim/pressure.ts', import.meta.url), 'utf8');
const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ES2022 } }).outputText;
const {
  pressureHierarchy, projectedDivergence, PRESSURE_CYCLES_DEFAULT,
  PRESSURE_SMOOTHS, PRESSURE_COARSE_SMOOTHS,
} = await import(`data:text/javascript;base64,${Buffer.from(js).toString('base64')}`);

function matrix(level, p) {
  const a = new Float32Array(level.nodes);
  const n = level.nx * level.ny;
  const result = Float64Array.from(p, (v, k) => a[k * 8 + 4] * v);
  const offsets = [1, level.nx, 2, 2 * level.nx];
  for (let k = 0; k < n; k++) {
    for (let d = 0; d < 4; d++) {
      const w = a[k * 8 + d];
      if (!w) continue;
      const j = k + offsets[d];
      result[k] -= w * p[j];
      result[j] -= w * p[k];
    }
  }
  return result;
}

function smooth(level, b, p, count) {
  const a = new Float32Array(level.nodes);
  const flags = new Uint32Array(level.nodes);
  for (let it = 0; it < count; it++) {
    const ap = matrix(level, p);
    p = Float64Array.from(p, (v, k) =>
      flags[k * 8 + 5] && a[k * 8 + 4] > 0 ? v + (2 / 3) * (b[k] - ap[k]) / a[k * 8 + 4] : 0);
  }
  return p;
}

function cycle(levels, b, p, index = 0) {
  const level = levels[index];
  if (index === levels.length - 1) return smooth(level, b, p, PRESSURE_COARSE_SMOOTHS);
  p = smooth(level, b, p, PRESSURE_SMOOTHS);
  const ap = matrix(level, p);
  const coarse = levels[index + 1];
  const restricted = new Float64Array(coarse.nx * coarse.ny);
  const axis = (x) => 2 * Math.floor(x / 4) + x % 2;
  const parent = (k) => axis(Math.floor(k / level.nx)) * coarse.nx + axis(k % level.nx);
  const active = new Uint32Array(level.nodes);
  for (let k = 0; k < b.length; k++) if (active[k * 8 + 5]) restricted[parent(k)] += b[k] - ap[k];
  const correction = cycle(levels, restricted, new Float64Array(restricted.length), index + 1);
  for (let k = 0; k < p.length; k++) p[k] += correction[parent(k)];
  return smooth(level, b, p, PRESSURE_SMOOTHS);
}

function legacyJacobi(b, solid, nx, ny) {
  let p = new Float64Array(nx * ny);
  for (let it = 0; it < 30; it++) {
    const out = new Float64Array(p.length);
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const k = y * nx + x;
        if (solid[k]) continue;
        let sum = b[k];
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const xx = x + dx, yy = y + dy;
          if (yy >= ny) continue;
          sum += xx < 0 || xx >= nx || yy < 0 || solid[yy * nx + xx] ? p[k] : p[yy * nx + xx];
        }
        out[k] = sum / 4;
      }
    }
    p = out;
  }
  return p;
}

// Exact transpose pair used by divergence/project, including walls and open top.
function gradient(p, solid, nx, ny, h = 1) {
  const fluid = (x, y) => x >= 0 && x < nx && y >= 0 && y < ny && !solid[y * nx + x];
  const v = new Float32Array(nx * ny * 2);
  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      const k = y * nx + x;
      if (!fluid(x, y)) continue;
      if ((x === 0 || fluid(x - 1, y)) && (x === nx - 1 || fluid(x + 1, y))) {
        v[2 * k] = (p[y * nx + Math.min(nx - 1, x + 1)] -
          p[y * nx + Math.max(0, x - 1)]) / (2 * h);
      }
      if (y > 0 && fluid(x, y - 1) && (y === ny - 1 || fluid(x, y + 1))) {
        v[2 * k + 1] = ((y === ny - 1 ? 0 : p[k + nx]) - p[k - nx]) / (2 * h);
      }
    }
  }
  return v;
}

const norm = (v) => v.reduce((max, x) => Math.max(max, Math.abs(x)), 0);
for (const obstacles of [false, true]) {
  const nx = 128, ny = 64, n = nx * ny;
  const solid = new Uint32Array(n);
  if (obstacles) {
    // House plus a one-cell trunk; neither restriction nor correction may bridge them.
    for (let y = 0; y < 24; y++) for (let x = 38; x < 53; x++) solid[y * nx + x] = 1;
    for (let y = 0; y < 18; y++) solid[y * nx + 85] = 2;
  }
  const levels = pressureHierarchy(nx, ny, solid);
  assert.equal(levels.length, 3);
  for (const l of levels) assert.equal((l.nx % 2) + (l.ny % 2), 0);
  const flags = new Uint32Array(levels[0].nodes);
  const p = Float64Array.from({ length: n }, (_, k) => flags[k * 8 + 5] ?
    Math.sin(k * 1.7) + Math.cos(k * 0.3) : 0);
  const ap = matrix(levels[0], p);
  const gp = gradient(p, solid, nx, ny);
  const energy = p.reduce((sum, v, k) => sum + v * ap[k], 0);
  const gradEnergy = gp.reduce((sum, v) => sum + v * v, 0);
  assert.ok(Math.abs(energy - gradEnergy) < 1e-3, 'A must equal GᵀG, including walls');
  for (let i = 0; i < levels.length - 1; i++) {
    const fine = levels[i], coarse = levels[i + 1];
    const active = new Uint32Array(coarse.nodes);
    const cp = Float64Array.from({ length: coarse.nx * coarse.ny }, (_, k) =>
      active[k * 8 + 5] ? Math.sin(k * 0.7) : 0);
    const axis = (x) => 2 * Math.floor(x / 4) + x % 2;
    const parent = (k) => axis(Math.floor(k / fine.nx)) * coarse.nx + axis(k % fine.nx);
    const prolong = Float64Array.from({ length: fine.nx * fine.ny }, (_, k) => cp[parent(k)]);
    const afp = matrix(fine, prolong);
    const rap = new Float64Array(cp.length);
    for (let k = 0; k < afp.length; k++) if (active[parent(k) * 8 + 5]) rap[parent(k)] += afp[k];
    assert.ok(norm(matrix(coarse, cp).map((v, k) => v - rap[k])) < 1e-10, 'Galerkin operator');
  }

  // Smooth, gust-scale error: the slow mode of the previous fixed Jacobi solve.
  const exact = Float64Array.from({ length: n }, (_, k) => flags[k * 8 + 5] ?
    Math.cos((k % nx) * Math.PI / nx) * Math.cos(Math.floor(k / nx) * Math.PI / (2 * ny)) : 0);
  const b = matrix(levels[0], exact);
  let mg = new Float64Array(n);
  for (let c = 0; c < PRESSURE_CYCLES_DEFAULT; c++) mg = cycle(levels, b, mg);
  const jacobi = legacyJacobi(b, solid, nx, ny);
  const residual = (p) => norm(matrix(levels[0], p).map((v, k) => b[k] - v));
  assert.ok(residual(mg) < residual(jacobi), 'V-cycles must beat the previous 30-iteration five-point solve');
  const error = Float64Array.from(exact, (v, k) => v - mg[k]);
  const velocity = gradient(error, solid, nx, ny);
  const divergence = projectedDivergence(velocity, solid, nx, ny, 1);
  assert.ok(Math.abs(divergence - residual(mg)) < 1e-7, 'report actual projected divergence');
  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      const k = y * nx + x;
      if (solid[k]) assert.equal(Math.abs(velocity[2 * k]) + Math.abs(velocity[2 * k + 1]), 0);
      if ((x > 0 && solid[k - 1]) || (x + 1 < nx && solid[k + 1])) assert.equal(velocity[2 * k], 0);
      if (y === 0 || (y > 0 && solid[k - nx]) || (y + 1 < ny && solid[k + nx])) assert.equal(velocity[2 * k + 1], 0);
    }
  }
  console.log(`ok ${obstacles ? 'obstacles' : 'all-fluid'}: V-cycle max divergence ${divergence.toExponential(3)}, 30 Jacobi ${residual(jacobi).toExponential(3)}`);
}

for (const [nx, ny] of [[512, 128], [504, 72], [64, 64]]) {
  const levels = pressureHierarchy(nx, ny, new Uint32Array(nx * ny));
  assert.equal(levels.length, 3);
  assert.ok(levels.every((l) => l.nx % 2 === 0 && l.ny % 2 === 0));
}
assert.equal(projectedDivergence(new Float32Array(16 * 16 * 2), new Uint32Array(16 * 16), 16, 16, 0.1), 0);
console.log('Pressure topology, transfer operators, wall constraints and diagnostics agree.');
