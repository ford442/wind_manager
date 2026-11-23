#!/usr/bin/env node
// Tie ref/ (the C++ scalar reference) to the code it claims to mirror.
//
// ref/wind_ref_test.cpp pins the C++ numbers to golden values, but on its own it
// cannot notice someone editing `L_VAP` in common.wgsl: the C++ would stay green
// while the shader (and the README) moved. This script closes that loop:
//
//   1. Every `constexpr double NAME = value;` in ref/wind_ref.hpp must exist in
//      src/shaders/common.wgsl as `const NAME : f32 = value;` with the same value.
//   2. The formulas ref/ re-implements must still appear verbatim (whitespace
//      collapsed) in common.wgsl, droplets.wgsl and params.ts.
//
// Step 2 is a tripwire, not a proof: it fails when a pinned expression changes
// *or is merely reformatted*. When it fires, update ref/wind_ref.cpp and the
// golden values in ref/wind_ref_test.cpp (re-derive them; see that file), then
// update the pin here.
//
// Run: npm run check:ref  (also part of the CI `ref` job)

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const read = (rel) => readFileSync(join(ROOT, rel), 'utf8');
const collapse = (s) => s.replace(/\s+/g, ' ');

const problems = [];

// --- 1. constants --------------------------------------------------------------

const wgsl = new Map(
  [...read('src/shaders/common.wgsl').matchAll(/^const\s+(\w+)\s*:\s*f32\s*=\s*([^;]+);/gm)].map((m) => [m[1], Number(m[2].trim())]),
);
const cpp = [...read('ref/wind_ref.hpp').matchAll(/^constexpr\s+double\s+(\w+)\s*=\s*([^;]+);/gm)].map((m) => [m[1], Number(m[2].trim())]);

if (cpp.length === 0) problems.push('found no `constexpr double NAME = value;` lines in ref/wind_ref.hpp');
for (const [name, value] of cpp) {
  if (!wgsl.has(name)) problems.push(`ref/wind_ref.hpp defines ${name}, but src/shaders/common.wgsl has no \`const ${name} : f32\``);
  else if (wgsl.get(name) !== value) problems.push(`${name}: ref/wind_ref.hpp has ${value}, common.wgsl has ${wgsl.get(name)}`);
  else console.log(`ok   const ${name.padEnd(8)} = ${value}`);
}

// --- 2. pinned formulas --------------------------------------------------------

/** [file, expression that must appear (whitespace-insensitive), what ref/ function mirrors it] */
const PINS = [
  ['src/shaders/common.wgsl', 'let es = 610.94 * exp(17.625 * t_c / (t_c + 243.04));', 'q_sat'],
  ['src/shaders/common.wgsl', 'let p = 101325.0;', 'q_sat'],
  ['src/shaders/common.wgsl', 'return 0.622 * es / (p - 0.378 * es);', 'q_sat'],
  ['src/shaders/common.wgsl', 'return (t_c + 273.15) * (1.0 + 0.61 * q);', 'virt_temp'],
  ['src/sim/params.ts', 'const es = 610.94 * Math.exp((17.625 * tC) / (tC + 243.04));', 'q_sat (TS twin)'],
  ['src/sim/params.ts', 'const p = 101325.0;', 'q_sat (TS twin)'],
  ['src/sim/params.ts', 'return (0.622 * es) / (p - 0.378 * es);', 'q_sat (TS twin)'],
  ['src/shaders/droplets.wgsl', 'let re = 2.0 * d.radius * rel / NU_AIR;', 'd2_step'],
  ['src/shaders/droplets.wgsl', 'let sh = 2.0 + 0.6 * sqrt(re) * pow(SC_AIR, 1.0 / 3.0);', 'd2_step'],
  ['src/shaders/droplets.wgsl', 'let deficit = max(0.0, q_sat(ta) - qa);', 'd2_step'],
  ['src/shaders/droplets.wgsl', 'let dr2 = -(2.0 * DV_H2O * sh * RHO_AIR / RHO_W) * deficit;', 'd2_step'],
  ['src/shaders/droplets.wgsl', 'let r_new = sqrt(max(0.0, d.radius * d.radius + dr2 * P.dt));', 'd2_step'],
  [
    'src/shaders/droplets.wgsl',
    'let dm = RHO_W * (4.0 / 3.0) * PI * max(0.0, d.radius * d.radius * d.radius - r_new * r_new * r_new);',
    'd2_step',
  ],
  ['src/shaders/droplets.wgsl', 'let mcell = RHO_AIR * P.h * P.h;', 'cell_mass'],
  ['src/shaders/droplets.wgsl', 'if (P.latent_on > 0.5) { dtemp = -(L_VAP * dm) / (mcell * CP_AIR); }', 'latent_dT'],
  // The README states these numbers; keep the prose and the code from parting ways.
  ['README.md', 'L_v ≈ 2.45 MJ/kg', 'L_VAP'],
  ['README.md', 'ΔT = −L_v · dm / (m_cell · c_p)', 'latent_dT'],
  ['README.md', 'T_v = (T + 273.15) · (1 + 0.61 · q)', 'virt_temp'],
];

const cache = new Map();
for (const [file, expr, owner] of PINS) {
  if (!cache.has(file)) cache.set(file, collapse(read(file)));
  if (cache.get(file).includes(collapse(expr))) console.log(`ok   ${file}: ${owner}`);
  else problems.push(`${file} no longer contains\n      ${expr}\n    (mirrored by ref/ ${owner}). Update ref/, re-derive the goldens, then update PINS.`);
}

if (problems.length) {
  console.error(`\n${problems.length} parity problem(s):\n`);
  for (const p of problems) console.error(`- ${p}\n`);
  process.exit(1);
}
console.log(`\nref/ agrees with common.wgsl, droplets.wgsl, params.ts and README.md (${cpp.length} constants, ${PINS.length} pinned expressions).`);
