#!/usr/bin/env node
// Validate every WGSL shader module exactly as the app assembles it, with Naga.
//
// Why this exists: `createSim`, `initTracerPipelines` and `createRenderer` do not
// feed a .wgsl file to the GPU on its own. Each one joins [generated prelude?,
// common.wgsl, <shader>.wgsl] with "\n" (see compileShaderModule in
// src/render/webgpu.ts). A validator that parses files one at a time rejects every
// entry point for missing `Params` and helpers, so this script rebuilds the same
// joined text and validates that. It needs no GPU and no browser.
//
// Run: npm run check:wgsl      Needs `naga` on PATH (or NAGA=/path/to/naga):
//      cargo install naga-cli --version 30.0.2 --locked
//
// Naga is not Tint (Chrome). Passing here means Naga accepts the module, which
// also covers Firefox's WebGPU; it does not prove Chrome accepts it. Runtime
// getCompilationInfo() in compileShaderModule remains the check for that.

import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const SHADER_DIR = join(ROOT, 'src', 'shaders');
const COMMON = 'common.wgsl';

/**
 * One row per shader module the app compiles. Keep in sync with the TypeScript
 * that builds it (`consumer`); this script fails if the two disagree.
 *
 * - `generated`: step.ts prepends `const MAX_EMITTERS : u32 = <n>u;`.
 * - `entryPoints`: the names pipelines are created with. Naga happily validates
 *   a module with no entry points, so a dropped `@compute` would otherwise pass.
 */
const MODULES = [
  { name: 'droplets', consumer: 'src/sim/step.ts', generated: true, entryPoints: ['emit', 'update', 'air_inject'] },
  { name: 'apply_scatter', consumer: 'src/sim/step.ts', generated: true, entryPoints: ['apply', 'apply_wet'] },
  { name: 'advect', consumer: 'src/sim/step.ts', generated: true, entryPoints: ['advect'] },
  { name: 'buoyancy', consumer: 'src/sim/step.ts', generated: true, entryPoints: ['buoyancy', 'boundaries'] },
  { name: 'pressure', consumer: 'src/sim/step.ts', generated: true, entryPoints: ['divergence', 'jacobi', 'project'] },
  { name: 'tracers', consumer: 'src/sim/tracers.ts', generated: false, entryPoints: ['seed_tracers', 'update_tracers'] },
  {
    name: 'render',
    consumer: 'src/render/overlays.ts',
    generated: false,
    entryPoints: [
      'vs_fullscreen', 'fs_overlay', 'vs_arrows', 'fs_arrows', 'vs_drops', 'fs_drops',
      'vs_emitter', 'fs_emitter', 'vs_tracer_streaks', 'vs_tracer_dots', 'fs_tracers',
    ],
  },
];

const read = (path) => readFileSync(path, 'utf8');
const problems = [];
const fail = (msg) => problems.push(msg);

// --- Source of truth for the generated prelude --------------------------------

function maxEmitters() {
  const m = /export\s+const\s+MAX_EMITTERS\s*=\s*(\d+)\s*;/.exec(read(join(ROOT, 'src/sim/emitters.ts')));
  if (!m) throw new Error('could not find `export const MAX_EMITTERS = <n>;` in src/sim/emitters.ts');
  return Number(m[1]);
}

/** Must match the `generated` part in createSim (src/sim/step.ts). */
const generatedPrelude = (n) => `const MAX_EMITTERS : u32 = ${n}u;`;

// --- Manifest <-> repository consistency --------------------------------------

function checkManifest() {
  const onDisk = readdirSync(SHADER_DIR).filter((f) => f.endsWith('.wgsl')).sort();
  const listed = [COMMON, ...MODULES.map((m) => `${m.name}.wgsl`)].sort();
  for (const f of onDisk) if (!listed.includes(f)) fail(`src/shaders/${f} is not in MODULES (scripts/validate-wgsl.mjs); it would ship unvalidated`);
  for (const f of listed) if (!onDisk.includes(f)) fail(`MODULES lists ${f} but src/shaders/${f} does not exist`);

  // The TS that builds each module must import the same files and prelude.
  for (const consumer of new Set(MODULES.map((m) => m.consumer))) {
    const ts = read(join(ROOT, consumer));
    const mine = MODULES.filter((m) => m.consumer === consumer);
    for (const m of mine) {
      for (const file of [COMMON, `${m.name}.wgsl`]) {
        if (!ts.includes(`shaders/${file}?raw`)) fail(`${consumer} does not import shaders/${file}?raw, but MODULES says it builds "${m.name}"`);
      }
    }
    const usesGenerated = /const\s+MAX_EMITTERS\s*:\s*u32/.test(ts);
    const expectGenerated = mine.some((m) => m.generated);
    if (mine.some((m) => m.generated !== expectGenerated)) fail(`MODULES mixes generated/non-generated modules for ${consumer}; the prelude is added per file there`);
    else if (expectGenerated && !usesGenerated) fail(`${consumer} no longer prepends the MAX_EMITTERS prelude; set generated: false for ${mine.map((m) => m.name).join(', ')}`);
    else if (!expectGenerated && usesGenerated) fail(`${consumer} prepends the MAX_EMITTERS prelude; set generated: true for ${mine.map((m) => m.name).join(', ')}`);
  }

  // compileShaderModule joins parts with "\n"; this script must do the same.
  if (!/\.join\('\\n'\)/.test(read(join(ROOT, 'src/render/webgpu.ts')))) {
    fail("compileShaderModule in src/render/webgpu.ts no longer joins parts with '\\n'; update compose() here");
  }
}

// --- Composition + Naga --------------------------------------------------------

/** parts: [{ name, code }] joined with "\n", as compileShaderModule does. */
function compose(m, emitters) {
  const parts = [];
  if (m.generated) parts.push({ name: 'generated', code: generatedPrelude(emitters) });
  parts.push({ name: COMMON, code: read(join(SHADER_DIR, COMMON)) });
  parts.push({ name: `${m.name}.wgsl`, code: read(join(SHADER_DIR, `${m.name}.wgsl`)) });
  return { parts, code: parts.map((p) => p.code).join('\n') };
}

const lineCount = (s) => s.split('\n').length;

/** 1-based line in the joined module -> `part:line` (mirror of locate() in webgpu.ts). */
function locate(parts, line) {
  let start = 1;
  for (const p of parts) {
    const n = lineCount(p.code);
    if (line < start + n) return `${p.name}:${line - start + 1}`;
    start += n;
  }
  return `${parts.at(-1).name}:${line}`;
}

function findNaga() {
  const bin = process.env.NAGA || 'naga';
  const r = spawnSync(bin, ['--version'], { encoding: 'utf8' });
  if (r.error || r.status !== 0) {
    console.error(
      `error: could not run \`${bin} --version\`.\n` +
        '  Install the validator:  cargo install naga-cli --version 30.0.2 --locked\n' +
        '  or point NAGA=/path/to/naga at an existing binary.',
    );
    process.exit(2);
  }
  return { bin, version: r.stdout.trim() };
}

function runNaga(naga, dir, m, composed) {
  const file = join(dir, `${m.name}.module.wgsl`);
  writeFileSync(file, composed.code);
  const r = spawnSync(naga.bin, [file], { encoding: 'utf8', env: { ...process.env, NO_COLOR: '1' } });
  if (r.status === 0) return;

  const out = `${r.stdout}${r.stderr}`.replaceAll(dir, '<module>');
  // Naga prints `┌─ <file>:<line>:<col>`; translate to the source file Naga cannot see.
  const where = [...out.matchAll(/(?:┌─|-->)\s+\S*?\.module\.wgsl:(\d+):(\d+)/g)].map((x) => locate(composed.parts, Number(x[1])));
  fail(
    `${m.name}: Naga rejected the concatenated module (exit ${r.status})\n` +
      out.trimEnd().split('\n').map((l) => `    ${l}`).join('\n') +
      (where.length ? `\n    => ${[...new Set(where)].join(', ')}` : ''),
  );
}

// --- Entry points --------------------------------------------------------------

/** Entry points declared in WGSL source: `@compute|@vertex|@fragment ... fn name`. */
function declaredEntryPoints(src) {
  const noComments = src.replace(/\/\/[^\n]*/g, '');
  return [...noComments.matchAll(/@(?:compute|vertex|fragment)\b[^{;]*?\bfn\s+(\w+)/g)].map((x) => x[1]);
}

function checkEntryPoints() {
  const known = new Set();
  for (const m of MODULES) {
    const declared = declaredEntryPoints(read(join(SHADER_DIR, `${m.name}.wgsl`)));
    for (const e of m.entryPoints) {
      if (!declared.includes(e)) fail(`${m.name}.wgsl: expected entry point \`${e}\` is not declared (missing @compute/@vertex/@fragment?)`);
      known.add(e);
    }
    for (const e of declared) {
      if (!m.entryPoints.includes(e)) fail(`${m.name}.wgsl declares entry point \`${e}\` that is not listed in MODULES`);
    }
  }
  // Pipelines are created from string names; a rename on one side only is a runtime error.
  for (const file of new Set(MODULES.map((m) => m.consumer))) {
    const ts = read(join(ROOT, file));
    const used = [...ts.matchAll(/entryPoint:\s*'(\w+)'/g), ...ts.matchAll(/\bpipe\(\s*\w+\s*,\s*'(\w+)'\s*\)/g)].map((x) => x[1]);
    for (const e of used) if (!known.has(e)) fail(`${file} creates a pipeline for entry point '${e}', which no shader module declares`);
  }
}

// --- Main ----------------------------------------------------------------------

const emitters = maxEmitters();
checkManifest();
checkEntryPoints();

const naga = findNaga();
console.log(`${naga.version}; MAX_EMITTERS=${emitters}`);
const dir = mkdtempSync(join(tmpdir(), 'wind-wgsl-'));
try {
  for (const m of MODULES) {
    const before = problems.length;
    runNaga(naga, dir, m, compose(m, emitters));
    const rel = relative(ROOT, join(SHADER_DIR, `${m.name}.wgsl`));
    console.log(`${problems.length === before ? 'ok  ' : 'FAIL'} ${m.name.padEnd(14)} (${generatedLabel(m)}common.wgsl + ${rel})`);
  }
} finally {
  rmSync(dir, { recursive: true, force: true });
}

function generatedLabel(m) {
  return m.generated ? 'generated + ' : '';
}

if (problems.length) {
  console.error(`\n${problems.length} problem(s):\n`);
  for (const p of problems) console.error(`- ${p}\n`);
  process.exit(1);
}
console.log(`\nAll ${MODULES.length} shader modules validate.`);
