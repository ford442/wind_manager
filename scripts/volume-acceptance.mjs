#!/usr/bin/env node
// Headless acceptance run for the yard volume (volume.html?harness).
//
// Starts a Vite dev server, drives the page in Chromium with WebGPU, and checks
// the "done when" list of the 3D issue against probes read back from the GPU:
//   1. A vertical hose's outflow moves in +x and +z along the ground, and the
//      hero tree responds when that front arrives.
//   2. Latent heat OFF does not produce that front.
//   3. A house footprint stops velocity in 3D as in 2D: zero inside, zero
//      into every face (checked with the hose aimed at a house).
//   4. Frame cost at the default grid is reported (software WebGPU here, so it
//      is an upper bound, not the interactive number on a real GPU).
//
// Usage: npm run check:volume   [-- --seconds 20] [--dt 0.0167] [--shots DIR]
// Needs Playwright with Chromium (PLAYWRIGHT_MODULE=/path/to/playwright to
// override the import) and a WebGPU adapter. Without a GPU it uses SwiftShader.

import { mkdirSync, writeFileSync } from 'node:fs';
import { createServer } from 'vite';

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : dflt;
};
const SECONDS = Number(arg('seconds', '20'));
// Substep length (s). Defaults to the app's; a larger one runs faster in software.
const DT = arg('dt', null);
const SHOTS = arg('shots', null);
const PROBE_EVERY = 2; // s of sim time
const FRONT_SPEED = 0.15; // m/s, as src/volume/probe.ts

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');

const server = await createServer({ server: { port: 0, host: '127.0.0.1', open: false, hmr: false, watch: null }, logLevel: 'error' });
await server.listen();
const url = `${server.resolvedUrls.local[0]}volume.html?harness`;

const browser = await chromium.launch({
  headless: true,
  args: [
    '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-webgpu-adapter=swiftshader',
    '--use-angle=swiftshader', '--disable-gpu-sandbox', '--no-sandbox', '--ignore-gpu-blocklist',
  ],
});
let failures = 0;
const check = (name, ok, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  (${detail})`);
  if (!ok) failures++;
};

try {
  const page = await browser.newPage();
  page.on('console', (m) => { if (m.type() === 'error') console.log('[page]', m.text().slice(0, 400)); });
  page.on('pageerror', (e) => console.log('[page error]', e.message));
  await page.goto(url, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__wmVolume || window.__wmVolumeError, null, { timeout: 120000 });
  const err = await page.evaluate(() => window.__wmVolumeError);
  if (err) throw new Error(`page failed to start: ${err}`);
  console.log(`adapter: ${await page.evaluate(() => window.__wmVolume.adapter())}`);

  const table = await page.evaluate(() => window.__wmVolume.bufferTable());
  const limit = await page.evaluate(async () => (await navigator.gpu.requestAdapter()).limits.maxStorageBufferBindingSize);
  const big = table.reduce((a, r) => (r.bytes > a.bytes ? r : a));
  check('every buffer fits maxStorageBufferBindingSize', table.every((r) => !r.storage || r.bytes <= limit),
    `largest ${big.name} ${(big.bytes / 2 ** 20).toFixed(2)} MiB of ${(limit / 2 ** 20).toFixed(0)} MiB`);

  const shot = async (name) => {
    if (!SHOTS) return;
    mkdirSync(SHOTS, { recursive: true });
    await page.evaluate(() => window.__wmVolume.set({ showVolume: true, showDroplets: true, showGrass: true }));
    await page.evaluate(() => window.__wmVolume.run(1));
    const data = await page.evaluate(() => window.__wmVolume.snapshot());
    writeFileSync(`${SHOTS}/${name}.png`, Buffer.from(data.split(',')[1], 'base64'));
    await page.evaluate(() => window.__wmVolume.set({ showVolume: false, showDroplets: false, showGrass: false }));
    console.log(`  saved ${SHOTS}/${name}.png`);
  };

  /** Load a preset, run SECONDS of sim time, probe every PROBE_EVERY s. */
  async function experiment(preset, patch = {}) {
    await page.evaluate(([id, pt]) => {
      const v = window.__wmVolume;
      v.applyPreset(id);
      return v.set({ ...pt, showVolume: false, showDroplets: false, showGrass: false });
    }, [preset, DT ? { ...patch, dt: Number(DT) } : patch]);
    const framesPer = await page.evaluate((s) => Math.round(s / (window.__wmVolume.params.dt * window.__wmVolume.params.substeps)), PROBE_EVERY);
    const series = [];
    const t0 = Date.now();
    let frames = 0;
    for (let t = PROBE_EVERY; t <= SECONDS + 1e-6; t += PROBE_EVERY) {
      await page.evaluate((n) => window.__wmVolume.run(n), framesPer);
      frames += framesPer;
      const p = await page.evaluate(() => window.__wmVolume.probe());
      series.push(p);
      console.log(`  t=${p.time.toFixed(1).padStart(5)}s front +x ${p.front.px.toFixed(2)} +z ${p.front.pz.toFixed(2)} -x ${p.front.mx.toFixed(2)} -z ${p.front.mz.toFixed(2)} m` +
        ` | u(2m) +x ${p.outflow2m.px.toFixed(2)} +z ${p.outflow2m.pz.toFixed(2)} | cool ${p.coolingAtSource.toFixed(2)} K` +
        ` | tree outward: ground ${p.tree.ground.toFixed(2)} m/s, sway ${(p.tree.sway * 100).toFixed(2)} cm | parcels ${p.liveDrops} | div ${p.maxDivergence.toExponential(1)}`);
    }
    const msPerFrame = (Date.now() - t0) / frames;
    return { series, msPerFrame };
  }

  const hose = await page.evaluate(() => {
    const v = window.__wmVolume;
    v.applyPreset('hose-tree');
    return { x: v.params.emitters[0].x, z: v.params.emitters[0].z };
  });
  const treeDist = 4.2; // defaultHose(): 4.2 m in -x of the hero tree
  console.log(`\n=== latent ON: vertical hose at x=${hose.x.toFixed(2)} z=${hose.z.toFixed(2)}, hero tree ${treeDist} m in +x ===`);
  const on = await experiment('hose-tree');
  await shot('volume-latent-on');
  const endOn = on.series.at(-1);
  const reach = 3;
  check('outflow front moves in +x along the ground', endOn.front.px >= reach, `+x front ${endOn.front.px.toFixed(2)} m`);
  check('outflow front moves in +z along the ground', endOn.front.pz >= reach, `+z front ${endOn.front.pz.toFixed(2)} m`);
  check('the hose cools the air at the ground', endOn.coolingAtSource > 0.5, `${endOn.coolingAtSource.toFixed(2)} K below ambient`);
  // Tree wind and sway are measured along hose → tree: the front pushes the
  // crown away from the hose; entrainment into the spray draws it back.
  // Arrival is judged at the tree: its ground wind passes the front speed.
  const calm = on.series.filter((p) => p.tree.ground < 0.05);
  const arrive = on.series.find((p) => p.tree.ground >= FRONT_SPEED);
  const after = arrive ? on.series.filter((p) => p.time >= arrive.time) : [];
  const swayCalm = Math.max(0, ...calm.map((p) => p.tree.sway));
  const swayAfter = Math.max(0, ...after.map((p) => p.tree.sway));
  check('front reaches the hero tree', Boolean(arrive) && on.series.some((p) => p.front.px >= treeDist - 0.6),
    arrive ? `ground wind at the tree ${arrive.tree.ground.toFixed(2)} m/s at t=${arrive.time.toFixed(1)} s` : 'never');
  check('hero tree responds when the front arrives', calm.length > 0 && swayAfter > Math.max(0.01, 3 * swayCalm),
    `outward crown sway ≤ ${(swayCalm * 100).toFixed(2)} cm while its base is calm, ${(swayAfter * 100).toFixed(2)} cm after the front arrives`);
  check('fields stay finite (latent ON)', on.series.every((p) => p.finite), 'all probes');

  console.log('\n=== latent OFF: same hose ===');
  const off = await experiment('hose-tree-dry');
  await shot('volume-latent-off');
  const endOff = off.series.at(-1);
  const maxOffX = Math.max(...off.series.map((p) => p.front.px));
  const maxOffZ = Math.max(...off.series.map((p) => p.front.pz));
  check('latent OFF does not produce the +x front', maxOffX < 0.5 * endOn.front.px, `max +x front ${maxOffX.toFixed(2)} m vs ${endOn.front.px.toFixed(2)} m with latent ON`);
  check('latent OFF does not produce the +z front', maxOffZ < 0.5 * endOn.front.pz, `max +z front ${maxOffZ.toFixed(2)} m vs ${endOn.front.pz.toFixed(2)} m with latent ON`);
  const swayOff = Math.max(0, ...off.series.map((p) => p.tree.sway));
  check('latent OFF does not push the hero tree outward', swayOff < 0.3 * swayAfter,
    `outward crown sway ${(swayOff * 100).toFixed(2)} cm vs ${(swayAfter * 100).toFixed(2)} cm with latent ON`);
  check('no cooling with latent OFF', endOff.coolingAtSource < 0.2, `${endOff.coolingAtSource.toFixed(2)} K`);

  console.log('\n=== hose aimed at the left house ===');
  const house = await experiment('house-spray');
  await shot('volume-house');
  const worst = house.series.reduce((m, p) => ({
    inside: Math.max(m.inside, p.solid.maxInside), face: Math.max(m.face, p.solid.maxIntoFace),
  }), { inside: 0, face: 0 });
  const cells = house.series.at(-1).solid.cells;
  check('solid cells exist in the mask', cells > 0, `${cells} cells`);
  check('no velocity inside houses or trunks', worst.inside === 0, `max |u| inside ${worst.inside}`);
  check('no velocity into any solid face', worst.face <= 1e-6, `max into-face ${worst.face.toExponential(2)} m/s`);
  // Approximate projection: the face divergence of cell-averaged velocities
  // is not zero, but must stay well below the flow's own rate of strain.
  check('face divergence after projection stays small', house.series.every((p) => p.maxDivergence < 0.5),
    `max ${Math.max(...house.series.map((p) => p.maxDivergence)).toExponential(2)} /s`);

  console.log(`\nframe cost on this adapter: ${on.msPerFrame.toFixed(0)} ms/frame (solver + scene update, overlays off)`);
} finally {
  await browser.close();
  await server.close();
}

if (failures) {
  console.log(`\n${failures} check(s) failed.`);
  process.exit(1);
}
console.log('\nAll volume acceptance checks pass.');
