// Headless acceptance test for the coupled air/droplet/thermal solver.
// Serves the repo (start `python3 -m http.server 8321` first, or pass
// WM_URL), drives test/harness.html in Chromium with software WebGPU, and
// checks the three acceptance criteria:
//   1. latent ON  -> sinking cooled column + floor outflow
//   2. latent OFF -> momentum-only updraft
//   3. no NaNs over the run
// Usage: node test/run_headless.mjs [--long]   (--long runs 60 s of sim time)
//
// Requires playwright (uses PLAYWRIGHT_MODULE or the plain `playwright`
// package) and a Chromium with WebGPU (CHROMIUM_PATH overrides).

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');

const URL = process.env.WM_URL ?? 'http://127.0.0.1:8321/test/harness.html';
const LONG = process.argv.includes('--long');
const T_ON = LONG ? 60 : 20;
const T_OFF = 20;
const SHOTS = process.env.SHOT_DIR ?? '.';

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH,
  headless: true,
  args: [
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan',
    '--use-webgpu-adapter=swiftshader',
    '--use-angle=swiftshader',
    '--disable-gpu-sandbox',
    '--no-sandbox',
  ],
});
const page = await browser.newPage();
page.on('console', (m) => {
  if (m.type() === 'error' || m.text().includes('LOST') || m.text().includes('FAILED')) {
    console.log('[page]', m.text().slice(0, 300));
  }
});

await page.goto(URL, { waitUntil: 'load' });
await page.waitForFunction(() => window.harness !== undefined, { timeout: 30000 });

const fs = await import('node:fs');
async function saveShot(overlay, name) {
  const dataUrl = await page.evaluate((ov) => window.harness.snapshot(ov), overlay);
  fs.writeFileSync(`${SHOTS}/${name}.png`,
    Buffer.from(dataUrl.split(',')[1], 'base64'));
  console.log(`saved ${name}.png`);
}

let failures = 0;
function check(name, cond, detail) {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}  (${detail})`);
  if (!cond) failures++;
}

// --- Phase 1: latent heat ON (default) — expect downdraft + outflow.
console.log(`\n=== latent ON, running to t=${T_ON}s ===`);
await page.evaluate((t) => window.harness.run(t), T_ON);
const on = await page.evaluate(() => window.harness.probe());
console.log(JSON.stringify(on, null, 1));
check('no NaN (latent on)', !on.nan && !on.lost, `nan=${on.nan} lost=${on.lost}`);
check('cooled air exists', on.minT < on.maxT - 0.5 && on.minT < 30,
  `minT=${on.minT.toFixed(2)}°C vs ambient 30°C`);
check('column sinks (downdraft)', on.vyColumn < -0.05,
  `mean vy above emitter=${on.vyColumn.toFixed(3)} m/s`);
check('floor outflow gust', on.floorOut > 0.15,
  `peak |u| near floor (>1m out)=${on.floorOut.toFixed(3)} m/s`);
check('velocities bounded', on.maxSpeed < 40, `max|u|=${on.maxSpeed.toFixed(2)} m/s`);
await saveShot(2, 'latent_on_temperature');
await saveShot(1, 'latent_on_velocity');
await saveShot(3, 'latent_on_humidity');

// --- Phase 2: latent heat OFF — momentum-only case must flip to an updraft.
console.log(`\n=== latent OFF, reset, running to t=${T_OFF}s ===`);
await page.evaluate(() => {
  window.harness.params.latentOn = false;
  window.harness.reset();
});
await page.evaluate((t) => window.harness.run(t), T_OFF);
const off = await page.evaluate(() => window.harness.probe());
console.log(JSON.stringify(off, null, 1));
check('no NaN (latent off)', !off.nan && !off.lost, `nan=${off.nan} lost=${off.lost}`);
check('no latent cooling', off.minT > 29.0, `minT=${off.minT.toFixed(2)}°C`);
check('column rises (updraft)', off.vyColumn > 0.02,
  `mean vy above emitter=${off.vyColumn.toFixed(3)} m/s`);
await saveShot(2, 'latent_off_temperature');
await saveShot(1, 'latent_off_velocity');

await browser.close();
console.log(failures === 0 ? '\nALL CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
