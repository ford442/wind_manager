import type { Params } from './sim/params';
import { defaultParams } from './sim/params';
import type { Fields } from './sim/fields';
import { createFields } from './sim/fields';
import type { Droplets } from './sim/droplets';
import { createDroplets } from './sim/droplets';
import type { Sim } from './sim/step';
import { createSim } from './sim/step';
import type { Renderer } from './render/overlays';
import { createRenderer } from './render/overlays';
import { setupControls } from './ui/controls';

const DIAG_INTERVAL = 30;

function fail(msg) {
  const el = document.getElementById('gpu-error');
  el.style.display = 'block';
  el.querySelector('.msg').textContent = msg;
  document.getElementById('status').textContent = 'unavailable';
}

async function init(): Promise<void> {
  if (!navigator.gpu) {
    fail('This browser does not expose WebGPU (navigator.gpu is missing). ' +
         'Use a recent Chrome/Edge, or enable WebGPU in your browser.');
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    fail('No WebGPU adapter available on this machine.');
    return;
  }
  const device = await adapter.requestDevice();
  device.addEventListener('uncapturederror', (e) => {
    console.error('[wind_manager] WebGPU error:', e.error.message);
    document.getElementById('nan-status').textContent = 'GPU error (see console)';
    document.getElementById('nan-status').classList.add('bad');
  });

  const canvas = document.getElementById('c') as HTMLCanvasElement;
  const ctx = canvas.getContext('webgpu')!;
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'opaque' });

  const params: Params = defaultParams();
  const fields: Fields = createFields(device, params);
  const drops: Droplets = createDroplets(device, params);
  const sim: Sim = await createSim(device, fields, drops, params);
  const renderer: Renderer = await createRenderer(device, format, fields, drops, params);

  sim.reset(params);
  setupControls(params, {
    onReset: () => sim.reset(params),
  });

  const n = params.nx * params.ny;
  const stagT = device.createBuffer({
    size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const stagV = device.createBuffer({
    size: n * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const stagC = device.createBuffer({
    size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let diagBusy = false;

  async function runDiagnostics() {
    diagBusy = true;
    try {
      const f = sim.getFields();
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(f.T0, 0, stagT, 0, n * 4);
      enc.copyBufferToBuffer(f.vel0, 0, stagV, 0, n * 8);
      enc.copyBufferToBuffer(drops.counter, 0, stagC, 0, 4);
      device.queue.submit([enc.finish()]);

      await Promise.all([
        stagT.mapAsync(GPUMapMode.READ),
        stagV.mapAsync(GPUMapMode.READ),
        stagC.mapAsync(GPUMapMode.READ),
      ]);
      const t = new Float32Array(stagT.getMappedRange());
      const v = new Float32Array(stagV.getMappedRange());
      const emitted = new Uint32Array(stagC.getMappedRange())[0];

      let minT = Infinity, maxT = -Infinity, maxSp2 = 0, nan = false;
      for (let i = 0; i < n; i++) {
        const ti = t[i];
        if (Number.isNaN(ti)) { nan = true; break; }
        if (ti < minT) minT = ti;
        if (ti > maxT) maxT = ti;
        const vx = v[2 * i], vy = v[2 * i + 1];
        if (Number.isNaN(vx) || Number.isNaN(vy)) { nan = true; break; }
        const s2 = vx * vx + vy * vy;
        if (s2 > maxSp2) maxSp2 = s2;
      }
      stagT.unmap(); stagV.unmap(); stagC.unmap();

      const $ = (id) => document.getElementById(id);
      const nanEl = $('nan-status');
      if (nan) {
        nanEl.textContent = 'NaN DETECTED';
        nanEl.classList.add('bad');
      } else {
        nanEl.textContent = 'fields finite';
        nanEl.classList.remove('bad');
        $('t-range').textContent = `${minT.toFixed(2)} … ${maxT.toFixed(2)} °C`;
        $('max-speed').textContent = `${Math.sqrt(maxSp2).toFixed(2)} m/s`;
      }
      $('emitted').textContent = emitted.toLocaleString();
    } finally {
      diagBusy = false;
    }
  }

  let lastFps = performance.now();
  let framesSinceFps = 0;

  function frame() {
    if (!params.paused) {
      sim.step(params);
    }
    renderer.draw(
      ctx.getCurrentTexture().createView(),
      params,
      sim.getFields(),
      canvas.width,
      canvas.height,
    );

    framesSinceFps++;
    const now = performance.now();
    if (now - lastFps > 500) {
      document.getElementById('fps').textContent =
        ((framesSinceFps * 1000) / (now - lastFps)).toFixed(0);
      framesSinceFps = 0;
      lastFps = now;
    }
    document.getElementById('sim-time').textContent = `${sim.time.toFixed(1)} s`;

    if (sim.frame % DIAG_INTERVAL === 0 && !diagBusy && !params.paused) {
      runDiagnostics().catch((e) => console.error('diagnostics failed', e));
    }
    requestAnimationFrame(frame);
  }

  document.getElementById('status').textContent = 'running';
  requestAnimationFrame(frame);
}

init().catch((e) => {
  console.error(e);
  fail(String(e && e.message ? e.message : e));
});
