// main.js — WebGPU init, frame loop, and lightweight diagnostics
// (NaN watch + field extrema, read back every 30 frames).

import { defaultParams } from './sim/params.js';
import { createFields } from './sim/fields.js';
import { createDroplets } from './sim/droplets.js';
import { createSim } from './sim/step.js';
import { createRenderer } from './render/overlays.js';
import { setupControls } from './ui/controls.js';

const DIAG_INTERVAL = 30; // frames

function fail(msg) {
  const el = document.getElementById('gpu-error');
  el.style.display = 'block';
  el.querySelector('.msg').textContent = msg;
  document.getElementById('status').textContent = 'unavailable';
}

async function init() {
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
  // Keep the adapter alive: if it gets GC'd, Chromium can drop the Dawn
  // instance and later mapAsync calls fail ("external Instance reference
  // no longer exists").
  window.__wgpuAdapter = adapter;
  device.lost.then((info) => {
    console.error('[wind_manager] GPU device lost:', info.reason, info.message);
    fail(`GPU device lost: ${info.message}`);
  });
  device.addEventListener('uncapturederror', (e) => {
    console.error('[wind_manager] WebGPU error:', e.error.message);
    document.getElementById('nan-status').textContent = 'GPU error (see console)';
    document.getElementById('nan-status').classList.add('bad');
  });

  const canvas = document.getElementById('c');
  const ctx = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'opaque' });

  const params = defaultParams();
  const fields = createFields(device, params);
  const drops = createDroplets(device, params);
  const sim = await createSim(device, fields, drops, params);
  const renderer = await createRenderer(device, format, fields, drops, params);

  sim.reset(params);
  setupControls(params, {
    onReset: () => sim.reset(params),
  });

  // --- diagnostics: staged readback of T, vel and the emit counter.
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
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(fields.T0, 0, stagT, 0, n * 4);
      enc.copyBufferToBuffer(fields.vel0, 0, stagV, 0, n * 8);
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

  // --- frame loop
  let lastFps = performance.now();
  let framesSinceFps = 0;

  function frame() {
    if (!params.paused) {
      sim.step(params);
    }
    renderer.draw(ctx.getCurrentTexture().createView(), params,
                  canvas.width, canvas.height);

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

  // Debug/verification hook: reads back the fields and summarizes the
  // physics — mean vertical velocity in the column above the emitter,
  // horizontal outflow speed along the floor, and the temperature minimum.
  window.__wm_debug = {
    params,
    sim,
    probe: async () => {
      // Dedicated staging buffers so a probe never collides with an
      // in-flight diagnostics readback on the shared ones.
      const pT = device.createBuffer({
        size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const pV = device.createBuffer({
        size: n * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(fields.T0, 0, pT, 0, n * 4);
      enc.copyBufferToBuffer(fields.vel0, 0, pV, 0, n * 8);
      device.queue.submit([enc.finish()]);
      await Promise.all([
        pT.mapAsync(GPUMapMode.READ),
        pV.mapAsync(GPUMapMode.READ),
      ]);
      const t = new Float32Array(pT.getMappedRange()).slice();
      const v = new Float32Array(pV.getMappedRange()).slice();
      pT.destroy(); pV.destroy();

      const h = params.domainW / params.nx;
      const ci = Math.round(params.emitX / h);
      // column: x within ±0.25 m of emitter, y in [1.0, 3.0] m
      let vySum = 0, vyN = 0, minT = Infinity, colT = Infinity;
      const di = Math.round(0.25 / h);
      for (let j = Math.round(1.0 / h); j < Math.round(3.0 / h); j++) {
        for (let i = ci - di; i <= ci + di; i++) {
          const k = j * params.nx + i;
          vySum += v[2 * k + 1];
          vyN++;
          if (t[k] < colT) colT = t[k];
        }
      }
      for (let k = 0; k < n; k++) if (t[k] < minT) minT = t[k];
      // floor outflow: max |u.x| in the lowest 0.3 m, at least 1 m from emitter
      let floorOut = 0;
      for (let j = 0; j < Math.round(0.3 / h); j++) {
        for (let i = 0; i < params.nx; i++) {
          if (Math.abs(i - ci) * h < 1.0) continue;
          const ux = Math.abs(v[2 * (j * params.nx + i)]);
          if (ux > floorOut) floorOut = ux;
        }
      }
      return {
        vyColumn: vySum / vyN,
        floorOut,
        minT,
        colMinT: colT,
        time: sim.time,
      };
    },
  };

  document.getElementById('status').textContent = 'running';
  requestAnimationFrame(frame);
}

init().catch((e) => {
  console.error(e);
  fail(String(e && e.message ? e.message : e));
});
