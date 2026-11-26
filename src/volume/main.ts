import { GpuLimitError, requestGpuDeviceWithLimits, type NumericLimit } from '../render/gpuDevice';
import { configureWebGPUCanvas, getWebGPUContext } from '../render/webgpu';
import { setupCanvasResize } from '../render/canvasSize';
import { YardCamera } from './camera';
import { createVolumeFields, type VolumeFields } from './fields';
import {
  VOLUME_GRIDS, checkVolumeBuffers, formatBytes, gridById, largestVolumeBuffer, volumeBufferTable,
  type BufferRow,
} from './grid';
import {
  MAX_EMITTERS, VOLUME_PRESETS, airEmitter, clampVolumeEmitter, defaultVolumeParams, waterEmitter, waterFlowGs,
  type VolumeEmitter, type VolumeParams,
} from './params';
import { createProbeBuffers, probeVolume, type ProbeBuffers, type VolumeProbe } from './probe';
import { createYardRenderer, viewports, type YardRenderer } from './scene';
import { createVolumeSim, type VolumeSim } from './step';

/**
 * Yard volume app (volume.html). One canvas, one command buffer per frame:
 * solver substeps, tree-sway update, opaque yard pass, raymarch + side view.
 *
 * `?harness` renders into an offscreen texture, does not start the frame loop
 * and exposes `window.__wmVolume` for scripts/volume-acceptance.mjs.
 */

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} missing`);
  return el as T;
};

function fail(msg: string): void {
  const el = $('gpu-error');
  el.style.display = 'block';
  el.querySelector('.msg')!.textContent = msg;
  $('hud').textContent = 'unavailable';
}

/** Limits the volume's shaders and buffers depend on (same policy as the slice). */
function volumeLimits(largest: number, maxDroplets: number): Record<NumericLimit, number> {
  return {
    // vol_update binds 8 storage buffers; scene vertex/fragment stages bind 7.
    maxStorageBuffersPerShaderStage: 8,
    maxStorageBufferBindingSize: largest,
    maxBufferSize: largest,
    maxUniformBuffersPerShaderStage: 2,
    maxBindingsPerBindGroup: 11,
    maxComputeWorkgroupsPerDimension: Math.ceil(maxDroplets / 128),
    maxComputeInvocationsPerWorkgroup: 128,
    maxComputeWorkgroupSizeX: 128,
  };
}

interface HarnessApi {
  params: VolumeParams;
  bufferTable: () => BufferRow[];
  applyPreset: (id: string) => void;
  set: (patch: Partial<VolumeParams>) => Promise<void>;
  reset: () => void;
  /** Advance `frames` frames (solver + scene update + draw), then wait for the GPU. */
  run: (frames: number) => Promise<void>;
  probe: () => Promise<VolumeProbe>;
  snapshot: () => Promise<string>;
  adapter: () => string;
  /** Raw readback of the current velocity (xyz + pad) and temperature, for debugging. */
  dump: () => Promise<{ vel: number[]; T: number[]; p: number[]; rhs: number[]; solid: number[] }>;
}

declare global {
  interface Window { __wmVolume?: HarnessApi; __wmVolumeError?: string }
}

/** Two timestamp pairs (solver, scene), resolved and read back every ~30 frames. */
function createGpuTimer(device: GPUDevice) {
  const querySet = device.createQuerySet({ type: 'timestamp', count: 4 });
  const resolveBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  let frame = 0;
  let pending = false;
  let solverTimed = false;
  return {
    solver: { querySet, begin: 0, end: 1 },
    scene: { querySet, begin: 2, end: 3 },
    ready: () => !pending && ++frame % 30 === 0,
    resolve(enc: GPUCommandEncoder, withSolver: boolean) {
      solverTimed = withSolver;
      enc.resolveQuerySet(querySet, 0, 4, resolveBuf, 0);
      enc.copyBufferToBuffer(resolveBuf, 0, readBuf, 0, 32);
      pending = true;
    },
    read(done: (ms: { solver: number; scene: number }) => void) {
      if (!pending || readBuf.mapState !== 'unmapped') return;
      readBuf.mapAsync(GPUMapMode.READ).then(() => {
        const t = new BigInt64Array(readBuf.getMappedRange().slice(0));
        readBuf.unmap();
        pending = false;
        const ms = (a: bigint, b: bigint) => Math.max(0, Number(b - a) / 1e6);
        done({ solver: solverTimed ? ms(t[0], t[1]) : 0, scene: ms(t[2], t[3]) });
      }).catch(() => { pending = false; });
    },
  };
}

async function init(): Promise<void> {
  const harness = new URLSearchParams(location.search).has('harness');
  const params = defaultVolumeParams();
  const camera = new YardCamera();

  let running = false;
  let raf = 0;
  const largest = largestVolumeBuffer(params.maxDroplets);
  let adapterInfo = '';
  const gpu = await requestGpuDeviceWithLimits(
    volumeLimits(largest, params.maxDroplets),
    (limits) => {
      // Every buffer of the selected grid, checked before anything is allocated.
      const problem = checkVolumeBuffers(volumeBufferTable(gridById(params.gridId), params.maxDroplets), limits);
      if (problem) throw new GpuLimitError(problem.limit, problem.row.bytes, problem.available);
    },
  ).catch(async (e: unknown) => {
    // The largest offered grid may not fit: retry with the selected grid only.
    if (!(e instanceof GpuLimitError) || e.limit !== 'maxStorageBufferBindingSize' && e.limit !== 'maxBufferSize') throw e;
    const own = Math.max(...volumeBufferTable(gridById(params.gridId), params.maxDroplets).map((r) => r.bytes));
    if (own >= largest) throw e;
    return requestGpuDeviceWithLimits(volumeLimits(own, params.maxDroplets));
  });
  const device = gpu.device;
  adapterInfo = `${gpu.adapter.info?.vendor ?? ''} ${gpu.adapter.info?.architecture ?? ''} ${gpu.adapter.info?.description ?? ''}`.trim();
  device.lost.then((info) => fail(`GPU device lost (${info.reason}). Reload the page.`));
  device.addEventListener('uncapturederror', (ev: GPUUncapturedErrorEvent) => {
    console.error('[wind_manager volume] WebGPU error:', ev.error.message);
    window.__wmVolumeError = ev.error.message;
    stop();
    fail(`GPU error: ${ev.error.message}`);
  });

  const canvas = $<HTMLCanvasElement>('c');
  const format = navigator.gpu.getPreferredCanvasFormat();
  let context: GPUCanvasContext | null = null;
  let offscreen: GPUTexture | null = null;
  if (harness) {
    // Headless SwiftShader cannot present a canvas; draw into a texture instead.
    canvas.width = 960;
    canvas.height = 600;
    offscreen = device.createTexture({
      size: [canvas.width, canvas.height], format: 'rgba8unorm',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
  } else {
    context = getWebGPUContext(canvas);
    configureWebGPUCanvas(context, device, format);
    setupCanvasResize({
      container: canvas.parentElement as HTMLElement,
      canvases: [canvas],
      onResize: () => configureWebGPUCanvas(context!, device, format),
    });
  }

  let fields: VolumeFields = createVolumeFields(device, params);
  const sim: VolumeSim = await createVolumeSim(device, fields);
  const renderer: YardRenderer = await createYardRenderer(device, offscreen ? 'rgba8unorm' : format);
  let probeBufs: ProbeBuffers = createProbeBuffers(device, fields, params.maxDroplets);

  function reset(): void {
    sim.reset(params);
    renderer.reset();
  }

  function rebuildGrid(): boolean {
    const table = volumeBufferTable(gridById(params.gridId), params.maxDroplets);
    const problem = checkVolumeBuffers(table, device.limits);
    if (problem) {
      alert(`${problem.row.name} needs ${formatBytes(problem.row.bytes)}; this device allows ${formatBytes(problem.available)} (${problem.limit}).`);
      return false;
    }
    fields.destroy();
    probeBufs.destroy();
    fields = createVolumeFields(device, params);
    probeBufs = createProbeBuffers(device, fields, params.maxDroplets);
    sim.setFields(fields);
    reset();
    renderBufferTable();
    return true;
  }

  reset();

  // --- Frame ----------------------------------------------------------------

  let last = performance.now();
  let fpsT = last;
  let fpsN = 0;
  let fps = 0;
  let frameMs = 0;
  let probe: VolumeProbe | null = null;

  // GPU time of the solver substeps and of the scene, when the adapter has
  // timestamp queries. This is the number to read for "interactive at the
  // budget" on real hardware (frame ms alone includes vsync waits).
  const gpuTimer = gpu.timestampQuery ? createGpuTimer(device) : null;
  let gpuMs: { solver: number; scene: number } | null = null;
  let probeBusy = false;
  let lastProbe = 0;

  function recordFrame(dt: number): GPUCommandEncoder {
    const enc = device.createCommandEncoder({ label: 'vol frame' });
    const timed = gpuTimer?.ready() ?? false;
    if (!params.paused) sim.step(params, enc, timed ? gpuTimer!.solver : undefined);
    const target = offscreen ? offscreen.createView() : context!.getCurrentTexture().createView();
    renderer.draw({
      enc, target, width: canvas.width, height: canvas.height, fields, params, camera,
      time: sim.time, dt: params.paused ? 0 : dt, timing: timed ? gpuTimer!.scene : undefined,
    });
    if (timed) gpuTimer!.resolve(enc, !params.paused);
    return enc;
  }

  function frame(): void {
    if (!running) return;
    const now = performance.now();
    const dt = Math.min(0.05, (now - last) / 1000);
    frameMs = frameMs * 0.9 + (now - last) * 0.1;
    last = now;
    handleWalkKeys(dt);
    device.queue.submit([recordFrame(params.substeps * params.dt).finish()]);
    gpuTimer?.read((ms) => { gpuMs = ms; });
    fpsN++;
    if (now - fpsT > 500) {
      fps = (fpsN * 1000) / (now - fpsT);
      fpsN = 0;
      fpsT = now;
    }
    if (!probeBusy && now - lastProbe > 1000 && !params.paused) {
      probeBusy = true;
      lastProbe = now;
      probeVolume(device, probeBufs, fields, renderer.state, params, sim.time)
        .then((r) => { probe = r; })
        .catch((e: unknown) => console.warn('probe failed', e))
        .finally(() => { probeBusy = false; });
    }
    renderHud();
    raf = requestAnimationFrame(frame);
  }

  function start(): void {
    if (running || harness || document.hidden) return;
    running = true;
    last = performance.now();
    raf = requestAnimationFrame(frame);
  }
  function stop(): void {
    running = false;
    cancelAnimationFrame(raf);
  }
  document.addEventListener('visibilitychange', () => (document.hidden ? stop() : start()));

  function renderHud(): void {
    const g = fields.grid;
    const lines = [
      `${fps.toFixed(0)} fps · ${frameMs.toFixed(1)} ms/frame · t = ${sim.time.toFixed(1)} s`,
      `grid ${g.nx}×${g.ny}×${g.nz} · latent ${params.latentOn ? 'ON' : 'OFF'} · ${params.pressureCycles} V-cycles`,
      gpuMs
        ? `GPU: solver ${gpuMs.solver.toFixed(2)} ms · scene ${gpuMs.scene.toFixed(2)} ms`
        : (gpuTimer ? 'GPU: timing…' : 'GPU timing unavailable (no timestamp-query)'),
    ];
    if (probe) {
      const p = probe;
      lines.push(
        `ground front +x ${p.front.px.toFixed(1)} m · +z ${p.front.pz.toFixed(1)} m · −x ${p.front.mx.toFixed(1)} · −z ${p.front.mz.toFixed(1)}`,
        `cooling at hose ${p.coolingAtSource.toFixed(2)} K · min T ${p.minT.toFixed(2)} °C`,
        `hero tree, outward from the hose: ground wind ${p.tree.ground.toFixed(2)} m/s · crown sway ${(p.tree.sway * 100).toFixed(1)} cm`,
        `max |u| ${p.maxSpeed.toFixed(2)} m/s · max div ${p.maxDivergence.toExponential(1)} /s · drops ${p.liveDrops.toLocaleString()}`,
      );
      if (!p.finite) lines.push('<span class="bad">NaN in fields</span>');
    }
    $('hud').innerHTML = lines.join('<br>');
  }

  // --- Camera input -----------------------------------------------------------

  const keys = new Set<string>();
  window.addEventListener('keydown', (e) => {
    if ((e.target as HTMLElement).tagName === 'INPUT') return;
    keys.add(e.key.toLowerCase());
  });
  window.addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));
  function handleWalkKeys(dt: number): void {
    if (camera.mode !== 'walk') return;
    const speed = 3 * dt;
    const f = (keys.has('w') || keys.has('arrowup') ? 1 : 0) - (keys.has('s') || keys.has('arrowdown') ? 1 : 0);
    const s = (keys.has('d') || keys.has('arrowright') ? 1 : 0) - (keys.has('a') || keys.has('arrowleft') ? 1 : 0);
    if (f || s) camera.move(f * speed, s * speed);
  }
  let drag: { x: number; y: number; pan: boolean } | null = null;
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey };
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x;
    const dy = e.clientY - drag.y;
    drag.x = e.clientX;
    drag.y = e.clientY;
    if (drag.pan) camera.pan(dx, dy);
    else camera.rotate(dx, dy);
  });
  canvas.addEventListener('pointerup', () => { drag = null; });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    camera.zoom(e.deltaY);
  }, { passive: false });
  canvas.addEventListener('dblclick', (e) => {
    const vp = viewports(params, canvas.width, canvas.height).yard;
    const em = params.emitters[selected];
    if (!vp || !em) return;
    const rect = canvas.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const py = ((e.clientY - rect.top) / rect.height) * canvas.height;
    if (px < vp[0] || py < vp[1] || px > vp[0] + vp[2] || py > vp[1] + vp[3]) return;
    const ndcX = ((px - vp[0]) / vp[2]) * 2 - 1;
    const ndcY = 1 - ((py - vp[1]) / vp[3]) * 2;
    const ray = camera.ray(ndcX, ndcY, vp[2] / vp[3]);
    if (ray.d[1] >= -1e-3) return;
    const t = -ray.o[1] / ray.d[1];
    em.x = ray.o[0] + ray.d[0] * t;
    em.z = ray.o[2] + ray.d[2] * t;
    clampVolumeEmitter(em);
    params.sideZ = em.z;
    syncUi();
  });

  // --- Panel ------------------------------------------------------------------

  let selected = 0;
  const presetSel = $<HTMLSelectElement>('preset');
  for (const pr of VOLUME_PRESETS) presetSel.add(new Option(pr.label, pr.id));
  const gridSel = $<HTMLSelectElement>('grid');
  for (const g of VOLUME_GRIDS) gridSel.add(new Option(g.label, g.id));

  function applyPreset(id: string): void {
    VOLUME_PRESETS.find((p) => p.id === id)?.apply(params);
    selected = 0;
    reset();
    syncUi();
  }

  const bindCheck = (id: string, get: () => boolean, set: (v: boolean) => void) => {
    const el = $<HTMLInputElement>(id);
    el.addEventListener('change', () => { set(el.checked); syncUi(); });
    return () => { el.checked = get(); };
  };
  const bindRange = (id: string, get: () => number, set: (v: number) => void, fmt: (v: number) => string) => {
    const el = $<HTMLInputElement>(id);
    el.addEventListener('input', () => { set(Number(el.value)); syncUi(); });
    return () => { el.value = String(get()); $(`${id}-v`).textContent = fmt(get()); };
  };
  const bindSelect = <T extends string>(id: string, get: () => T, set: (v: T) => void) => {
    const el = $<HTMLSelectElement>(id);
    el.addEventListener('change', () => { set(el.value as T); syncUi(); });
    return () => { el.value = get(); };
  };

  const syncers = [
    bindCheck('latent', () => params.latentOn, (v) => { params.latentOn = v; }),
    bindCheck('solids', () => params.solidObstacles, (v) => {
      params.solidObstacles = v;
      fields.writeSolids(device.queue, params);
    }),
    bindRange('tamb', () => params.tAmb, (v) => { params.tAmb = v; }, (v) => `${v} °C`),
    bindRange('rh', () => params.rhAmb, (v) => { params.rhAmb = v; }, (v) => `${v} %`),
    bindSelect('grid', () => params.gridId, (v) => {
      const prev = params.gridId;
      params.gridId = v;
      if (!rebuildGrid()) params.gridId = prev;
    }),
    bindSelect('view-mode', () => params.viewMode, (v) => { params.viewMode = v; }),
    bindSelect('camera-mode', () => camera.mode, (v) => { camera.mode = v; params.cameraMode = v; }),
    bindCheck('show-volume', () => params.showVolume, (v) => { params.showVolume = v; }),
    bindRange('contrast', () => params.volumeContrast, (v) => { params.volumeContrast = v; }, (v) => `${v}`),
    bindCheck('show-drops', () => params.showDroplets, (v) => { params.showDroplets = v; }),
    bindCheck('show-grass', () => params.showGrass, (v) => { params.showGrass = v; }),
    bindRange('side-z', () => params.sideZ, (v) => { params.sideZ = v; }, (v) => v.toFixed(2)),
    bindSelect('side-field', () => params.sideField, (v) => { params.sideField = v; }),
  ];

  $('load-preset').addEventListener('click', () => applyPreset(presetSel.value));
  $('reset').addEventListener('click', reset);
  $('pause').addEventListener('click', () => {
    params.paused = !params.paused;
    $('pause').textContent = params.paused ? 'Resume' : 'Pause';
  });
  $('add-water').addEventListener('click', () => addEmitter(waterEmitter(camera.target[0], camera.target[2])));
  $('add-air').addEventListener('click', () => addEmitter(airEmitter(camera.target[0], camera.target[2])));
  $('remove-emitter').addEventListener('click', () => {
    if (params.emitters.length === 0) return;
    params.emitters.splice(selected, 1);
    selected = Math.max(0, Math.min(selected, params.emitters.length - 1));
    syncUi();
  });
  $<HTMLSelectElement>('emitter-select').addEventListener('change', (e) => {
    selected = Number((e.target as HTMLSelectElement).value);
    syncUi();
  });

  function addEmitter(em: VolumeEmitter): void {
    if (params.emitters.length >= MAX_EMITTERS) return;
    clampVolumeEmitter(em);
    params.emitters.push(em);
    selected = params.emitters.length - 1;
    syncUi();
  }

  const EMITTER_FIELDS: [keyof VolumeEmitter, string, number, number, number][] = [
    ['x', 'x (m)', 0, 32, 0.1], ['z', 'z (m)', 0, 16, 0.1], ['y', 'height (m)', 0.1, 7, 0.05],
    ['size', 'nozzle radius (m)', 0.02, 1.5, 0.02], ['angleDeg', 'tilt from vertical (°)', 0, 180, 1],
    ['azimuthDeg', 'azimuth (°, 0 = +x)', -180, 180, 1], ['spreadDeg', 'cone half-angle (°)', 0, 60, 1],
    ['speed', 'speed (m/s)', 0, 30, 0.5], ['rate', 'parcels or impulses (/s)', 0, 150000, 1000],
    ['dropsPerParcel', 'drops per parcel', 1, 500, 1],
    ['rMinUm', 'smallest drop (µm)', 5, 2000, 5], ['rMaxUm', 'largest drop (µm)', 5, 2000, 5],
  ];
  function renderEmitterFields(): void {
    const sel = $<HTMLSelectElement>('emitter-select');
    sel.innerHTML = '';
    params.emitters.forEach((em, i) => sel.add(new Option(`${i + 1}: ${em.type === 'water' ? 'hose' : 'fan'}`, String(i))));
    sel.value = String(selected);
    const host = $('emitter-fields');
    host.innerHTML = '';
    const em = params.emitters[selected];
    if (!em) return;
    if (em.type === 'water') {
      const flow = document.createElement('div');
      flow.className = 'hint';
      flow.textContent = `Water flow ≈ ${waterFlowGs(em).toFixed(0)} g/s (${(waterFlowGs(em) * 0.06).toFixed(1)} L/min)`;
      host.append(flow);
    }
    for (const [key, label, min, max, step] of EMITTER_FIELDS) {
      if (em.type === 'air' && (key === 'dropsPerParcel' || key === 'rMinUm' || key === 'rMaxUm')) continue;
      const row = document.createElement('label');
      row.className = 'row';
      row.textContent = label;
      const input = document.createElement('input');
      input.type = 'number';
      Object.assign(input, { min, max, step, value: String(Number(em[key]).toFixed(step < 1 ? 2 : 0)) });
      input.addEventListener('change', () => {
        (em as unknown as Record<string, number>)[key] = Number(input.value);
        clampVolumeEmitter(em);
        if (key === 'z') params.sideZ = em.z;
        syncUi();
      });
      row.append(input);
      host.append(row);
    }
  }

  function renderBufferTable(): void {
    const rows = volumeBufferTable(fields.grid, params.maxDroplets);
    const lim = device.limits.maxStorageBufferBindingSize;
    $('buffer-table').innerHTML = `<table>${rows.map((r) =>
      `<tr><td>${r.name}</td><td class="num${r.bytes > lim ? ' bad' : ''}">${formatBytes(r.bytes)}</td></tr>`).join('')}
      <tr><td>maxStorageBufferBindingSize</td><td class="num">${formatBytes(lim)}</td></tr></table>`;
  }

  function syncUi(): void {
    for (const s of syncers) s();
    renderEmitterFields();
  }
  syncUi();
  renderBufferTable();

  if (harness) {
    window.__wmVolume = {
      params,
      bufferTable: () => volumeBufferTable(fields.grid, params.maxDroplets),
      applyPreset,
      async set(patch) {
        const gridChanged = patch.gridId !== undefined && patch.gridId !== params.gridId;
        const solidsChanged = patch.solidObstacles !== undefined && patch.solidObstacles !== params.solidObstacles;
        Object.assign(params, patch);
        if (gridChanged) rebuildGrid();
        else if (solidsChanged) fields.writeSolids(device.queue, params);
        await device.queue.onSubmittedWorkDone();
      },
      reset,
      async run(frames) {
        for (let i = 0; i < frames; i++) {
          device.queue.submit([recordFrame(params.substeps * params.dt).finish()]);
          if (i % 8 === 7) await device.queue.onSubmittedWorkDone();
        }
        await device.queue.onSubmittedWorkDone();
      },
      probe: () => probeVolume(device, probeBufs, fields, renderer.state, params, sim.time),
      async snapshot() {
        const w = canvas.width, h = canvas.height;
        const bpr = Math.ceil((w * 4) / 256) * 256;
        const buf = device.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const enc = device.createCommandEncoder();
        enc.copyTextureToBuffer({ texture: offscreen! }, { buffer: buf, bytesPerRow: bpr }, [w, h]);
        device.queue.submit([enc.finish()]);
        await buf.mapAsync(GPUMapMode.READ);
        const src = new Uint8Array(buf.getMappedRange());
        const img = new ImageData(w, h);
        for (let y = 0; y < h; y++) img.data.set(src.subarray(y * bpr, y * bpr + w * 4), y * w * 4);
        buf.unmap();
        buf.destroy();
        const c2 = document.createElement('canvas');
        c2.width = w;
        c2.height = h;
        c2.getContext('2d')!.putImageData(img, 0, 0);
        return c2.toDataURL('image/png');
      },
      adapter: () => adapterInfo,
      async dump() {
        const read = async (src: GPUBuffer, bytes: number, u32 = false) => {
          const b = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
          const enc = device.createCommandEncoder();
          enc.copyBufferToBuffer(src, 0, b, 0, bytes);
          device.queue.submit([enc.finish()]);
          await b.mapAsync(GPUMapMode.READ);
          const m = b.getMappedRange();
          const out = Array.from(u32 ? new Uint32Array(m) : new Float32Array(m));
          b.destroy();
          return out;
        };
        const lv = fields.pressure[0];
        return {
          vel: await read(fields.vel0, fields.n * 16),
          T: await read(fields.T0, fields.n * 4),
          p: await read(lv.p0, fields.n * 4),
          rhs: await read(lv.rhs, fields.n * 4),
          solid: await read(fields.solid, fields.n * 4, true),
        };
      },
    };
  } else {
    start();
  }
}

init().catch((e: unknown) => {
  console.error(e);
  window.__wmVolumeError = e instanceof Error ? e.message : String(e);
  fail(e instanceof Error ? e.message : String(e));
});
