import type { Params } from './sim/params';
import { defaultParams } from './sim/params';
import type { Fields } from './sim/fields';
import { createFields } from './sim/fields';
import type { Droplets } from './sim/droplets';
import { createDroplets } from './sim/droplets';
import type { TracerSystem } from './sim/tracers';
import { createTracerSystem } from './sim/tracers';
import type { Sim } from './sim/step';
import { createSim } from './sim/step';
import type { Renderer } from './render/overlays';
import { createRenderer } from './render/overlays';
import {
  createFieldDiagnosticsBuffers,
  sampleFieldDiagnostics,
  type FieldDiagnosticsBuffers,
} from './render/fieldDiagnostics';
import { configureWebGPUCanvas, getWebGPUContext } from './render/webgpu';
import type { VelSampler } from './render/velSampler';
import { createVelSampler } from './render/velSampler';
import { setupControls } from './ui/controls';
import { setupPresets } from './ui/presets';
import { setupAdvanced, type AdvancedControls } from './ui/advanced';
import { setupEmitterInteraction } from './ui/emitter';
import { setupEmitterPanel, syncEmitterPanelUI } from './ui/emitterPanel';
import { setupBurst, syncBurstUI } from './ui/burst';
import { setupShare } from './ui/share';
import { setupHelp } from './ui/help';
import { applySavedState, loadStateFromHash, type ApplyStateResult } from './sim/state';
import { tickBurst } from './sim/burst';
import { createGrassLayer } from './render/grass';
import { createTreesLayer } from './render/trees';
import { createBackyardLayer } from './render/backyard';
import { drawEnvironmentOverlay, needsVelSampler } from './render/environment';
import { setupCanvasResize } from './render/canvasSize';
import { $, $button, setText, setTextClass } from './ui/dom';

const DIAG_INTERVAL = 30;

function fail(msg: string): void {
  const el = document.getElementById('gpu-error');
  if (el) {
    el.style.display = 'block';
    const msgEl = el.querySelector('.msg');
    if (msgEl) msgEl.textContent = msg;
  }
  setText('status', 'unavailable');
}

function formatInitError(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

async function init(): Promise<void> {
  if (!navigator.gpu) {
    fail('This browser does not expose WebGPU (navigator.gpu is missing). ' +
         'Use a recent Chrome/Edge, or enable WebGPU in your browser.');
    return;
  }
  const adapter: GPUAdapter | null = await navigator.gpu.requestAdapter();
  if (!adapter) {
    fail('No WebGPU adapter available on this machine.');
    return;
  }
  const device: GPUDevice = await adapter.requestDevice();
  device.addEventListener('uncapturederror', (event: GPUUncapturedErrorEvent) => {
    console.error('[wind_manager] WebGPU error:', event.error.message);
    setText('nan-status', 'GPU error (see console)');
    setTextClass('nan-status', 'bad', true);
  });

  const canvas = $<HTMLCanvasElement>('c');
  const grassCanvas = $<HTMLCanvasElement>('grass-overlay');
  const canvasWrap = document.querySelector('.canvas-wrap') as HTMLElement;
  const gpuContext: GPUCanvasContext = getWebGPUContext(canvas);
  const format: GPUTextureFormat = navigator.gpu.getPreferredCanvasFormat();
  configureWebGPUCanvas(gpuContext, device, format);

  const params: Params = defaultParams();
  const hashState = loadStateFromHash();
  let loadedFromHash = false;
  let hashError: string | null = null;
  if (hashState && 'state' in hashState) {
    applySavedState(params, hashState.state);
    loadedFromHash = true;
  } else if (hashState && 'error' in hashState) {
    hashError = hashState.error;
  }

  let fields: Fields = createFields(device, params);
  const drops: Droplets = createDroplets(device, params);
  const tracers: TracerSystem = createTracerSystem(device);
  const sim: Sim = await createSim(device, fields, drops, tracers, params);
  const renderer: Renderer = await createRenderer(device, format, fields, drops, tracers, params);
  let velSampler: VelSampler = createVelSampler(device, params);
  const grass = createGrassLayer(grassCanvas, params);
  const trees = createTreesLayer(grassCanvas);
  const backyard = createBackyardLayer(grassCanvas);

  const diagnostics: FieldDiagnosticsBuffers = createFieldDiagnosticsBuffers(
    device,
    params.nx * params.ny,
  );

  function resizeDiagnostics(): void {
    diagnostics.resize(device, params.nx * params.ny);
  }

  function pauseSim(): void {
    params.paused = true;
    $button('pause').textContent = 'Resume';
  }

  function rebuildGridIfNeeded(): void {
    const need = fields.n !== params.nx * params.ny;
    if (!need) return;
    fields = createFields(device, params);
    sim.setFields(fields);
    velSampler.resize(device, params);
    renderer.rebuildBindGroups(fields);
    resizeDiagnostics();
  }

  function applySimulationChanges(result?: ApplyStateResult): void {
    if (result?.needsGridRebuild) {
      fields = createFields(device, params);
      sim.setFields(fields);
      velSampler.resize(device, params);
      renderer.rebuildBindGroups(fields);
      resizeDiagnostics();
    } else {
      rebuildGridIfNeeded();
    }
    grass.rebuild(params);
    sim.reset(params);
    syncEmitterPanelUI(params);
    syncBurstUI(params);
  }

  sim.reset(params);
  setupControls(params, {
    onReset: () => sim.reset(params),
    onGrassDensity: () => grass.rebuild(params),
  });
  const share = setupShare(params, {
    onApply: (result: ApplyStateResult) => applySimulationChanges(result),
    loadedFromHash,
    hashError,
  });
  setupPresets(params, {
    onApply: () => {
      applySimulationChanges();
      share.notifyChange();
    },
  });
  const advanced: AdvancedControls = setupAdvanced(params, {
    onPause: pauseSim,
    onApply: (result: ApplyStateResult) => {
      applySimulationChanges(result);
      share.notifyChange();
    },
  });
  setupEmitterPanel(params);
  setupBurst(params);
  setupEmitterInteraction(canvas, params, () => {
    syncEmitterPanelUI(params);
    syncBurstUI(params);
    share.notifyChange();
  });
  setupHelp();

  setupCanvasResize({
    container: canvasWrap,
    canvases: [canvas, grassCanvas],
    onResize: () => {
      configureWebGPUCanvas(gpuContext, device, format);
      grass.resize(grassCanvas.width, grassCanvas.height);
    },
  });

  let diagBusy = false;

  async function runDiagnostics(): Promise<void> {
    diagBusy = true;
    try {
      const sample = await sampleFieldDiagnostics(device, diagnostics, sim.getFields(), drops);
      const nanEl = $('nan-status');
      if (sample.hasNaN) {
        nanEl.textContent = 'NaN DETECTED';
        nanEl.classList.add('bad');
        advanced.notifyInstability('Non-finite field values detected.');
      } else {
        nanEl.textContent = 'fields finite';
        nanEl.classList.remove('bad');
        setText(
          't-range',
          `${sample.minTemperature.toFixed(2)} … ${sample.maxTemperature.toFixed(2)} °C`,
        );
        setText('max-speed', `${Math.sqrt(sample.maxSpeedSquared).toFixed(2)} m/s`);
        setTextClass('max-speed', 'bad', sample.maxSpeedSquared > 3600);
      }
      setText('emitted', sample.emitted.toLocaleString());
    } finally {
      diagBusy = false;
    }
  }

  let lastFps = performance.now();
  let framesSinceFps = 0;
  let lastFrame = performance.now();
  let wasBursting = false;

  function frame(): void {
    const now = performance.now();
    const dt = Math.min(0.05, (now - lastFrame) / 1000);
    lastFrame = now;

    if (!params.paused) {
      tickBurst(params, dt);
      sim.step(params);
    }
    const bursting = params.burstRemaining > 0;
    if (bursting || wasBursting) syncBurstUI(params);
    wasBursting = bursting;

    const currentFields = sim.getFields();
    velSampler.scheduleRead(device, currentFields, params, needsVelSampler(params));
    renderer.draw(
      gpuContext.getCurrentTexture().createView(),
      params,
      currentFields,
      canvas.width,
      canvas.height,
    );
    drawEnvironmentOverlay(
      grassCanvas,
      params,
      sim.time,
      dt,
      velSampler,
      grass,
      trees,
      backyard,
    );

    framesSinceFps++;
    if (now - lastFps > 500) {
      setText('fps', ((framesSinceFps * 1000) / (now - lastFps)).toFixed(0));
      framesSinceFps = 0;
      lastFps = now;
    }
    setText('sim-time', `${sim.time.toFixed(1)} s`);

    if (sim.frame % DIAG_INTERVAL === 0 && !diagBusy && !params.paused) {
      runDiagnostics().catch((error: unknown) => {
        console.error('diagnostics failed', error);
      });
    }
    requestAnimationFrame(frame);
  }

  setText('status', 'running');
  requestAnimationFrame(frame);
}

init().catch((error: unknown) => {
  console.error(error);
  fail(formatInitError(error));
});
