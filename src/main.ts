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
import { createEnvironmentRenderer, type EnvironmentRenderer } from './render/environment';
import { setupCanvasResize } from './render/canvasSize';
import { requestGpuDevice, type GpuContext } from './render/gpuDevice';
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

  let simRunning = false;
  let rafId = 0;
  let gpuReady = false;
  let device: GPUDevice | null = null;
  let fields: Fields | null = null;
  let drops: Droplets | null = null;
  let sim: Sim | null = null;
  let renderer: Renderer | null = null;
  let env: EnvironmentRenderer | null = null;
  let diagnostics: FieldDiagnosticsBuffers | null = null;
  let advanced: AdvancedControls | null = null;
  let cleanupCanvasResize: (() => void) | null = null;

  function pauseSim(): void {
    params.paused = true;
    $button('pause').textContent = 'Resume';
  }

  function stopFrameLoop(): void {
    simRunning = false;
    if (rafId) {
      cancelAnimationFrame(rafId);
      rafId = 0;
    }
  }

  function handleGpuFatal(msg: string): void {
    stopFrameLoop();
    fail(msg);
  }

  async function rebuildGridIfNeeded(): Promise<void> {
    if (!gpuReady || !device || !fields || !sim || !renderer || !diagnostics) return;
    const need = fields.n !== params.nx * params.ny;
    if (!need) return;
    fields.destroy();
    fields = createFields(device, params);
    sim.setFields(fields);
    renderer.rebuildBindGroups(fields);
    diagnostics.resize(device, params.nx * params.ny);
  }

  async function applySimulationChanges(result?: ApplyStateResult): Promise<void> {
    if (!gpuReady || !device || !fields || !sim || !renderer || !diagnostics) return;
    if (result?.needsGridRebuild) {
      fields.destroy();
      fields = createFields(device, params);
      sim.setFields(fields);
      renderer.rebuildBindGroups(fields);
      diagnostics.resize(device, params.nx * params.ny);
    } else {
      await rebuildGridIfNeeded();
    }
    resetSim();
    syncEmitterPanelUI(params);
    syncBurstUI(params);
  }

  function resetSim(): void {
    sim?.reset(params);
    env?.reset();
  }

  setupControls(params, {
    onReset: resetSim,
    // Rewrite the mask in place (no reset): the next substep's boundary pass
    // clears velocity inside the new solids.
    onSolidObstacles: () => {
      if (device && fields) fields.writeSolids(device.queue, params);
      share.notifyChange();
    },
  });
  const share = setupShare(params, {
    onApply: (result: ApplyStateResult) => {
      void applySimulationChanges(result);
    },
    loadedFromHash,
    hashError,
  });
  setupPresets(params, {
    onApply: () => {
      void applySimulationChanges();
      share.notifyChange();
    },
  });
  advanced = setupAdvanced(params, {
    onPause: pauseSim,
    onApply: (result: ApplyStateResult) => {
      void applySimulationChanges(result);
      share.notifyChange();
    },
  });
  setupEmitterPanel(params);
  setupBurst(params);
  setupHelp();

  // Throws with a readable message (missing WebGPU, no adapter, or the name of
  // an unmet limit); `init().catch` below surfaces it in the error panel.
  const gpu: GpuContext = await requestGpuDevice(params);
  device = gpu.device;
  // `gpu.timestampQuery` reports whether GPU timing is available; absence is fine.
  device.lost.then((info) => {
    handleGpuFatal(`GPU device lost (${info.reason}). Reload the page to restart.`);
  });
  device.addEventListener('uncapturederror', (event: GPUUncapturedErrorEvent) => {
    console.error('[wind_manager] WebGPU error:', event.error.message);
    setText('nan-status', 'GPU error (see console)');
    setTextClass('nan-status', 'bad', true);
    handleGpuFatal(`GPU error: ${event.error.message}. Reload the page to restart.`);
  });

  const canvas = $<HTMLCanvasElement>('c');
  const canvasWrap = document.querySelector('.canvas-wrap') as HTMLElement;
  const gpuContext: GPUCanvasContext = getWebGPUContext(canvas);
  const format: GPUTextureFormat = navigator.gpu.getPreferredCanvasFormat();
  configureWebGPUCanvas(gpuContext, device, format);

  fields = createFields(device, params);
  drops = createDroplets(device, params);
  const tracers: TracerSystem = await createTracerSystem(device);
  sim = await createSim(device, fields, drops, tracers, params);
  renderer = await createRenderer(device, format, fields, drops, tracers, params);
  env = await createEnvironmentRenderer(device, format);

  diagnostics = createFieldDiagnosticsBuffers(device, params.nx * params.ny);

  setupEmitterInteraction(canvas, params, () => {
    syncEmitterPanelUI(params);
    syncBurstUI(params);
    share.notifyChange();
  });

  cleanupCanvasResize = setupCanvasResize({
    container: canvasWrap,
    canvases: [canvas],
    onResize: () => {
      configureWebGPUCanvas(gpuContext, device!, format);
    },
  });

  resetSim();
  gpuReady = true;

  let diagBusy = false;

  async function runDiagnostics(): Promise<void> {
    if (!device || !diagnostics || !sim || !drops) return;
    diagBusy = true;
    try {
      const sample = await sampleFieldDiagnostics(device, diagnostics, sim.getFields(), drops);
      const nanEl = $('nan-status');
      if (sample.hasNaN) {
        nanEl.textContent = 'NaN DETECTED';
        nanEl.classList.add('bad');
        advanced?.notifyInstability('Non-finite field values detected.');
      } else {
        nanEl.textContent = 'fields finite';
        nanEl.classList.remove('bad');
        setText(
          't-range',
          `${sample.minTemperature.toFixed(2)} … ${sample.maxTemperature.toFixed(2)} °C`,
        );
        setText('max-speed', `${Math.sqrt(sample.maxSpeedSquared).toFixed(2)} m/s`);
        setTextClass('max-speed', 'bad', sample.maxSpeedSquared > 3600);
        setText('max-divergence', `${sample.maxDivergence.toExponential(2)} /s`);
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
    if (!simRunning || !sim || !renderer || !env || !fields || !device) return;
    const now = performance.now();
    const dt = Math.min(0.05, (now - lastFrame) / 1000);
    lastFrame = now;

    // One command buffer per frame: sim substeps (+ tracers), environment
    // state update, then the field + environment render pass. The environment
    // binds the post-step velocity buffer directly; nothing is mapped.
    const enc = device.createCommandEncoder({ label: 'frame' });
    if (!params.paused) {
      tickBurst(params, dt);
      sim.step(params, enc);
    }
    const bursting = params.burstRemaining > 0;
    if (bursting || wasBursting) syncBurstUI(params);
    wasBursting = bursting;

    const currentFields = sim.getFields();
    renderer.rebuildBindGroups(currentFields);
    env.update(enc, params, currentFields, sim.time, dt, canvas.width, canvas.height);
    const stylized = env;
    renderer.draw(
      enc,
      gpuContext.getCurrentTexture().createView(),
      params,
      currentFields,
      canvas.width,
      canvas.height,
      (pass) => stylized.draw(pass, params),
    );
    device.queue.submit([enc.finish()]);

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
    rafId = requestAnimationFrame(frame);
  }

  function startFrameLoop(): void {
    if (simRunning || document.hidden) return;
    simRunning = true;
    rafId = requestAnimationFrame(frame);
  }

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopFrameLoop();
    } else {
      startFrameLoop();
    }
  });

  window.addEventListener('beforeunload', () => {
    cleanupCanvasResize?.();
    fields?.destroy();
    drops?.destroy();
    tracers.destroy();
  });

  setText('status', 'running');
  startFrameLoop();
}

init().catch((error: unknown) => {
  console.error(error);
  fail(formatInitError(error));
});
