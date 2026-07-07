import type { Params } from '../sim/params';
import { syncAdvancedUI } from './advanced';
import { $, $button, $input } from './dom';

type SliderParamKey =
  | 'tracerDensity'
  | 'tracerLifetime'
  | 'tAmb'
  | 'rhAmb'
  | 'grassDensity'
  | 'grassStiffness';

function bindCheckbox(
  id: string,
  get: () => boolean,
  set: (checked: boolean) => void,
): HTMLInputElement {
  const el = $input(id);
  el.checked = get();
  el.addEventListener('change', () => set(el.checked));
  return el;
}

function bindSlider(
  params: Params,
  id: string,
  key: SliderParamKey,
  fmt: (v: number) => string,
  onInput?: () => void,
): void {
  const el = $input(id);
  const label = $<HTMLElement>(`${id}-val`);
  el.value = String(params[key]);
  label.textContent = fmt(params[key]);
  el.addEventListener('input', () => {
    params[key] = parseFloat(el.value);
    label.textContent = fmt(params[key]);
    onInput?.();
  });
}

function syncSlider(params: Params, id: string, key: SliderParamKey, fmt: (v: number) => string): void {
  const el = $input(id);
  const label = $<HTMLElement>(`${id}-val`);
  el.value = String(params[key]);
  label.textContent = fmt(params[key]);
}

export function setupControls(
  params: Params,
  { onReset, onGrassDensity }: { onReset: () => void; onGrassDensity?: () => void },
) {
  bindCheckbox(
    'latent-toggle',
    () => params.latentOn,
    (checked) => {
      params.latentOn = checked;
      $<HTMLElement>('latent-state').textContent = params.latentOn
        ? 'ON — evaporative cooling → downdraft'
        : 'OFF — momentum only → updraft';
    },
  );

  const modes: [string, number][] = [
    ['ov-none', 0],
    ['ov-vel', 1],
    ['ov-temp', 2],
    ['ov-hum', 3],
    ['ov-moisture', 4],
  ];
  function setOverlay(mode: number) {
    params.overlay = mode;
    for (const [id, m] of modes) {
      $button(id).classList.toggle('active', m === mode);
    }
  }
  for (const [id, m] of modes) {
    $button(id).addEventListener('click', () => setOverlay(m));
  }
  setOverlay(params.overlay);

  bindCheckbox('show-arrows', () => params.showArrows, (v) => (params.showArrows = v));
  bindCheckbox('show-droplets', () => params.showDroplets, (v) => (params.showDroplets = v));
  bindCheckbox('show-tracers', () => params.showTracers, (v) => (params.showTracers = v));
  bindCheckbox('show-tracer-streaks', () => params.showTracerStreaks, (v) => (params.showTracerStreaks = v));

  bindSlider(params, 'tracer-density', 'tracerDensity', (v) => `${Math.round(v).toLocaleString()}/s`);
  bindSlider(params, 'tracer-lifetime', 'tracerLifetime', (v) => `${v.toFixed(1)} s`);

  bindCheckbox('show-grass', () => params.showGrass, (v) => (params.showGrass = v));
  bindCheckbox('stylized-view', () => params.stylizedView, (v) => (params.stylizedView = v));
  bindCheckbox('show-trees', () => params.showTrees, (v) => (params.showTrees = v));
  bindCheckbox('show-houses', () => params.showHouses, (v) => (params.showHouses = v));
  bindCheckbox('show-clouds', () => params.showClouds, (v) => (params.showClouds = v));
  bindCheckbox('show-ground-mist', () => params.showGroundMist, (v) => (params.showGroundMist = v));
  bindCheckbox('show-wet-ground', () => params.showWetGround, (v) => (params.showWetGround = v));
  bindCheckbox(
    'show-ground-moisture',
    () => params.showGroundMoisture,
    (v) => (params.showGroundMoisture = v),
  );

  bindSlider(params, 't-amb', 'tAmb', (v) => `${v.toFixed(0)} °C`);
  bindSlider(params, 'rh-amb', 'rhAmb', (v) => `${v.toFixed(0)} %`);
  bindSlider(params, 'grass-density', 'grassDensity', (v) => `${Math.round(v)} blades`, onGrassDensity);
  bindSlider(params, 'grass-stiffness', 'grassStiffness', (v) => `${(v * 100).toFixed(0)}%`);

  const pauseBtn = $button('pause');
  pauseBtn.addEventListener('click', () => {
    params.paused = !params.paused;
    pauseBtn.textContent = params.paused ? 'Resume' : 'Pause';
  });

  $button('reset').addEventListener('click', onReset);
}

export function syncControlsUI(params: Params): void {
  const latent = $input('latent-toggle');
  latent.checked = params.latentOn;
  $<HTMLElement>('latent-state').textContent = params.latentOn
    ? 'ON — evaporative cooling → downdraft'
    : 'OFF — momentum only → updraft';

  const modes: [string, number][] = [
    ['ov-none', 0],
    ['ov-vel', 1],
    ['ov-temp', 2],
    ['ov-hum', 3],
    ['ov-moisture', 4],
  ];
  for (const [id, m] of modes) {
    $button(id).classList.toggle('active', m === params.overlay);
  }

  $input('show-arrows').checked = params.showArrows;
  $input('show-droplets').checked = params.showDroplets;
  $input('show-tracers').checked = params.showTracers;
  $input('show-tracer-streaks').checked = params.showTracerStreaks;

  syncSlider(params, 'tracer-density', 'tracerDensity', (v) => `${Math.round(v).toLocaleString()}/s`);
  syncSlider(params, 'tracer-lifetime', 'tracerLifetime', (v) => `${v.toFixed(1)} s`);

  $input('show-grass').checked = params.showGrass;
  $input('stylized-view').checked = params.stylizedView;
  $input('show-trees').checked = params.showTrees;
  $input('show-houses').checked = params.showHouses;
  $input('show-clouds').checked = params.showClouds;
  $input('show-ground-mist').checked = params.showGroundMist;
  $input('show-wet-ground').checked = params.showWetGround;
  $input('show-ground-moisture').checked = params.showGroundMoisture;

  syncSlider(params, 't-amb', 'tAmb', (v) => `${v.toFixed(0)} °C`);
  syncSlider(params, 'rh-amb', 'rhAmb', (v) => `${v.toFixed(0)} %`);
  syncSlider(params, 'grass-density', 'grassDensity', (v) => `${Math.round(v)} blades`);
  syncSlider(params, 'grass-stiffness', 'grassStiffness', (v) => `${(v * 100).toFixed(0)}%`);

  $button('pause').textContent = params.paused ? 'Resume' : 'Pause';

  syncAdvancedUI(params);
}
