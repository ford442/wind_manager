import type { Params } from '../sim/params';
import { syncAdvancedUI } from './advanced';

export function setupControls(
  params: Params,
  { onReset, onGrassDensity }: { onReset: () => void; onGrassDensity?: () => void },
) {
  const $ = (id: string): any => document.getElementById(id);

  const latent = $('latent-toggle');
  latent.checked = params.latentOn;
  latent.addEventListener('change', () => {
    params.latentOn = latent.checked;
    $('latent-state').textContent = params.latentOn
      ? 'ON — evaporative cooling → downdraft'
      : 'OFF — momentum only → updraft';
  });

  const modes: [string, number][] = [
    ['ov-none', 0],
    ['ov-vel', 1],
    ['ov-temp', 2],
    ['ov-hum', 3],
  ];
  function setOverlay(mode: number) {
    params.overlay = mode;
    for (const [id, m] of modes) {
      $(id as string).classList.toggle('active', m === mode);
    }
  }
  for (const [id, m] of modes) {
    $(id).addEventListener('click', () => setOverlay(m));
  }
  setOverlay(params.overlay);

  const arrows = $('show-arrows');
  arrows.checked = params.showArrows;
  arrows.addEventListener('change', () => (params.showArrows = arrows.checked));

  const dropsChk = $('show-droplets');
  dropsChk.checked = params.showDroplets;
  dropsChk.addEventListener('change', () => (params.showDroplets = dropsChk.checked));

  const tracersChk = $('show-tracers');
  tracersChk.checked = params.showTracers;
  tracersChk.addEventListener('change', () => (params.showTracers = tracersChk.checked));

  const streaksChk = $('show-tracer-streaks');
  streaksChk.checked = params.showTracerStreaks;
  streaksChk.addEventListener('change', () => (params.showTracerStreaks = streaksChk.checked));

  const tracerSliders: [string, keyof Params, (v: number) => string][] = [
    ['tracer-density', 'tracerDensity', (v) => `${Math.round(v).toLocaleString()}/s`],
    ['tracer-lifetime', 'tracerLifetime', (v) => `${v.toFixed(1)} s`],
  ];
  for (const [id, key, fmt] of tracerSliders) {
    const el = $(id) as HTMLInputElement;
    const label = $(id + '-val') as HTMLElement;
    el.value = String(params[key]);
    label.textContent = fmt(params[key] as number);
    el.addEventListener('input', () => {
      (params as any)[key] = parseFloat(el.value);
      label.textContent = fmt(params[key] as number);
    });
  }

  const grassChk = $('show-grass');
  grassChk.checked = params.showGrass;
  grassChk.addEventListener('change', () => (params.showGrass = grassChk.checked));

  const treesChk = $('show-trees');
  treesChk.checked = params.showTrees;
  treesChk.addEventListener('change', () => (params.showTrees = treesChk.checked));

  const housesChk = $('show-houses');
  housesChk.checked = params.showHouses;
  housesChk.addEventListener('change', () => (params.showHouses = housesChk.checked));

  const cloudsChk = $('show-clouds');
  cloudsChk.checked = params.showClouds;
  cloudsChk.addEventListener('change', () => (params.showClouds = cloudsChk.checked));

  const mistChk = $('show-ground-mist');
  mistChk.checked = params.showGroundMist;
  mistChk.addEventListener('change', () => (params.showGroundMist = mistChk.checked));

  const wetChk = $('show-wet-ground');
  wetChk.checked = params.showWetGround;
  wetChk.addEventListener('change', () => (params.showWetGround = wetChk.checked));

  const sliders: [string, keyof Params, (v: number) => string][] = [
    ['t-amb', 'tAmb', (v) => `${v.toFixed(0)} °C`],
    ['rh-amb', 'rhAmb', (v) => `${v.toFixed(0)} %`],
    ['grass-density', 'grassDensity', (v) => `${Math.round(v)} blades`],
    ['grass-stiffness', 'grassStiffness', (v) => `${(v * 100).toFixed(0)}%`],
  ];
  for (const [id, key, fmt] of sliders) {
    const el = $(id) as HTMLInputElement;
    const label = $(id + '-val') as HTMLElement;
    el.value = String(params[key]);
    label.textContent = fmt(params[key] as number);
    el.addEventListener('input', () => {
      (params as any)[key] = parseFloat(el.value);
      label.textContent = fmt(params[key] as number);
      if (key === 'grassDensity') onGrassDensity?.();
    });
  }

  const pauseBtn = $('pause');
  pauseBtn.addEventListener('click', () => {
    params.paused = !params.paused;
    pauseBtn.textContent = params.paused ? 'Resume' : 'Pause';
  });

  $('reset').addEventListener('click', onReset);
}

export function syncControlsUI(params: Params): void {
  const $ = (id: string): any => document.getElementById(id);

  const latent = $('latent-toggle') as HTMLInputElement;
  latent.checked = params.latentOn;
  $('latent-state').textContent = params.latentOn
    ? 'ON — evaporative cooling → downdraft'
    : 'OFF — momentum only → updraft';

  const modes: [string, number][] = [
    ['ov-none', 0],
    ['ov-vel', 1],
    ['ov-temp', 2],
    ['ov-hum', 3],
  ];
  for (const [id, m] of modes) {
    $(id).classList.toggle('active', m === params.overlay);
  }

  ($('show-arrows') as HTMLInputElement).checked = params.showArrows;
  ($('show-droplets') as HTMLInputElement).checked = params.showDroplets;
  ($('show-tracers') as HTMLInputElement).checked = params.showTracers;
  ($('show-tracer-streaks') as HTMLInputElement).checked = params.showTracerStreaks;

  const tracerSliders: [string, keyof Params, (v: number) => string][] = [
    ['tracer-density', 'tracerDensity', (v) => `${Math.round(v).toLocaleString()}/s`],
    ['tracer-lifetime', 'tracerLifetime', (v) => `${v.toFixed(1)} s`],
  ];
  for (const [id, key, fmt] of tracerSliders) {
    const el = $(id) as HTMLInputElement;
    const label = $(id + '-val') as HTMLElement;
    el.value = String(params[key]);
    label.textContent = fmt(params[key] as number);
  }

  ($('show-grass') as HTMLInputElement).checked = params.showGrass;
  ($('show-trees') as HTMLInputElement).checked = params.showTrees;
  ($('show-houses') as HTMLInputElement).checked = params.showHouses;
  ($('show-clouds') as HTMLInputElement).checked = params.showClouds;
  ($('show-ground-mist') as HTMLInputElement).checked = params.showGroundMist;
  ($('show-wet-ground') as HTMLInputElement).checked = params.showWetGround;

  const sliders: [string, keyof Params, (v: number) => string][] = [
    ['t-amb', 'tAmb', (v) => `${v.toFixed(0)} °C`],
    ['rh-amb', 'rhAmb', (v) => `${v.toFixed(0)} %`],
    ['grass-density', 'grassDensity', (v) => `${Math.round(v)} blades`],
    ['grass-stiffness', 'grassStiffness', (v) => `${(v * 100).toFixed(0)}%`],
  ];
  for (const [id, key, fmt] of sliders) {
    const el = $(id) as HTMLInputElement;
    const label = $(id + '-val') as HTMLElement;
    el.value = String(params[key]);
    label.textContent = fmt(params[key] as number);
  }

  const pauseBtn = $('pause') as HTMLButtonElement;
  pauseBtn.textContent = params.paused ? 'Resume' : 'Pause';

  syncAdvancedUI(params);
}
