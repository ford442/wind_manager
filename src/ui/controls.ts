import type { Params } from '../sim/params';

export function setupControls(params: Params, { onReset }: { onReset: () => void }) {
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

  const sliders: [string, keyof Params, (v: number) => string][] = [
    ['emit-speed', 'emitSpeed', (v) => `${v.toFixed(1)} m/s`],
    ['emit-spread', 'emitSpreadDeg', (v) => `±${v.toFixed(0)}°`],
    ['emit-angle', 'emitAngleDeg', (v) => `${v.toFixed(0)}°`],
    ['emit-rate', 'emitRate', (v) => `${(v / 1000).toFixed(1)}k/s`],
    ['r-min', 'rMinUm', (v) => `${v.toFixed(0)} µm`],
    ['r-max', 'rMaxUm', (v) => `${(v / 1000).toFixed(2)} mm`],
    ['t-amb', 'tAmb', (v) => `${v.toFixed(0)} °C`],
    ['rh-amb', 'rhAmb', (v) => `${v.toFixed(0)} %`],
  ];
  for (const [id, key, fmt] of sliders) {
    const el = $(id) as HTMLInputElement;
    const label = $(id + '-val') as HTMLElement;
    el.value = String(params[key]);
    label.textContent = fmt(params[key] as number);
    el.addEventListener('input', () => {
      (params as any)[key] = parseFloat(el.value);
      if (params.rMinUm > params.rMaxUm) {
        if (key === 'rMinUm') params.rMaxUm = params.rMinUm;
        else params.rMinUm = params.rMaxUm;
      }
      label.textContent = fmt(params[key] as number);
    });
  }

  const pauseBtn = $('pause');
  pauseBtn.addEventListener('click', () => {
    params.paused = !params.paused;
    pauseBtn.textContent = params.paused ? 'Resume' : 'Pause';
  });

  $('reset').addEventListener('click', onReset);
}
