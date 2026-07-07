import type { Params } from '../sim/params';
import { syncBurstUI } from './burst';
import type { Emitter } from '../sim/emitters';
import {
  MAX_EMITTERS,
  addEmitter,
  clampEmitterPosition,
  emitterPositionBounds,
  getSelectedEmitter,
  removeSelectedEmitter,
  selectEmitter,
} from '../sim/emitters';

type SliderKey =
  | 'speed'
  | 'spreadDeg'
  | 'angleDeg'
  | 'rate'
  | 'rMinUm'
  | 'rMaxUm';

const SLIDER_KEYS: SliderKey[] = ['speed', 'spreadDeg', 'angleDeg', 'rate', 'rMinUm', 'rMaxUm'];

const SLIDER_FMT: Record<SliderKey, (v: number, em: Emitter) => string> = {
  speed: (v) => `${v.toFixed(1)} m/s`,
  spreadDeg: (v) => `±${v.toFixed(0)}°`,
  angleDeg: (v) => `${v.toFixed(0)}°`,
  rate: (v, em) => (em.type === 'water' ? `${(v / 1000).toFixed(1)}k drops/s` : `${(v / 1000).toFixed(1)}k impulses/s`),
  rMinUm: (v) => `${v.toFixed(0)} µm`,
  rMaxUm: (v) => `${(v / 1000).toFixed(2)} mm`,
};

const POS_DECIMALS = 2;
const NUDGE_FINE = 0.01;
const NUDGE_COARSE = 0.1;

function formatM(v: number): string {
  return v.toFixed(POS_DECIMALS);
}

function positionReadout(em: Emitter): string {
  return `(${formatM(em.x)}, ${formatM(em.y)}) m`;
}

function selectedPositionText(em: Emitter): string {
  return `${em.type} x=${formatM(em.x)} y=${formatM(em.y)}`;
}

function arrangementText(params: Params): string {
  const lines = params.emitters.map(
    (e) =>
      `${e.type} x=${formatM(e.x)} y=${formatM(e.y)} angle=${e.angleDeg.toFixed(0)} spread=${e.spreadDeg.toFixed(0)} speed=${e.speed.toFixed(1)}`,
  );
  return `domain ${formatM(params.domainW)}×${formatM(params.domainH)} m\n${lines.join('\n')}`;
}

async function copyText(text: string, btn: HTMLButtonElement): Promise<void> {
  try {
    await navigator.clipboard.writeText(text);
    const prev = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => {
      btn.textContent = prev;
    }, 1200);
  } catch {
    btn.textContent = 'Copy failed';
    setTimeout(() => {
      btn.textContent = btn.id === 'emit-pos-copy' ? 'Copy position' : 'Copy all';
    }, 1200);
  }
}

function syncPositionControls(params: Params, em: Emitter | null): void {
  const bounds = emitterPositionBounds(params);
  const xNum = document.getElementById('emit-x') as HTMLInputElement;
  const yNum = document.getElementById('emit-y') as HTMLInputElement;
  const xSlider = document.getElementById('emit-x-slider') as HTMLInputElement;
  const ySlider = document.getElementById('emit-y-slider') as HTMLInputElement;
  const readout = document.getElementById('emit-pos-readout')!;
  const domainLbl = document.getElementById('emit-pos-domain')!;
  const copyPos = document.getElementById('emit-pos-copy') as HTMLButtonElement;
  const copyAll = document.getElementById('emit-arr-copy') as HTMLButtonElement;

  domainLbl.textContent = `${formatM(params.domainW)} × ${formatM(params.domainH)} m`;

  for (const [slider, min, max] of [
    [xSlider, bounds.xMin, bounds.xMax],
    [ySlider, bounds.yMin, bounds.yMax],
  ] as const) {
    slider.min = String(min);
    slider.max = String(max);
    slider.step = '0.05';
  }
  xNum.min = String(bounds.xMin);
  xNum.max = String(bounds.xMax);
  xNum.step = '0.01';
  yNum.min = String(bounds.yMin);
  yNum.max = String(bounds.yMax);
  yNum.step = '0.01';

  const enabled = !!em;
  for (const el of [xNum, yNum, xSlider, ySlider, copyPos]) {
    el.disabled = !enabled;
  }
  copyAll.disabled = params.emitters.length === 0;

  if (!em) {
    readout.textContent = '—';
    xNum.value = '';
    yNum.value = '';
    return;
  }

  readout.textContent = positionReadout(em);
  xNum.value = formatM(em.x);
  yNum.value = formatM(em.y);
  xSlider.value = String(em.x);
  ySlider.value = String(em.y);
}

function applyAxisValue(
  params: Params,
  axis: 'x' | 'y',
  raw: string,
  sync: () => void,
): void {
  const em = getSelectedEmitter(params);
  if (!em) return;
  const v = parseFloat(raw);
  if (!Number.isFinite(v)) return;
  em[axis] = v;
  clampEmitterPosition(em, params);
  sync();
}

function nudgeAxis(
  params: Params,
  axis: 'x' | 'y',
  delta: number,
  sync: () => void,
): void {
  const em = getSelectedEmitter(params);
  if (!em) return;
  em[axis] += delta;
  clampEmitterPosition(em, params);
  sync();
}

export function syncEmitterPanelUI(params: Params): void {
  const list = document.getElementById('emitter-list')!;
  const info = document.getElementById('emitter-selected-info')!;
  const em = getSelectedEmitter(params);

  list.innerHTML = '';
  for (const e of params.emitters) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'emitter-item' + (e.id === params.selectedEmitterId ? ' active' : '');
    const tag = e.type === 'water' ? '💧' : '〰️';
    btn.textContent = `${tag} ${e.type === 'water' ? 'water' : 'air stream'} @ (${formatM(e.x)}, ${formatM(e.y)})`;
    btn.addEventListener('click', () => {
      selectEmitter(params, e.id);
      syncEmitterPanelUI(params);
    });
    list.appendChild(btn);
  }

  const addWater = document.getElementById('add-water') as HTMLButtonElement;
  const addAir = document.getElementById('add-air') as HTMLButtonElement;
  const removeBtn = document.getElementById('remove-emitter') as HTMLButtonElement;
  const atMax = params.emitters.length >= MAX_EMITTERS;
  addWater.disabled = atMax;
  addAir.disabled = atMax;
  removeBtn.disabled = !em;

  syncPositionControls(params, em);

  syncBurstUI(params);

  if (!em) {
    const touch = window.matchMedia('(pointer: coarse)').matches;
    info.textContent = params.emitters.length
      ? 'Click an emitter in the list or on the canvas'
      : touch
        ? 'Tap canvas for water • double-tap empty canvas for air stream'
        : 'Click canvas to place water • Shift+click for air stream';
    for (const key of SLIDER_KEYS) {
      const id =
        key === 'spreadDeg' ? 'emit-spread' :
        key === 'angleDeg' ? 'emit-angle' :
        key === 'rMinUm' ? 'r-min' :
        key === 'rMaxUm' ? 'r-max' :
        `emit-${key}`;
      (document.getElementById(id) as HTMLInputElement).disabled = true;
    }
    return;
  }

  info.innerHTML = em.type === 'water'
    ? '<strong>Water spray</strong> — droplets + evaporative cooling'
    : '<strong>Air stream</strong> — momentum only (no mass / latent heat)';

  for (const key of SLIDER_KEYS) {
    const id =
      key === 'spreadDeg' ? 'emit-spread' :
      key === 'angleDeg' ? 'emit-angle' :
      key === 'rMinUm' ? 'r-min' :
      key === 'rMaxUm' ? 'r-max' :
      `emit-${key}`;
    const el = document.getElementById(id) as HTMLInputElement;
    const label = document.getElementById(id + '-val')!;
    const val = em[key];
    el.disabled = em.type === 'air' && (key === 'rMinUm' || key === 'rMaxUm');
    el.value = String(val);
    label.textContent = SLIDER_FMT[key](val as number, em);
  }
}

export function setupEmitterPanel(params: Params): void {
  const $ = (id: string) => document.getElementById(id) as HTMLButtonElement;
  const sync = () => syncEmitterPanelUI(params);

  $('add-water').addEventListener('click', () => {
    if (addEmitter(params, 'water', params.domainW * 0.35, 0.35)) {
      sync();
    }
  });

  $('add-air').addEventListener('click', () => {
    if (addEmitter(params, 'air', params.domainW * 0.55, 0.45)) {
      sync();
    }
  });

  $('remove-emitter').addEventListener('click', () => {
    if (removeSelectedEmitter(params)) sync();
  });

  const sliderMap: [string, SliderKey][] = [
    ['emit-speed', 'speed'],
    ['emit-spread', 'spreadDeg'],
    ['emit-angle', 'angleDeg'],
    ['emit-rate', 'rate'],
    ['r-min', 'rMinUm'],
    ['r-max', 'rMaxUm'],
  ];

  for (const [id, key] of sliderMap) {
    const el = document.getElementById(id) as HTMLInputElement;
    el.addEventListener('input', () => {
      const em = getSelectedEmitter(params);
      if (!em) return;
      em[key] = parseFloat(el.value);
      if (em.rMinUm > em.rMaxUm) {
        if (key === 'rMinUm') em.rMaxUm = em.rMinUm;
        else em.rMinUm = em.rMaxUm;
      }
      sync();
    });
  }

  const xNum = document.getElementById('emit-x') as HTMLInputElement;
  const yNum = document.getElementById('emit-y') as HTMLInputElement;
  const xSlider = document.getElementById('emit-x-slider') as HTMLInputElement;
  const ySlider = document.getElementById('emit-y-slider') as HTMLInputElement;

  const bindAxis = (
    axis: 'x' | 'y',
    num: HTMLInputElement,
    slider: HTMLInputElement,
  ): void => {
    num.addEventListener('change', () => applyAxisValue(params, axis, num.value, sync));
    num.addEventListener('input', () => {
      if (document.activeElement !== num) return;
      applyAxisValue(params, axis, num.value, sync);
    });
    slider.addEventListener('input', () => {
      const em = getSelectedEmitter(params);
      if (!em) return;
      em[axis] = parseFloat(slider.value);
      clampEmitterPosition(em, params);
      sync();
    });
    num.addEventListener('keydown', (e) => {
      if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) return;
      const em = getSelectedEmitter(params);
      if (!em) return;
      const step = e.shiftKey ? NUDGE_COARSE : NUDGE_FINE;
      let delta: number;
      if (axis === 'x') {
        if (e.key === 'ArrowLeft') delta = -step;
        else if (e.key === 'ArrowRight') delta = step;
        else return;
      } else {
        if (e.key === 'ArrowUp') delta = step;
        else if (e.key === 'ArrowDown') delta = -step;
        else return;
      }
      e.preventDefault();
      nudgeAxis(params, axis, delta, sync);
    });
  };

  bindAxis('x', xNum, xSlider);
  bindAxis('y', yNum, ySlider);

  document.getElementById('emit-pos-copy')!.addEventListener('click', () => {
    const em = getSelectedEmitter(params);
    if (!em) return;
    copyText(selectedPositionText(em), document.getElementById('emit-pos-copy') as HTMLButtonElement);
  });

  document.getElementById('emit-arr-copy')!.addEventListener('click', () => {
    if (!params.emitters.length) return;
    copyText(arrangementText(params), document.getElementById('emit-arr-copy') as HTMLButtonElement);
  });

  sync();
}
