import type { Params } from '../sim/params';
import type { Emitter } from '../sim/emitters';
import {
  MAX_EMITTERS,
  addEmitter,
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
    btn.textContent = `${tag} ${e.type === 'water' ? 'water' : 'air stream'} @ (${e.x.toFixed(1)}, ${e.y.toFixed(1)})`;
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

  if (!em) {
    info.textContent = params.emitters.length
      ? 'Click an emitter in the list or on the canvas'
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

  $('add-water').addEventListener('click', () => {
    if (addEmitter(params, 'water', params.domainW * 0.35, 0.35)) {
      syncEmitterPanelUI(params);
    }
  });

  $('add-air').addEventListener('click', () => {
    if (addEmitter(params, 'air', params.domainW * 0.55, 0.45)) {
      syncEmitterPanelUI(params);
    }
  });

  $('remove-emitter').addEventListener('click', () => {
    if (removeSelectedEmitter(params)) syncEmitterPanelUI(params);
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
      (em as any)[key] = parseFloat(el.value);
      if (em.rMinUm > em.rMaxUm) {
        if (key === 'rMinUm') em.rMaxUm = em.rMinUm;
        else em.rMinUm = em.rMaxUm;
      }
      syncEmitterPanelUI(params);
    });
  }

  syncEmitterPanelUI(params);
}
