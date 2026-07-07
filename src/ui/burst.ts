import { burstTargetLabel, triggerBurst } from '../sim/burst';
import type { Params } from '../sim/params';

function isTypingTarget(el: EventTarget | null): boolean {
  if (!el || !(el instanceof HTMLElement)) return false;
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
}

export function syncBurstUI(params: Params): void {
  const btn = document.getElementById('emit-burst') as HTMLButtonElement | null;
  const status = document.getElementById('burst-status');
  if (!btn || !status) return;

  const active = params.burstRemaining > 0;
  btn.classList.toggle('active', active);
  btn.disabled = params.emitters.length === 0;

  if (active) {
    status.textContent =
      `Bursting ${burstTargetLabel(params)} · ${params.burstRemaining.toFixed(2)} s · ×${params.burstMultiplier}`;
  } else if (params.emitters.length === 0) {
    status.textContent = 'Add an emitter to burst';
  } else {
    status.textContent =
      `Space or Burst — ${burstTargetLabel(params)} for ${params.burstDuration.toFixed(2)} s at ×${params.burstMultiplier}`;
  }
}

export function setupBurst(params: Params): void {
  const btn = document.getElementById('emit-burst') as HTMLButtonElement;
  const duration = document.getElementById('burst-duration') as HTMLInputElement;
  const durationVal = document.getElementById('burst-duration-val')!;
  const multiplier = document.getElementById('burst-multiplier') as HTMLInputElement;
  const multiplierVal = document.getElementById('burst-multiplier-val')!;

  const fire = () => {
    if (triggerBurst(params)) syncBurstUI(params);
  };

  btn.addEventListener('click', fire);

  duration.value = String(params.burstDuration);
  durationVal.textContent = `${params.burstDuration.toFixed(2)} s`;
  duration.addEventListener('input', () => {
    params.burstDuration = parseFloat(duration.value);
    durationVal.textContent = `${params.burstDuration.toFixed(2)} s`;
    syncBurstUI(params);
  });

  multiplier.value = String(params.burstMultiplier);
  multiplierVal.textContent = `×${params.burstMultiplier}`;
  multiplier.addEventListener('input', () => {
    params.burstMultiplier = parseFloat(multiplier.value);
    multiplierVal.textContent = `×${params.burstMultiplier}`;
    syncBurstUI(params);
  });

  window.addEventListener('keydown', (e) => {
    if (e.code !== 'Space' && e.key !== ' ') return;
    if (e.repeat || isTypingTarget(e.target)) return;
    e.preventDefault();
    fire();
  });

  syncBurstUI(params);
}
