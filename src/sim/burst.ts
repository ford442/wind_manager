import type { Emitter } from './emitters';
import type { Params } from './params';

/** True when burst applies to this emitter (selected only, or all if none selected). */
export function burstTargetsEmitter(p: Params, em: Emitter): boolean {
  if (p.burstRemaining <= 0) return false;
  if (!p.selectedEmitterId) return true;
  return em.id === p.selectedEmitterId;
}

export function effectiveEmitRate(p: Params, em: Emitter): number {
  if (burstTargetsEmitter(p, em)) {
    return em.rate * p.burstMultiplier;
  }
  return em.rate;
}

export function triggerBurst(p: Params): boolean {
  if (p.emitters.length === 0) return false;
  p.burstRemaining = p.burstDuration;
  return true;
}

export function tickBurst(p: Params, dt: number): void {
  if (p.burstRemaining <= 0) return;
  p.burstRemaining = Math.max(0, p.burstRemaining - dt);
}

export function burstTargetLabel(p: Params): string {
  if (!p.emitters.length) return 'no emitters';
  if (p.selectedEmitterId) {
    const em = p.emitters.find((e) => e.id === p.selectedEmitterId);
    if (em) {
      return em.type === 'water' ? 'selected water spray' : 'selected air stream';
    }
  }
  return `all ${p.emitters.length} emitter${p.emitters.length === 1 ? '' : 's'}`;
}
