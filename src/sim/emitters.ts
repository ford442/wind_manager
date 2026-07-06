import type { Params } from './params';

export type EmitterType = 'water' | 'air';

export const MAX_EMITTERS = 8;

export interface Emitter {
  id: string;
  type: EmitterType;
  x: number;
  y: number;
  angleDeg: number;
  spreadDeg: number;
  speed: number;
  rate: number;
  rMinUm: number;
  rMaxUm: number;
}

let nextId = 1;

export function createEmitter(
  type: EmitterType,
  x: number,
  y: number,
  partial?: Partial<Emitter>,
): Emitter {
  return {
    id: `em-${nextId++}`,
    type,
    x,
    y,
    angleDeg: 0,
    spreadDeg: type === 'water' ? 15 : 22,
    speed: type === 'water' ? 10 : 9,
    rate: type === 'water' ? 6000 : 5000,
    rMinUm: 50,
    rMaxUm: 800,
    ...partial,
  };
}

export function defaultEmitters(domainW = 32): Emitter[] {
  return [createEmitter('water', domainW * 0.5, 0.3)];
}

export function getSelectedEmitter(p: Params): Emitter | null {
  if (!p.selectedEmitterId) return null;
  return p.emitters.find((e) => e.id === p.selectedEmitterId) ?? null;
}

export function selectEmitter(p: Params, id: string | null): void {
  p.selectedEmitterId = id;
}

export function addEmitter(
  p: Params,
  type: EmitterType,
  x: number,
  y: number,
): Emitter | null {
  if (p.emitters.length >= MAX_EMITTERS) return null;
  const em = createEmitter(type, x, y);
  clampEmitterPosition(em, p);
  p.emitters.push(em);
  p.selectedEmitterId = em.id;
  return em;
}

export function removeSelectedEmitter(p: Params): boolean {
  const id = p.selectedEmitterId;
  if (!id) return false;
  const i = p.emitters.findIndex((e) => e.id === id);
  if (i < 0) return false;
  p.emitters.splice(i, 1);
  p.selectedEmitterId = p.emitters[i]?.id ?? p.emitters[i - 1]?.id ?? null;
  return true;
}

export function findEmitterAt(p: Params, wx: number, wy: number, hit = 0.2): Emitter | null {
  let best: Emitter | null = null;
  let bestD = hit;
  for (const em of p.emitters) {
    const d = Math.hypot(wx - em.x, wy - em.y);
    if (d < bestD) {
      bestD = d;
      best = em;
    }
  }
  return best;
}

export function clampEmitterPosition(em: Emitter, p: Params): void {
  const m = 0.12;
  em.x = Math.max(m, Math.min(p.domainW - m, em.x));
  em.y = Math.max(m, Math.min(p.domainH - m, em.y));
}

export function emitDirection(em: Emitter): [number, number] {
  const a = (em.angleDeg * Math.PI) / 180;
  return [Math.sin(a), Math.cos(a)];
}

export function waterEmitters(p: Params): Emitter[] {
  return p.emitters.filter((e) => e.type === 'water');
}

export function airEmitters(p: Params): Emitter[] {
  return p.emitters.filter((e) => e.type === 'air');
}
