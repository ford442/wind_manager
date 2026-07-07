import type { Params } from '../sim/params';
import type { Emitter, EmitterType } from '../sim/emitters';
import {
  MAX_EMITTERS,
  addEmitter,
  clampEmitterPosition,
  emitDirection,
  findEmitterAt,
  getSelectedEmitter,
  selectEmitter,
} from '../sim/emitters';

const NOZZLE_HIT = 0.18;
const ANGLE_RING_IN = 0.12;
const ANGLE_RING_OUT = 0.62;
const HANDLE_LEN = 0.48;
const HANDLE_HIT = 0.22;

const DOUBLE_TAP_MS = 400;
const DOUBLE_TAP_DIST = 0.45;
const TAP_MOVE_DIST = 0.12;

type DragMode = 'move' | 'angle' | 'pending' | null;

function coarsePointer(): boolean {
  return window.matchMedia('(pointer: coarse)').matches;
}

function hitScale(): number {
  return coarsePointer() ? 1.85 : 1;
}

export function canvasToWorld(
  clientX: number,
  clientY: number,
  canvas: HTMLCanvasElement,
  p: Params,
): { x: number; y: number } {
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    return { x: 0, y: 0 };
  }
  const px = ((clientX - rect.left) / rect.width) * canvas.width;
  const py = ((clientY - rect.top) / rect.height) * canvas.height;
  return {
    x: (px / canvas.width) * p.domainW,
    y: (1 - py / canvas.height) * p.domainH,
  };
}

function angleFromWorld(em: Emitter, wx: number, wy: number): number {
  const dx = wx - em.x;
  const dy = wy - em.y;
  const deg = (Math.atan2(dx, dy) * 180) / Math.PI;
  return Math.max(-45, Math.min(45, deg));
}

function distToHandle(em: Emitter, wx: number, wy: number): number {
  const [dx, dy] = emitDirection(em);
  const hx = em.x + dx * HANDLE_LEN;
  const hy = em.y + dy * HANDLE_LEN;
  return Math.hypot(wx - hx, wy - hy);
}

function pickDragMode(em: Emitter, wx: number, wy: number): DragMode {
  const hs = hitScale();
  if (distToHandle(em, wx, wy) < HANDLE_HIT * hs) return 'angle';
  const d = Math.hypot(wx - em.x, wy - em.y);
  if (d < NOZZLE_HIT * hs) return 'move';
  if (d > ANGLE_RING_IN * hs && d < ANGLE_RING_OUT * hs) return 'angle';
  return null;
}

function emitterHitRadius(): number {
  return 0.22 * hitScale();
}

export function setupEmitterInteraction(
  canvas: HTMLCanvasElement,
  params: Params,
  onChange: () => void,
): void {
  let mode: DragMode = null;
  let dragOffX = 0;
  let dragOffY = 0;
  let activePointerId: number | null = null;
  let pendingDownX = 0;
  let pendingDownY = 0;
  let pendingPlaceTimer: ReturnType<typeof setTimeout> | null = null;
  const lastTap = { t: 0, x: 0, y: 0 };

  canvas.style.touchAction = 'none';

  function clearPendingPlace(): void {
    if (pendingPlaceTimer !== null) {
      clearTimeout(pendingPlaceTimer);
      pendingPlaceTimer = null;
    }
  }

  function placeEmitter(type: EmitterType, x: number, y: number): void {
    if (params.emitters.length >= MAX_EMITTERS) return;
    addEmitter(params, type, x, y);
    onChange();
    mode = 'move';
    dragOffX = 0;
    dragOffY = 0;
  }

  function scheduleWaterPlace(x: number, y: number): void {
    clearPendingPlace();
    pendingPlaceTimer = setTimeout(() => {
      pendingPlaceTimer = null;
      placeEmitter('water', x, y);
    }, DOUBLE_TAP_MS);
  }

  function isDoubleTap(x: number, y: number): boolean {
    const now = performance.now();
    const d = Math.hypot(x - lastTap.x, y - lastTap.y);
    return now - lastTap.t < DOUBLE_TAP_MS && d < DOUBLE_TAP_DIST;
  }

  function recordTap(x: number, y: number): void {
    lastTap.t = performance.now();
    lastTap.x = x;
    lastTap.y = y;
  }

  function setCursor(wx: number, wy: number): void {
    if (mode || coarsePointer()) return;
    const hit = findEmitterAt(params, wx, wy, emitterHitRadius());
    if (!hit) {
      canvas.style.cursor = 'crosshair';
      return;
    }
    const m = pickDragMode(hit, wx, wy);
    if (m === 'angle') canvas.style.cursor = 'grab';
    else if (m === 'move') canvas.style.cursor = 'move';
    else canvas.style.cursor = 'pointer';
  }

  function applyDrag(wx: number, wy: number): void {
    const em = getSelectedEmitter(params);
    if (!em || !mode || mode === 'pending') return;

    if (mode === 'move') {
      em.x = wx - dragOffX;
      em.y = wy - dragOffY;
      clampEmitterPosition(em, params);
      onChange();
    } else if (mode === 'angle') {
      em.angleDeg = angleFromWorld(em, wx, wy);
      onChange();
      if (!coarsePointer()) canvas.style.cursor = 'grabbing';
    }
  }

  function beginHitDrag(hit: Emitter, x: number, y: number): void {
    selectEmitter(params, hit.id);
    onChange();
    mode = pickDragMode(hit, x, y);
    if (mode === 'move') {
      dragOffX = x - hit.x;
      dragOffY = y - hit.y;
    } else if (mode === 'angle') {
      hit.angleDeg = angleFromWorld(hit, x, y);
      onChange();
    }
  }

  function onDown(e: PointerEvent): void {
    if (!e.isPrimary) return;
    if (e.pointerType === 'mouse' && e.button !== 0) return;

    const { x, y } = canvasToWorld(e.clientX, e.clientY, canvas, params);
    const hit = findEmitterAt(params, x, y, emitterHitRadius());

    if (hit) {
      clearPendingPlace();
      beginHitDrag(hit, x, y);
    } else if (params.emitters.length < MAX_EMITTERS) {
      if (e.pointerType === 'touch') {
        if (isDoubleTap(x, y)) {
          clearPendingPlace();
          lastTap.t = 0;
          placeEmitter('air', x, y);
        } else {
          recordTap(x, y);
          mode = 'pending';
          pendingDownX = x;
          pendingDownY = y;
        }
      } else {
        const type: EmitterType = e.shiftKey ? 'air' : 'water';
        placeEmitter(type, x, y);
      }
    } else {
      clearPendingPlace();
      selectEmitter(params, null);
      onChange();
    }

    activePointerId = e.pointerId;
    canvas.setPointerCapture(e.pointerId);
    window.addEventListener('pointermove', onMove, { passive: false });
    window.addEventListener('pointerup', onUp, { passive: false });
    window.addEventListener('pointercancel', onUp, { passive: false });
    e.preventDefault();
  }

  function onMove(e: PointerEvent): void {
    if (e.pointerId !== activePointerId) return;

    const events = e.getCoalescedEvents?.() ?? [e];
    for (const ev of events) {
      const { x, y } = canvasToWorld(ev.clientX, ev.clientY, canvas, params);

      if (mode === 'pending') {
        const moved = Math.hypot(x - pendingDownX, y - pendingDownY);
        if (moved > TAP_MOVE_DIST) {
          clearPendingPlace();
          placeEmitter('water', pendingDownX, pendingDownY);
          applyDrag(x, y);
        }
        continue;
      }

      if (!mode) {
        setCursor(x, y);
        continue;
      }
      applyDrag(x, y);
    }
    e.preventDefault();
  }

  function endDrag(e: PointerEvent): void {
    if (e.pointerId !== activePointerId) return;

    if (canvas.hasPointerCapture(e.pointerId)) {
      canvas.releasePointerCapture(e.pointerId);
    }
    window.removeEventListener('pointermove', onMove);
    window.removeEventListener('pointerup', onUp);
    window.removeEventListener('pointercancel', onUp);

    const { x, y } = canvasToWorld(e.clientX, e.clientY, canvas, params);

    if (mode === 'pending') {
      const moved = Math.hypot(x - pendingDownX, y - pendingDownY);
      if (moved <= TAP_MOVE_DIST) {
        scheduleWaterPlace(pendingDownX, pendingDownY);
      }
    }

    activePointerId = null;
    mode = null;
    setCursor(x, y);
  }

  function onUp(e: PointerEvent): void {
    endDrag(e);
    e.preventDefault();
  }

  canvas.addEventListener('pointerdown', onDown, { passive: false });
  canvas.addEventListener('pointermove', (e) => {
    if (activePointerId !== null) return;
    const { x, y } = canvasToWorld(e.clientX, e.clientY, canvas, params);
    setCursor(x, y);
  });
  canvas.addEventListener('pointerleave', () => {
    if (!mode && !coarsePointer()) canvas.style.cursor = 'crosshair';
  });
}
