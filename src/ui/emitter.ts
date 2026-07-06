import type { Params } from '../sim/params';
import type { Emitter } from '../sim/emitters';
import {
  MAX_EMITTERS,
  addEmitter,
  clampEmitterPosition,
  emitDirection,
  findEmitterAt,
  getSelectedEmitter,
  removeSelectedEmitter,
  selectEmitter,
} from '../sim/emitters';

const NOZZLE_HIT = 0.18;
const ANGLE_RING_IN = 0.12;
const ANGLE_RING_OUT = 0.62;
const HANDLE_LEN = 0.48;
const HANDLE_HIT = 0.22;

type DragMode = 'move' | 'angle' | null;

export function canvasToWorld(
  clientX: number,
  clientY: number,
  canvas: HTMLCanvasElement,
  p: Params,
): { x: number; y: number } {
  const rect = canvas.getBoundingClientRect();
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
  if (distToHandle(em, wx, wy) < HANDLE_HIT) return 'angle';
  const d = Math.hypot(wx - em.x, wy - em.y);
  if (d < NOZZLE_HIT) return 'move';
  if (d > ANGLE_RING_IN && d < ANGLE_RING_OUT) return 'angle';
  return null;
}

export function setupEmitterInteraction(
  canvas: HTMLCanvasElement,
  params: Params,
  onChange: () => void,
): void {
  let mode: DragMode = null;
  let dragOffX = 0;
  let dragOffY = 0;

  canvas.style.touchAction = 'none';

  function setCursor(wx: number, wy: number): void {
    if (mode) return;
    const hit = findEmitterAt(params, wx, wy);
    if (!hit) {
      canvas.style.cursor = 'crosshair';
      return;
    }
    const m = pickDragMode(hit, wx, wy);
    if (m === 'angle') canvas.style.cursor = 'grab';
    else if (m === 'move') canvas.style.cursor = 'move';
    else canvas.style.cursor = 'pointer';
  }

  function onDown(e: PointerEvent): void {
    if (e.button !== 0) return;
    const { x, y } = canvasToWorld(e.clientX, e.clientY, canvas, params);
    const hit = findEmitterAt(params, x, y);

    if (hit) {
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
    } else if (params.emitters.length < MAX_EMITTERS) {
      const type = e.shiftKey ? 'air' : 'water';
      addEmitter(params, type, x, y);
      onChange();
      const em = getSelectedEmitter(params)!;
      mode = 'move';
      dragOffX = 0;
      dragOffY = 0;
    } else {
      selectEmitter(params, null);
      onChange();
    }

    canvas.setPointerCapture(e.pointerId);
    e.preventDefault();
  }

  function onMove(e: PointerEvent): void {
    const { x, y } = canvasToWorld(e.clientX, e.clientY, canvas, params);
    const em = getSelectedEmitter(params);

    if (!mode || !em) {
      setCursor(x, y);
      return;
    }

    if (mode === 'move') {
      em.x = x - dragOffX;
      em.y = y - dragOffY;
      clampEmitterPosition(em, params);
      onChange();
    } else if (mode === 'angle') {
      em.angleDeg = angleFromWorld(em, x, y);
      onChange();
      canvas.style.cursor = 'grabbing';
    }
    e.preventDefault();
  }

  function onUp(e: PointerEvent): void {
    if (canvas.hasPointerCapture(e.pointerId)) {
      canvas.releasePointerCapture(e.pointerId);
    }
    mode = null;
    const { x, y } = canvasToWorld(e.clientX, e.clientY, canvas, params);
    setCursor(x, y);
  }

  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerup', onUp);
  canvas.addEventListener('pointercancel', onUp);
  canvas.addEventListener('pointerleave', (e) => {
    if (!mode) canvas.style.cursor = 'crosshair';
    else onUp(e);
  });
}
