import type { Params } from '../sim/params';
import { SCENE_COLORS } from './sceneData';

export { SCENE_COLORS };

export function worldToPx(
  x: number,
  y: number,
  p: Params,
  canvasW: number,
  canvasH: number,
): [number, number] {
  return worldToPxDomain(x, y, p.domainW, p.domainH, canvasW, canvasH);
}

export function worldToPxDomain(
  x: number,
  y: number,
  domainW: number,
  domainH: number,
  canvasW: number,
  canvasH: number,
): [number, number] {
  const px = (x / domainW) * canvasW;
  const py = (1 - y / domainH) * canvasH;
  return [px, py];
}

export function lowPass(state: number, target: number, tau: number, dt: number): number {
  const a = 1 - Math.exp(-dt / Math.max(0.05, tau));
  return state + (target - state) * a;
}

export function lowPassRate(state: number, target: number, rate: number, dt: number): number {
  const a = 1 - Math.exp(-rate * dt);
  return state + (target - state) * a;
}

/** Ground strip height as fraction of canvas (matches grass layer). */
export function groundStripFrac(): number {
  return 0.09;
}

export function groundStripPx(canvasH: number): number {
  return canvasH * groundStripFrac();
}

export function groundLineY(canvasH: number): number {
  return canvasH - groundStripPx(canvasH);
}

/**
 * Painterly sky gradient from the legacy playground — drawn at partial opacity
 * so physics field overlays remain visible underneath.
 */
export function drawStylizedSky(
  ctx: CanvasRenderingContext2D,
  canvasW: number,
  canvasH: number,
  alpha = 0.42,
): void {
  const groundY = groundLineY(canvasH);
  const g = ctx.createLinearGradient(0, 0, 0, groundY + canvasH * 0.04);
  g.addColorStop(0, SCENE_COLORS.skyTop);
  g.addColorStop(0.45, SCENE_COLORS.skyMid);
  g.addColorStop(0.78, SCENE_COLORS.skyHorizon);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, canvasW, groundY + canvasH * 0.05);
  ctx.restore();
}

/** Rich ground band beneath the grass strip (prototype-style layers). */
export function drawStylizedGroundBase(
  ctx: CanvasRenderingContext2D,
  canvasW: number,
  canvasH: number,
  alpha = 0.88,
): void {
  const strip = groundStripPx(canvasH);
  const groundY = canvasH - strip;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = SCENE_COLORS.ground;
  ctx.fillRect(0, groundY, canvasW, strip);
  ctx.fillStyle = SCENE_COLORS.groundDark;
  for (let i = 0; i < 5; i++) {
    ctx.fillRect(0, groundY + 4 + i * (strip / 6), canvasW, 3);
  }
  ctx.strokeStyle = SCENE_COLORS.grass;
  ctx.globalAlpha = alpha * 0.18;
  ctx.beginPath();
  ctx.moveTo(0, groundY);
  ctx.lineTo(canvasW, groundY);
  ctx.stroke();
  ctx.restore();
}
