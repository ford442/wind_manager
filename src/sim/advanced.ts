import { clampEmitterPosition } from './emitters';
import { cellSize, type Params } from './params';

export const ADVANCED_DEFAULTS = {
  dt: 1 / 120,
  substeps: 2,
  jacobiIters: 30,
  damp: 0.02,
  relax: 1 / 120,
  domainW: 32.0,
  domainH: 8.0,
  nx: 512,
  ny: 128,
} as const;

export type AdvancedKey = keyof typeof ADVANCED_DEFAULTS;

export interface AdvancedSanitizeResult {
  warnings: string[];
  adjusted: string[];
  needsGridRebuild: boolean;
}

function snapGrid(n: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, Math.round(n / 8) * 8));
}

export function resetAdvancedToDefaults(p: Params): void {
  Object.assign(p, ADVANCED_DEFAULTS);
}

export function sanitizeAdvancedParams(
  p: Params,
  prev?: { nx: number; ny: number },
): AdvancedSanitizeResult {
  const warnings: string[] = [];
  const adjusted: string[] = [];
  const prevNx = prev?.nx ?? p.nx;
  const prevNy = prev?.ny ?? p.ny;

  const nxIn = p.nx;
  const nyIn = p.ny;
  p.nx = snapGrid(p.nx, 64, 512);
  p.ny = snapGrid(p.ny, 64, 256);
  if (p.nx !== nxIn) adjusted.push(`Grid width snapped to ${p.nx} (multiple of 8)`);
  if (p.ny !== nyIn) adjusted.push(`Grid height snapped to ${p.ny} (multiple of 8)`);

  p.domainW = Math.max(8, Math.min(40, p.domainW));
  p.domainH = Math.max(4, Math.min(12, p.domainH));

  if (p.dt > 1 / 40) {
    p.dt = 1 / 40;
    adjusted.push('Timestep capped at 25 ms');
  }
  if (p.dt < 1 / 480) {
    p.dt = 1 / 480;
    adjusted.push('Timestep raised to ~2.1 ms minimum');
  }

  p.substeps = Math.max(1, Math.min(8, Math.round(p.substeps)));

  if (p.jacobiIters < 8) {
    p.jacobiIters = 8;
    adjusted.push('Jacobi iterations raised to 8');
  }
  p.jacobiIters = Math.min(80, Math.round(p.jacobiIters));

  if (p.damp > 0.12) {
    p.damp = 0.12;
    adjusted.push('Damping capped at 0.12');
  }
  p.damp = Math.max(0, p.damp);

  if (p.relax > 0.1) {
    p.relax = 0.1;
    adjusted.push('T/q relaxation capped at 0.10');
  }
  p.relax = Math.max(0.002, p.relax);

  const h = cellSize(p);
  const cflDt = h / 22;
  const effectiveDt = p.dt / p.substeps;
  if (effectiveDt > cflDt * 1.15) {
    warnings.push(
      `Substep dt ${(effectiveDt * 1000).toFixed(1)} ms may exceed advection limit ` +
        `(~${(cflDt * 1000).toFixed(1)} ms for h=${(h * 1000).toFixed(0)} mm)`,
    );
    if (p.substeps < 4 && p.dt > cflDt * 1.4) {
      p.substeps = Math.min(8, p.substeps + 1);
      adjusted.push(`Substeps raised to ${p.substeps}`);
    }
  }

  if (p.jacobiIters < 18) {
    warnings.push('Low Jacobi count — divergence may linger (try ≥ 20)');
  }
  if (p.damp > 0.06) {
    warnings.push('High damping — flow may feel sluggish');
  }
  if (p.relax > 0.04) {
    warnings.push('Fast T/q relaxation — perturbations fade quickly');
  }
  if (p.nx * p.ny > 512 * 192) {
    warnings.push(`Large grid (${p.nx}×${p.ny}) — expect lower FPS`);
  }
  if (p.dt * p.substeps > 1 / 45) {
    warnings.push('Large frame timestep — droplet coupling may be stiff');
  }

  for (const em of p.emitters) {
    clampEmitterPosition(em, p);
  }

  const needsGridRebuild = p.nx !== prevNx || p.ny !== prevNy;
  return { warnings, adjusted, needsGridRebuild };
}

export function previewAdvancedWarnings(p: Params): string[] {
  const scratch: Params = {
    ...p,
    emitters: p.emitters.map((e) => ({ ...e })),
  };
  return sanitizeAdvancedParams(scratch, { nx: p.nx, ny: p.ny }).warnings;
}
