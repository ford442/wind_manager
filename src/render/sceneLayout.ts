import type { Params } from '../sim/params';
import { DOMAIN, LAYOUT } from './sceneData';

export { DOMAIN, LAYOUT };

/** Horizontal fraction [0, 1] along the domain width. */
export function sceneX(p: Params, xFrac: number): number {
  return xFrac * p.domainW;
}

/** Vertical fraction [0, 1] from ground up. */
export function sceneY(p: Params, yFrac: number): number {
  return yFrac * p.domainH;
}

export function heroTreeX(p: Params): number {
  return sceneX(p, LAYOUT.trees[4].xFrac);
}

export function heroTreeReachFt(p: Params, emitterX = sceneX(p, LAYOUT.emitterLeft)): number {
  return (heroTreeX(p) - emitterX) * 3.28084;
}
