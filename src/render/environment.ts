import type { Params } from '../sim/params';
import type { VelSampler } from './velSampler';
import type { GrassLayer } from './grass';
import type { TreesLayer } from './trees';
import type { BackyardLayer } from './backyard';

export function envOverlayActive(p: Params): boolean {
  return p.showGrass || p.showTrees || p.showHouses || p.showClouds || p.showGroundMist || p.showWetGround;
}

export function drawEnvironmentOverlay(
  overlay: HTMLCanvasElement,
  p: Params,
  simTime: number,
  dt: number,
  sampler: VelSampler,
  grass: GrassLayer,
  trees: TreesLayer,
  backyard: BackyardLayer,
): void {
  const ctx = overlay.getContext('2d')!;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!envOverlayActive(p)) return;

  backyard.drawClouds(p, simTime, dt, sampler);
  grass.drawGround(p, sampler);
  backyard.drawGroundMist(p, simTime, sampler);
  grass.drawBlades(p, simTime, dt, sampler);
  backyard.drawHouses(p, simTime, dt, sampler);
  trees.draw(p, simTime, dt, sampler);
}
