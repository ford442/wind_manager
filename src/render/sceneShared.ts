/**
 * Browser-safe scene helpers exported for the legacy art playground (`public/scene-shared.js`).
 * Physics sim layers import from sceneCanvas / sceneLayout directly.
 */
import {
  DOMAIN,
  LAYOUT,
  PLAYGROUND_TREE_SCALE,
  SCENE_COLORS,
} from './sceneData';
import {
  drawStylizedGroundBase,
  drawStylizedSky,
  groundLineY,
  groundStripPx,
  worldToPxDomain,
} from './sceneCanvas';

export {
  DOMAIN,
  LAYOUT,
  PLAYGROUND_TREE_SCALE,
  SCENE_COLORS,
  drawStylizedGroundBase,
  drawStylizedSky,
  groundLineY,
  groundStripPx,
  worldToPxDomain,
};

export function treeHeightClass(heightM: number): 'tall' | 'medium' | 'small' {
  if (heightM >= 4.5) return 'tall';
  if (heightM >= 3.5) return 'medium';
  return 'small';
}

export function layoutTreesForCanvas(canvasW: number, canvasH: number) {
  const groundY = groundLineY(canvasH);
  const skyH = groundY;
  return LAYOUT.trees.map((t) => ({
    x: t.xFrac * canvasW,
    h: (t.height / DOMAIN.h) * skyH * PLAYGROUND_TREE_SCALE * t.depth,
    type: treeHeightClass(t.height),
    phase: t.phase,
    sway: 0,
    windAmp: t.windAmp,
    depth: t.depth,
  }));
}

export function layoutHousesForCanvas(canvasW: number, canvasH: number) {
  const groundY = groundLineY(canvasH);
  return LAYOUT.houses.map((h) => ({
    x: h.xFrac * canvasW,
    w: h.wFrac * DOMAIN.w * (canvasW / DOMAIN.w),
    h: h.hFrac * DOMAIN.h * (canvasH / DOMAIN.h) * 6.2,
    groundY,
    chimney: h.chimney,
    flagSway: 0,
  }));
}

export function layoutCloudsForCanvas(canvasW: number, canvasH: number) {
  return LAYOUT.clouds.map((c) => {
    const worldY = c.yFrac * DOMAIN.h;
    const y = worldToPxDomain(0, worldY, DOMAIN.w, DOMAIN.h, canvasW, canvasH)[1];
    return {
      x: c.xFrac * canvasW,
      y,
      s: c.s,
      phase: c.phase,
    };
  });
}

export function defaultPlaygroundEmitter(canvasW: number, canvasH: number) {
  const groundY = groundLineY(canvasH);
  return {
    x: LAYOUT.emitterLeft * canvasW,
    y: groundY - 28,
    angle: -4,
  };
}
