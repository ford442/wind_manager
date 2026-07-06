import type { Params } from '../sim/params';
import type { VelSampler } from './velSampler';
import { LAYOUT, sceneX } from './sceneLayout';

interface Tree {
  xFrac: number;
  height: number;
  depth: number;
  phase: number;
  sampleYFrac: number;
  trunkTau: number;
  foliageTau: number;
  windAmp: number;
  trunkSway: number;
  foliageSway: number;
}

export interface TreesLayer {
  draw: (p: Params, simTime: number, dt: number, sampler: VelSampler) => void;
}

export function createTreesLayer(overlay: HTMLCanvasElement): TreesLayer {
  const ctx = overlay.getContext('2d')!;
  const trees: Tree[] = LAYOUT.trees.map((t) => ({
    ...t,
    trunkSway: 0,
    foliageSway: 0,
  }));

  function worldToPx(x: number, y: number, p: Params): [number, number] {
    const px = (x / p.domainW) * overlay.width;
    const py = (1 - y / p.domainH) * overlay.height;
    return [px, py];
  }

  function lowPass(state: number, target: number, tau: number, dt: number): number {
    const a = 1 - Math.exp(-dt / Math.max(0.05, tau));
    return state + (target - state) * a;
  }

  function drawTree(tree: Tree, p: Params): void {
    const groundX = sceneX(p, tree.xFrac);
    const hPx = (tree.height / p.domainH) * overlay.height * tree.depth;
    const trunkSway = tree.trunkSway;
    const foliageSway = tree.foliageSway;
    const topSway = foliageSway * 1.18;
    const midSway = trunkSway * 0.72 + foliageSway * 0.35;

    const [groundPx, groundY] = worldToPx(groundX, 0, p);
    const trunkW = (tree.depth > 0.85 ? 7 : tree.depth > 0.7 ? 5.5 : 4) * tree.depth;

    ctx.strokeStyle = '#3f2a1f';
    ctx.lineWidth = trunkW;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(groundPx, groundY);
    ctx.quadraticCurveTo(
      groundPx + midSway * 0.38,
      groundY - hPx * 0.52,
      groundPx + topSway * 0.52,
      groundY - hPx,
    );
    ctx.stroke();

    const layers = tree.height > 4.2 ? 5 : tree.height > 3.2 ? 4 : 3;
    const foliageBase = tree.depth > 0.85 ? '#166534' : '#15803d';
    ctx.fillStyle = foliageBase;

    for (let i = 0; i < layers; i++) {
      const t = i / Math.max(1, layers - 1);
      const ly = groundY - hPx * (0.16 + t * 0.72);
      const layerSway = trunkSway * (1 - t) * 0.45 + foliageSway * (0.55 + t * 0.55);
      const lx = groundPx + layerSway * (0.65 + t * 0.35);
      const r = (14 + (hPx / 7) * (0.8 + Math.sin(i + tree.phase) * 0.12) - i * 1.2) * tree.depth;

      ctx.beginPath();
      ctx.ellipse(lx, ly, r * 0.92, r * (0.68 + t * 0.08), layerSway * 0.008, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = 'rgba(74, 222, 128, 0.16)';
      ctx.beginPath();
      ctx.ellipse(lx - r * 0.22, ly - r * 0.18, r * 0.42, r * 0.3, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = foliageBase;
    }

    ctx.beginPath();
    ctx.ellipse(
      groundPx + topSway * 0.55,
      groundY - hPx - 5 * tree.depth,
      10 * tree.depth,
      8 * tree.depth,
      topSway * 0.015,
      0,
      Math.PI * 2,
    );
    ctx.fill();
    ctx.lineWidth = 1;
  }

  return {
    draw(p, simTime, dt, sampler) {
      if (!p.showTrees) return;

      const sorted = [...trees].sort((a, b) => a.depth - b.depth);

      for (const tree of sorted) {
        const wx = sceneX(p, tree.xFrac);
        const sampleY = tree.height * tree.sampleYFrac;
        const { vx, vy } = sampler.sampleWind(wx, sampleY, p);
        const swayScale = overlay.width / p.domainW * 0.11 * tree.depth;
        const windX = vx * 0.82 + vy * 0.14;
        const gust = windX * tree.windAmp * swayScale;
        const idle = Math.sin(simTime * 1.35 + tree.phase) * swayScale * 0.35;

        tree.trunkSway = lowPass(
          tree.trunkSway,
          gust * 0.42 + idle * 0.25,
          tree.trunkTau,
          dt,
        );
        tree.foliageSway = lowPass(
          tree.foliageSway,
          gust * 1.15 + idle * 0.9 + Math.sin(simTime * 2.4 + tree.phase * 1.7) * swayScale * 0.2,
          tree.foliageTau,
          dt,
        );
      }

      for (const tree of sorted) {
        drawTree(tree, p);
      }
    },
  };
}
