import type { Params } from '../sim/params';
import type { VelSampler } from './velSampler';

const BEND_SCALE = 18;
const MAX_ANGLE = 1.15;

interface Blade {
  x: number;
  height: number;
  phase: number;
  thickness: number;
  angle: number;
}

export interface GrassLayer {
  resize: (w: number, h: number) => void;
  rebuild: (p: Params) => void;
  drawGround: (p: Params, sampler: VelSampler) => void;
  drawBlades: (p: Params, simTime: number, dt: number, sampler: VelSampler) => void;
}

export function createGrassLayer(overlay: HTMLCanvasElement, p: Params): GrassLayer {
  const ctx = overlay.getContext('2d')!;
  let blades: Blade[] = [];

  function rebuild(params: Params): void {
    const count = Math.round(params.grassDensity);
    blades = [];
    const margin = 0.08;
    for (let i = 0; i < count; i++) {
      const t = count <= 1 ? 0.5 : i / (count - 1);
      const x = margin + t * (params.domainW - 2 * margin) + (Math.random() - 0.5) * 0.06;
      blades.push({
        x,
        height: 0.22 + Math.random() * 0.38,
        phase: Math.random() * Math.PI * 2,
        thickness: 0.9 + Math.random() * 0.7,
        angle: 0,
      });
    }
    blades.sort((a, b) => a.x - b.x);
  }

  function resize(w: number, h: number): void {
    overlay.width = w;
    overlay.height = h;
  }

  function worldToPx(x: number, y: number, params: Params): [number, number] {
    const px = (x / params.domainW) * overlay.width;
    const py = (1 - y / params.domainH) * overlay.height;
    return [px, py];
  }

  function drawGround(params: Params, sampler: VelSampler): void {
    if (!params.showGrass && !params.showTrees && !params.showHouses) return;
    const groundStrip = overlay.height * 0.09;
    const alpha = params.showGrass ? 0.55 : 0.35;

    if (params.showWetGround) {
      const cols = 64;
      const colW = overlay.width / cols;
      for (let c = 0; c < cols; c++) {
        const x = (c + 0.5) / cols * params.domainW;
        const wv = sampler.sampleWet(x, params);
        const damp = 1 - wv * 0.45;
        const g = Math.round(83 * damp);
        const b = Math.round(45 * damp);
        ctx.fillStyle = `rgba(20, ${g}, ${b}, ${alpha})`;
        ctx.fillRect(c * colW, overlay.height - groundStrip, colW + 1, groundStrip);
      }
    } else {
      ctx.fillStyle = `rgba(20, 83, 45, ${alpha})`;
      ctx.fillRect(0, overlay.height - groundStrip, overlay.width, groundStrip);
    }
  }

  function drawBlades(params: Params, simTime: number, dt: number, sampler: VelSampler): void {
    if (!params.showGrass) return;

    const stiffness = Math.max(0.02, Math.min(0.98, params.grassStiffness));
    const response = (1 - stiffness) * 14;
    const step = Math.min(1, response * dt);

    ctx.strokeStyle = '#4ade80';
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    for (const blade of blades) {
      const vx = sampler.sampleVx(blade.x, 0.05, params);
      const target =
        Math.max(-MAX_ANGLE, Math.min(MAX_ANGLE, vx * BEND_SCALE)) +
        Math.sin(simTime * 2.1 + blade.phase) * 0.04;
      blade.angle += (target - blade.angle) * step;

      const [bx, by] = worldToPx(blade.x, 0, params);
      const bladePx = blade.height * overlay.height;
      const bendX = Math.sin(blade.angle) * bladePx;
      const tipX = bx + bendX;
      const tipY = by - bladePx;

      ctx.lineWidth = blade.thickness;
      ctx.beginPath();
      ctx.moveTo(bx, by);
      const midX = bx + bendX * 0.45;
      const midY = by - bladePx * 0.55;
      ctx.quadraticCurveTo(midX, midY, tipX, tipY);
      ctx.stroke();
    }
  }

  rebuild(p);

  return { resize, rebuild, drawGround, drawBlades };
}
