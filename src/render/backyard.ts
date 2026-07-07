import type { Params } from '../sim/params';
import type { VelSampler } from './velSampler';
import { LAYOUT, sceneX, sceneY } from './sceneLayout';
import { lowPassRate, SCENE_COLORS, worldToPx } from './sceneCanvas';

interface House {
  xFrac: number;
  wFrac: number;
  hFrac: number;
  chimney: boolean;
  flagSway: number;
  smokePhase: number;
}

interface Cloud {
  xFrac: number;
  drift: number;
  yFrac: number;
  s: number;
  phase: number;
}

const MIST_SAMPLES = 18;

export interface BackyardLayer {
  drawClouds: (p: Params, simTime: number, dt: number, sampler: VelSampler) => void;
  drawGroundMist: (p: Params, simTime: number, sampler: VelSampler) => void;
  drawHouses: (p: Params, simTime: number, dt: number, sampler: VelSampler) => void;
}

export function createBackyardLayer(overlay: HTMLCanvasElement): BackyardLayer {
  const ctx = overlay.getContext('2d')!;
  const houses: House[] = LAYOUT.houses.map((h, i) => ({
    ...h,
    flagSway: 0,
    smokePhase: i * 1.7,
  }));
  const clouds: Cloud[] = LAYOUT.clouds.map((c) => ({
    xFrac: c.xFrac,
    drift: 0,
    yFrac: c.yFrac,
    s: c.s,
    phase: c.phase,
  }));

  function worldToPxLocal(x: number, y: number, p: Params): [number, number] {
    return worldToPx(x, y, p, overlay.width, overlay.height);
  }

  function drawCloud(c: Cloud, p: Params, simTime: number, sampler: VelSampler, dt: number): void {
    const baseX = sceneX(p, c.xFrac);
    const cy = sceneY(p, c.yFrac);
    const { vx, vy } = sampler.sampleWind(baseX + c.drift, cy, p);
    c.drift += vx * dt * 0.22 + 0.012 * dt;
    const span = p.domainW * 0.08;
    if (c.drift > span) c.drift = -span;
    if (c.drift < -span) c.drift = span;

    const wx = baseX + c.drift;
    const [cx, cyp] = worldToPxLocal(wx, cy, p);
    const bob = Math.sin(simTime * 0.45 + c.phase) * 2 + vy * 1.5;
    const s = c.s * (overlay.width / p.domainW) * 0.028;

    ctx.fillStyle = SCENE_COLORS.cloud;
    ctx.beginPath();
    ctx.ellipse(cx, cyp + bob, 26 * s, 11 * s, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.ellipse(cx - 18 * s, cyp - 1 + bob, 17 * s, 12 * s, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.ellipse(cx + 19 * s, cyp + 1 + bob, 15 * s, 9.5 * s, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.ellipse(cx - 5 * s, cyp - 7 + bob, 12 * s, 9 * s, 0, 0, Math.PI * 2);
    ctx.fill();
  }

  function drawHouse(h: House, p: Params, simTime: number, dt: number, sampler: VelSampler): void {
    const wx = sceneX(p, h.xFrac);
    const hw = h.wFrac * p.domainW;
    const hh = h.hFrac * p.domainH;
    const { vx } = sampler.sampleWind(wx, hh * 0.35, p);
    const swayPx = overlay.width / p.domainW;
    h.flagSway = lowPassRate(h.flagSway, vx * swayPx * 0.55, 5.5, dt);

    const [hx, groundY] = worldToPxLocal(wx, 0, p);
    const wPx = (hw / p.domainW) * overlay.width;
    const hPx = (hh / p.domainH) * overlay.height;

    ctx.fillStyle = SCENE_COLORS.houseBody;
    ctx.fillRect(hx - wPx / 2, groundY - hPx, wPx, hPx);

    ctx.fillStyle = SCENE_COLORS.houseRoof;
    ctx.beginPath();
    ctx.moveTo(hx - wPx / 2 - 5, groundY - hPx);
    ctx.lineTo(hx, groundY - hPx - hPx * 0.55);
    ctx.lineTo(hx + wPx / 2 + 5, groundY - hPx);
    ctx.fill();

    ctx.fillStyle = '#64748b';
    const winW = wPx * 0.11;
    const winH = hPx * 0.14;
    ctx.fillRect(hx - wPx * 0.28, groundY - hPx * 0.68, winW, winH);
    ctx.fillRect(hx + wPx * 0.12, groundY - hPx * 0.68, winW, winH);
    if (wPx > 55) {
      ctx.fillRect(hx - wPx * 0.28, groundY - hPx * 0.42, winW, winH);
    }

    ctx.fillStyle = '#1e2937';
    ctx.fillRect(hx - wPx * 0.06, groundY - hPx * 0.22, wPx * 0.12, hPx * 0.22);

    if (h.chimney) {
      const chimX = hx + wPx * 0.22;
      const chimBase = groundY - hPx - hPx * 0.42;
      ctx.fillStyle = '#475569';
      ctx.fillRect(chimX, chimBase - hPx * 0.18, wPx * 0.08, hPx * 0.18);

      ctx.strokeStyle = 'rgba(203, 213, 225, 0.55)';
      ctx.lineWidth = 1.2;
      const smokeDrift = h.flagSway * 0.35 + vx * swayPx * 0.2;
      for (let i = 0; i < 3; i++) {
        const t = i / 2;
        const sy = chimBase - hPx * (0.22 + t * 0.2);
        const sx = chimX + wPx * 0.04 + smokeDrift * (0.4 + t * 0.5);
        const wobble = Math.sin(simTime * 2.2 + h.smokePhase + i) * 3;
        ctx.beginPath();
        ctx.ellipse(sx + wobble, sy, 4 + t * 3, 3 + t * 2, 0, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.lineWidth = 1;
    } else {
      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 1.5;
      const fx = hx + wPx / 2 + 4;
      const fy = groundY - hPx - hPx * 0.35;
      const fs = h.flagSway;
      ctx.beginPath();
      ctx.moveTo(fx, fy);
      ctx.quadraticCurveTo(fx + 12 + fs * 0.8, fy - 2 + fs * 0.2, fx + 22 + fs * 1.6, fy + 4 + fs * 0.6);
      ctx.stroke();
      ctx.lineWidth = 1;
    }
  }

  return {
    drawClouds(p, simTime, dt, sampler) {
      if (!p.showClouds) return;
      for (const c of clouds) {
        drawCloud(c, p, simTime, sampler, dt);
      }
    },

    drawGroundMist(p, simTime, sampler) {
      if (!p.showGroundMist) return;
      const stripH = overlay.height * 0.14;
      const baseY = overlay.height - stripH * 0.35;

      for (let i = 0; i < MIST_SAMPLES; i++) {
        const t = (i + 0.5) / MIST_SAMPLES;
        const wx = 0.2 + t * (p.domainW - 0.4);
        const wy = 0.12 + Math.sin(i * 1.3) * 0.04;
        const { vx } = sampler.sampleWind(wx, wy, p);
        const speed = Math.abs(vx);

        let sprayShadow = 0;
        for (const em of p.emitters) {
          if (em.type !== 'water') continue;
          const dist = Math.hypot(wx - em.x, wy - em.y);
          sprayShadow = Math.max(sprayShadow, Math.max(0, 1 - dist / 3.5) * 0.35);
        }
        const gust = Math.min(1, speed / 1.8) * 0.5;
        const alpha = Math.min(0.42, sprayShadow + gust * 0.28);
        if (alpha < 0.06) continue;

        const [px] = worldToPxLocal(wx, 0, p);
        const wobble = Math.sin(simTime * 1.6 + i * 0.9) * stripH * 0.08;
        const drift = vx * (overlay.width / p.domainW) * 8;
        const r = stripH * (0.22 + (i % 3) * 0.04);

        const grad = ctx.createRadialGradient(px + drift, baseY + wobble, 0, px + drift, baseY + wobble, r);
        grad.addColorStop(0, `rgba(140, 200, 230, ${alpha})`);
        grad.addColorStop(0.55, `rgba(100, 170, 210, ${alpha * 0.45})`);
        grad.addColorStop(1, 'rgba(80, 140, 180, 0)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.ellipse(px + drift, baseY + wobble, r * 1.35, r * 0.55, vx * 0.04, 0, Math.PI * 2);
        ctx.fill();
      }
    },

    drawHouses(p, simTime, dt, sampler) {
      if (!p.showHouses) return;
      for (const h of houses) {
        drawHouse(h, p, simTime, dt, sampler);
      }
    },
  };
}
