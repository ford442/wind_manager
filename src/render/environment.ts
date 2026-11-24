import { cellSize, type Params } from '../sim/params';
import type { Fields } from '../sim/fields';
import { MAX_EMITTERS } from '../sim/emitters';
import { UniformWriter } from '../sim/uniformWriter';
import { LAYOUT } from './sceneData';
import { N_SOLIDS, SOLID_DEF_FLOATS, packSolidDefs, type SolidLayout } from '../sim/solids';
import {
  blendedColorTarget,
  compileShaderModule,
  withValidationScope,
  workgroups1d,
} from './webgpu';

import commonSrc from '../shaders/common.wgsl?raw';
import environmentSrc from '../shaders/environment.wgsl?raw';

/**
 * Stylized backyard (sky, ground, grass, trees, houses, clouds, mist) drawn in
 * WebGPU from the sim's velocity buffer. Nothing is read back to the CPU.
 *
 * Per frame, recorded into the caller's encoder after the sim step:
 * - {@link EnvironmentRenderer.update}: writes the 192-byte `EnvParams` uniform
 *   (once per frame, before submit) and records `env_update`, which advances the
 *   stateful sway filters (blade angles, tree trunk/foliage lag, flags, cloud
 *   drift) in a small GPU state buffer.
 * - {@link EnvironmentRenderer.draw}: issues the backdrop / shape / stroke draws
 *   inside the field render pass. Vertex shaders sample `vel` themselves.
 *
 * Houses and trunks are drawn from the solid boxes the sim rasterized
 * (`Fields.solids`), re-uploaded whenever the mask is rewritten, so a drawn
 * house covers exactly the cells that block the flow.
 *
 * Instance ranges and state offsets are constants in `environment.wgsl`; the
 * ones mirrored here must stay in sync with it.
 */
export interface EnvironmentRenderer {
  update: (
    enc: GPUCommandEncoder,
    p: Params,
    f: Fields,
    simTime: number,
    dt: number,
    canvasW: number,
    canvasH: number,
  ) => void;
  draw: (pass: GPURenderPassEncoder, p: Params) => void;
  /** Zero sway / drift state (e.g. on sim reset). */
  reset: () => void;
}

/** Mirrors of environment.wgsl constants. */
const N_TREES = 5;
const N_HOUSES = 3;
const N_CLOUDS = 5;
const N_MIST = 18;
export const MAX_BLADES = 1024;
const STATE_FLOATS = 32 + MAX_BLADES;
const OBJ_THREADS = 16;
const SH_CLOUD = 0;
const SH_MIST = 20;
const SH_HOUSE = 38;
const HOUSE_PRIMS = 12;
const SH_TREE = 74;
const TREE_PRIMS = 11;
const ST_TRUNK = 0;
const ST_FLAG = 5;
const ST_BLADE = 8;
const STROKE_VERTS = 2 * (12 + 1);

const F_GRASS = 1;
const F_TREES = 2;
const F_HOUSES = 4;
const F_CLOUDS = 8;
const F_MIST = 16;
const F_WET = 32;
const F_MOISTURE = 64;
const F_ACTIVE = 128;

/** Byte offsets of `EnvParams` in environment.wgsl. */
export const ENV_PARAMS_LAYOUT = {
  nx: 0,
  ny: 4,
  h: 8,
  nBlades: 12,
  domainW: 16,
  domainH: 20,
  canvasW: 24,
  canvasH: 28,
  simTime: 32,
  dt: 36,
  skyAlpha: 40,
  flags: 44,
  grassResp: 48,
  rhAmb: 52,
  waterCount: 56,
  water: 64,
  BYTE_LENGTH: 64 + MAX_EMITTERS * 16,
} as const;

/** `SceneLayout.solids` starts after trees, houses and clouds. */
const SCENE_SOLIDS_OFFSET = N_TREES * 32 + N_HOUSES * 16 + N_CLOUDS * 16;
const SCENE_BYTES = SCENE_SOLIDS_OFFSET + N_SOLIDS * SOLID_DEF_FLOATS * 4;

export function envOverlayActive(p: Params): boolean {
  if (!p.stylizedView) return false;
  return p.showGrass || p.showTrees || p.showHouses || p.showClouds || p.showGroundMist
    || p.showWetGround || p.showGroundMoisture;
}

export function bladeCount(p: Params): number {
  return Math.max(0, Math.min(MAX_BLADES, Math.round(p.grassDensity)));
}

function envFlags(p: Params): number {
  let f = 0;
  if (p.showGrass) f |= F_GRASS;
  if (p.showTrees) f |= F_TREES;
  if (p.showHouses) f |= F_HOUSES;
  if (p.showClouds) f |= F_CLOUDS;
  if (p.showGroundMist) f |= F_MIST;
  if (p.showWetGround) f |= F_WET;
  if (p.showGroundMoisture) f |= F_MOISTURE;
  if (envOverlayActive(p)) f |= F_ACTIVE;
  return f;
}

function packSceneLayout(): Float32Array {
  if (
    LAYOUT.trees.length !== N_TREES ||
    LAYOUT.houses.length !== N_HOUSES ||
    LAYOUT.clouds.length !== N_CLOUDS
  ) {
    throw new Error(
      'environment.wgsl expects 5 trees, 3 houses, 5 clouds; update N_* there and in environment.ts',
    );
  }
  const out = new Float32Array(SCENE_BYTES / 4);
  let o = 0;
  for (const t of LAYOUT.trees) {
    out.set([t.xFrac, t.height, t.depth, t.phase, t.sampleYFrac, t.trunkTau, t.foliageTau, t.windAmp], o);
    o += 8;
  }
  for (const h of LAYOUT.houses) {
    out.set([h.xFrac, h.wFrac, h.hFrac, h.chimney ? 1 : 0], o);
    o += 4;
  }
  for (const c of LAYOUT.clouds) {
    out.set([c.xFrac, c.yFrac, c.s, c.phase], o);
    o += 4;
  }
  return out;
}

/** Trees drawn back to front, as the canvas layer did (stable sort by depth). */
const TREE_ORDER: readonly number[] = LAYOUT.trees
  .map((t, i) => ({ d: t.depth, i }))
  .sort((a, b) => a.d - b.d)
  .map((t) => t.i);

export async function createEnvironmentRenderer(
  device: GPUDevice,
  format: GPUTextureFormat,
): Promise<EnvironmentRenderer> {
  const module = await compileShaderModule(device, 'environment', [
    { name: 'generated', code: `const MAX_EMITTERS : u32 = ${MAX_EMITTERS}u;` },
    { name: 'common.wgsl', code: commonSrc },
    { name: 'environment.wgsl', code: environmentSrc },
  ]);

  const V = GPUShaderStage.VERTEX;
  const F = GPUShaderStage.FRAGMENT;
  const ro: GPUBufferBindingLayout = { type: 'read-only-storage' };
  const renderLayout = device.createBindGroupLayout({
    label: 'environment render',
    entries: [
      { binding: 0, visibility: V | F, buffer: { type: 'uniform' } },
      { binding: 1, visibility: V, buffer: ro },
      { binding: 2, visibility: V, buffer: ro },
      { binding: 3, visibility: V, buffer: ro },
      { binding: 4, visibility: F, buffer: ro },
      { binding: 5, visibility: F, buffer: ro },
      { binding: 6, visibility: V | F, buffer: ro }, // wet: ground (F) + wall faces (V)
      { binding: 7, visibility: F, buffer: ro },
    ],
  });
  const C = GPUShaderStage.COMPUTE;
  const computeLayout = device.createBindGroupLayout({
    label: 'environment update',
    entries: [
      { binding: 0, visibility: C, buffer: { type: 'uniform' } },
      { binding: 1, visibility: C, buffer: ro },
      { binding: 3, visibility: C, buffer: ro },
      { binding: 8, visibility: C, buffer: { type: 'storage' } },
    ],
  });
  const renderPL = device.createPipelineLayout({ bindGroupLayouts: [renderLayout] });
  const computePL = device.createPipelineLayout({ bindGroupLayouts: [computeLayout] });

  const pipes = await withValidationScope(device, 'environment', () => {
    const target = [blendedColorTarget(format)];
    return {
      update: device.createComputePipeline({
        label: 'env_update',
        layout: computePL,
        compute: { module, entryPoint: 'env_update' },
      }),
      backdrop: device.createRenderPipeline({
        label: 'env_backdrop',
        layout: renderPL,
        vertex: { module, entryPoint: 'vs_env_backdrop' },
        fragment: { module, entryPoint: 'fs_env_backdrop', targets: target },
        primitive: { topology: 'triangle-list' },
      }),
      shapes: device.createRenderPipeline({
        label: 'env_shapes',
        layout: renderPL,
        vertex: { module, entryPoint: 'vs_env_shapes' },
        fragment: { module, entryPoint: 'fs_env_shapes', targets: target },
        primitive: { topology: 'triangle-list' },
      }),
      strokes: device.createRenderPipeline({
        label: 'env_strokes',
        layout: renderPL,
        vertex: { module, entryPoint: 'vs_env_strokes' },
        fragment: { module, entryPoint: 'fs_env_strokes', targets: target },
        primitive: { topology: 'triangle-strip' },
      }),
    };
  });

  const uniform = device.createBuffer({
    label: 'env params',
    size: ENV_PARAMS_LAYOUT.BYTE_LENGTH,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const state = device.createBuffer({
    label: 'env state',
    size: STATE_FLOATS * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const scene = device.createBuffer({
    label: 'env scene layout',
    size: SCENE_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(scene, 0, packSceneLayout());

  const computeBG = new Map<GPUBuffer, GPUBindGroup>();
  interface RenderBG { group: GPUBindGroup; T0: GPUBuffer; q0: GPUBuffer; wet: GPUBuffer; qDep: GPUBuffer }
  const renderBG = new Map<GPUBuffer, RenderBG>();

  // vel/T/q ping-pong between two buffer sets; a grid rebuild replaces them, so
  // the caches are bounded and rebuilt when any buffer identity changes.
  function computeGroup(f: Fields): GPUBindGroup {
    let bg = computeBG.get(f.vel0);
    if (!bg) {
      if (computeBG.size >= 4) computeBG.clear();
      bg = device.createBindGroup({
        label: 'env_update',
        layout: computeLayout,
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: { buffer: f.vel0 } },
          { binding: 3, resource: { buffer: scene } },
          { binding: 8, resource: { buffer: state } },
        ],
      });
      computeBG.set(f.vel0, bg);
    }
    return bg;
  }

  function renderGroup(f: Fields): GPUBindGroup {
    const hit = renderBG.get(f.vel0);
    if (hit && hit.T0 === f.T0 && hit.q0 === f.q0 && hit.wet === f.wet && hit.qDep === f.qDep) {
      return hit.group;
    }
    if (renderBG.size >= 4) renderBG.clear();
    const group = device.createBindGroup({
      label: 'env render',
      layout: renderLayout,
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: f.vel0 } },
        { binding: 2, resource: { buffer: state } },
        { binding: 3, resource: { buffer: scene } },
        { binding: 4, resource: { buffer: f.T0 } },
        { binding: 5, resource: { buffer: f.q0 } },
        { binding: 6, resource: { buffer: f.wet } },
        { binding: 7, resource: { buffer: f.qDep } },
      ],
    });
    renderBG.set(f.vel0, { group, T0: f.T0, q0: f.q0, wet: f.wet, qDep: f.qDep });
    return group;
  }

  let uploadedSolids: SolidLayout | null = null;
  function syncSolids(f: Fields): void {
    if (uploadedSolids === f.solids) return;
    uploadedSolids = f.solids;
    device.queue.writeBuffer(scene, SCENE_SOLIDS_OFFSET, packSolidDefs(f.solids.boxes));
  }

  const writer = new UniformWriter(ENV_PARAMS_LAYOUT.BYTE_LENGTH);
  const L = ENV_PARAMS_LAYOUT;
  let frameGroup: GPUBindGroup | null = null;

  function reset(): void {
    device.queue.writeBuffer(state, 0, new Float32Array(STATE_FLOATS));
  }
  reset();

  return {
    reset,

    update(enc, p, f, simTime, dt, canvasW, canvasH) {
      frameGroup = null;
      if (!p.stylizedView) return;
      syncSolids(f);
      const nBlades = bladeCount(p);
      const stiffness = Math.max(0.02, Math.min(0.98, p.grassStiffness));
      writer.u32(L.nx, p.nx);
      writer.u32(L.ny, p.ny);
      writer.f32(L.h, cellSize(p));
      writer.u32(L.nBlades, nBlades);
      writer.f32(L.domainW, p.domainW);
      writer.f32(L.domainH, p.domainH);
      writer.f32(L.canvasW, canvasW);
      writer.f32(L.canvasH, canvasH);
      writer.f32(L.simTime, simTime);
      writer.f32(L.dt, dt);
      writer.f32(L.skyAlpha, p.overlay === 0 ? 0.72 : 0.38);
      writer.u32(L.flags, envFlags(p));
      writer.f32(L.grassResp, (1 - stiffness) * 14);
      writer.f32(L.rhAmb, p.rhAmb / 100);
      let water = 0;
      for (const em of p.emitters) {
        if (em.type !== 'water' || water >= MAX_EMITTERS) continue;
        writer.f32(L.water + water * 16, em.x);
        writer.f32(L.water + water * 16 + 4, em.y);
        water++;
      }
      writer.u32(L.waterCount, water);
      // Written once per frame and consumed by this frame's submit only.
      device.queue.writeBuffer(uniform, 0, writer.buffer);

      if (envOverlayActive(p)) {
        const pass = enc.beginComputePass({ label: 'env_update' });
        pass.setPipeline(pipes.update);
        pass.setBindGroup(0, computeGroup(f));
        pass.dispatchWorkgroups(workgroups1d(OBJ_THREADS + nBlades, 64));
        pass.end();
      }
      frameGroup = renderGroup(f);
    },

    draw(pass, p) {
      if (!frameGroup || !p.stylizedView) return;
      pass.setBindGroup(0, frameGroup);
      pass.setPipeline(pipes.backdrop);
      pass.draw(3);
      if (!envOverlayActive(p)) return;

      if (p.showClouds) {
        pass.setPipeline(pipes.shapes);
        pass.draw(6, N_CLOUDS * 4, 0, SH_CLOUD);
      }
      if (p.showGroundMist) {
        pass.setPipeline(pipes.shapes);
        pass.draw(6, N_MIST, 0, SH_MIST);
      }
      const nBlades = bladeCount(p);
      if (p.showGrass && nBlades > 0) {
        pass.setPipeline(pipes.strokes);
        pass.draw(STROKE_VERTS, nBlades, 0, ST_BLADE);
      }
      if (p.showHouses) {
        pass.setPipeline(pipes.shapes);
        pass.draw(6, N_HOUSES * HOUSE_PRIMS, 0, SH_HOUSE);
        pass.setPipeline(pipes.strokes);
        pass.draw(STROKE_VERTS, N_HOUSES, 0, ST_FLAG);
      }
      if (p.showTrees) {
        for (const t of TREE_ORDER) {
          pass.setPipeline(pipes.strokes);
          pass.draw(STROKE_VERTS, 1, 0, ST_TRUNK + t);
          pass.setPipeline(pipes.shapes);
          pass.draw(6, TREE_PRIMS, 0, SH_TREE + t * TREE_PRIMS);
        }
      }
    },
  };
}
