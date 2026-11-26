import { MAX_EMITTERS } from '../sim/emitters';
import { UniformWriter } from '../sim/uniformWriter';
import { LAYOUT } from '../render/sceneData';
import { N_SOLIDS } from '../sim/solids';
import { compileShaderModule, withValidationScope } from '../render/webgpu';
import type { VolumeFields } from './fields';
import { VOLUME_DOMAIN, spacing } from './grid';
import type { VolumeParams } from './params';
import { packVolumeSolids, VOL_SOLID_FLOATS, type VolumeSolids } from './solids';
import { volumeShaderParts } from './shaderParts';
import type { YardCamera } from './camera';
import type { PassTiming } from './step';

import volSceneSrc from '../shaders/vol_scene.wgsl?raw';

/**
 * Draws the yard from the volume's buffers (vol_scene.wgsl). Nothing is read
 * back: every pass binds the solver's current velocity / temperature buffers.
 *
 * Per frame, recorded into the caller's encoder after the solver:
 * - `scene_update` (compute): tree sway from the wind up each tree.
 * - Opaque pass (depth): sky, ground sheet, grass, solid boxes + upper
 *   trunks, crowns + emitter markers, drops (blended, no depth write).
 * - Overlay pass: temperature raymarch (reads the depth texture), side view.
 */
export interface YardRenderer {
  /** Zero the tree sway state (sim reset). */
  reset: () => void;
  draw: (args: DrawArgs) => void;
  /** Hero-tree crown sway (m) needs the state buffer; exposed for the probe. */
  state: GPUBuffer;
  destroy: () => void;
}

export interface DrawArgs {
  enc: GPUCommandEncoder;
  target: GPUTextureView;
  width: number;
  height: number;
  fields: VolumeFields;
  params: VolumeParams;
  camera: YardCamera;
  time: number;
  dt: number;
  /** Optional GPU timestamps: start of the scene update, end of the overlay pass. */
  timing?: PassTiming;
}

const N_TREES = 5;
const CROWN_BLOBS = 9;
const STATE_PER_TREE = 4;
const BLADES = 24000;
const BOX_VERTS = 60;
const BLADE_VERTS = 5;

const F_VOLUME = 1;
const F_DROPS = 2;
const F_GRASS = 4;
const F_YARD = 8;
const F_SIDE_SPEED = 16;

/** Byte offsets of `SceneU` in vol_scene.wgsl (384 bytes). */
export const SCENE_LAYOUT = {
  viewProj: 0, invViewProj: 64, camPos: 128, time: 140, domain: 144, tAmb: 156,
  h: 160, contrast: 172, n: 176, flags: 188, sideRect: 192, side: 208,
  nBlades: 224, maxDrops: 228, dt: 232, emitterCount: 236, light: 240, emitters: 256,
  BYTE_LENGTH: 256 + MAX_EMITTERS * 16,
} as const;

/** `VScene`: N_SOLIDS boxes then N_TREES trees. */
const SCENE_BYTES = N_SOLIDS * VOL_SOLID_FLOATS * 4 + N_TREES * 32;

export interface Viewports {
  yard: [number, number, number, number] | null;
  side: [number, number, number, number] | null;
}

/** Split the canvas: yard above, side slice below (4:1 strip), or one of them. */
export function viewports(p: VolumeParams, w: number, h: number): Viewports {
  if (p.viewMode === 'yard') return { yard: [0, 0, w, h], side: null };
  if (p.viewMode === 'side') return { yard: null, side: [0, 0, w, h] };
  const sideH = Math.max(1, Math.round(Math.min(h * 0.34, w / 4)));
  return { yard: [0, 0, w, Math.max(1, h - sideH)], side: [0, h - sideH, w, sideH] };
}

export async function createYardRenderer(device: GPUDevice, format: GPUTextureFormat): Promise<YardRenderer> {
  if (LAYOUT.trees.length !== N_TREES) throw new Error('vol_scene.wgsl expects 5 trees');
  const module = await compileShaderModule(device, 'vol_scene', volumeShaderParts('vol_scene', volSceneSrc));

  const V = GPUShaderStage.VERTEX;
  const F = GPUShaderStage.FRAGMENT;
  const C = GPUShaderStage.COMPUTE;
  const ro: GPUBufferBindingLayout = { type: 'read-only-storage' };
  const renderLayout = device.createBindGroupLayout({
    label: 'vol scene',
    entries: [
      { binding: 0, visibility: V | F, buffer: { type: 'uniform' } },
      { binding: 1, visibility: V | F, buffer: ro }, // vel
      { binding: 2, visibility: F, buffer: ro }, // T
      { binding: 3, visibility: V | F, buffer: ro }, // solid
      { binding: 4, visibility: V | F, buffer: ro }, // wet
      { binding: 5, visibility: V, buffer: ro }, // tree state
      { binding: 6, visibility: V, buffer: ro }, // boxes + trees
      { binding: 7, visibility: V, buffer: ro }, // drops
    ],
  });
  const depthLayout = device.createBindGroupLayout({
    label: 'vol depth',
    entries: [{ binding: 0, visibility: F, texture: { sampleType: 'depth' } }],
  });
  const computeLayout = device.createBindGroupLayout({
    label: 'vol scene update',
    entries: [
      { binding: 0, visibility: C, buffer: { type: 'uniform' } },
      { binding: 1, visibility: C, buffer: ro },
      { binding: 6, visibility: C, buffer: ro },
      { binding: 9, visibility: C, buffer: { type: 'storage' } },
    ],
  });
  const opaquePL = device.createPipelineLayout({ bindGroupLayouts: [renderLayout] });
  const overlayPL = device.createPipelineLayout({ bindGroupLayouts: [renderLayout, depthLayout] });
  const computePL = device.createPipelineLayout({ bindGroupLayouts: [computeLayout] });

  const depthFormat: GPUTextureFormat = 'depth24plus';
  const opaqueDepth: GPUDepthStencilState = { format: depthFormat, depthWriteEnabled: true, depthCompare: 'less' };
  const noWrite: GPUDepthStencilState = { format: depthFormat, depthWriteEnabled: false, depthCompare: 'less' };
  const blend: GPUBlendState = {
    color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
  };
  const premul: GPUBlendState = {
    color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
  };
  const render = (
    label: string, vs: string, fs: string, depth: GPUDepthStencilState | undefined,
    opts: { blend?: GPUBlendState; topology?: GPUPrimitiveTopology; overlay?: boolean } = {},
  ) => device.createRenderPipeline({
    label,
    layout: opts.overlay ? overlayPL : opaquePL,
    vertex: { module, entryPoint: vs },
    fragment: { module, entryPoint: fs, targets: [{ format, blend: opts.blend }] },
    primitive: { topology: opts.topology ?? 'triangle-list' },
    depthStencil: depth,
  });

  const pipes = await withValidationScope(device, 'vol scene', () => ({
    update: device.createComputePipeline({ label: 'scene_update', layout: computePL, compute: { module, entryPoint: 'scene_update' } }),
    sky: render('sky', 'vs_full', 'fs_sky', noWrite),
    ground: render('ground', 'vs_ground', 'fs_ground', opaqueDepth),
    grass: render('grass', 'vs_grass', 'fs_grass', opaqueDepth, { topology: 'triangle-strip' }),
    boxes: render('boxes', 'vs_boxes', 'fs_boxes', opaqueDepth),
    blobs: render('blobs', 'vs_blobs', 'fs_blobs', opaqueDepth),
    drops: render('drops', 'vs_drops', 'fs_drops', noWrite, { blend }),
    volume: render('volume', 'vs_full', 'fs_volume', undefined, { blend: premul, overlay: true }),
    side: render('side', 'vs_full', 'fs_side', undefined, { overlay: true }),
  }));

  const uniform = device.createBuffer({
    label: 'vol scene params', size: SCENE_LAYOUT.BYTE_LENGTH,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const state = device.createBuffer({
    label: 'vol tree state', size: N_TREES * STATE_PER_TREE * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const scene = device.createBuffer({
    label: 'vol scene layout', size: SCENE_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const trees = new Float32Array(N_TREES * 8);
  LAYOUT.trees.forEach((t, i) => trees.set([
    t.xFrac * VOLUME_DOMAIN.w, t.zFrac * VOLUME_DOMAIN.d, t.height, t.depth,
    t.trunkTau, t.foliageTau, t.windAmp, t.phase,
  ], i * 8));
  device.queue.writeBuffer(scene, N_SOLIDS * VOL_SOLID_FLOATS * 4, trees);

  let uploaded: VolumeSolids | null = null;
  let depthTex: GPUTexture | null = null;
  let depthView: GPUTextureView | null = null;
  let depthGroup: GPUBindGroup | null = null;
  const groups = new Map<GPUBuffer, { group: GPUBindGroup; T0: GPUBuffer; fields: VolumeFields }>();
  const computeGroups = new Map<GPUBuffer, GPUBindGroup>();
  const writer = new UniformWriter(SCENE_LAYOUT.BYTE_LENGTH);

  function ensureDepth(w: number, h: number): void {
    if (depthTex && depthTex.width === w && depthTex.height === h) return;
    depthTex?.destroy();
    depthTex = device.createTexture({
      label: 'vol depth', size: [w, h], format: depthFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    depthView = depthTex.createView();
    depthGroup = device.createBindGroup({ layout: depthLayout, entries: [{ binding: 0, resource: depthView }] });
  }

  function renderGroup(f: VolumeFields): GPUBindGroup {
    const hit = groups.get(f.vel0);
    if (hit && hit.T0 === f.T0 && hit.fields === f) return hit.group;
    if (groups.size >= 4) groups.clear();
    const e = (binding: number, buffer: GPUBuffer): GPUBindGroupEntry => ({ binding, resource: { buffer } });
    const group = device.createBindGroup({
      label: 'vol scene', layout: renderLayout,
      entries: [e(0, uniform), e(1, f.vel0), e(2, f.T0), e(3, f.solid), e(4, f.wet), e(5, state), e(6, scene), e(7, f.drops)],
    });
    groups.set(f.vel0, { group, T0: f.T0, fields: f });
    return group;
  }

  function computeGroup(f: VolumeFields): GPUBindGroup {
    let g = computeGroups.get(f.vel0);
    if (!g) {
      if (computeGroups.size >= 4) computeGroups.clear();
      g = device.createBindGroup({
        label: 'vol scene update', layout: computeLayout,
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: { buffer: f.vel0 } },
          { binding: 6, resource: { buffer: scene } },
          { binding: 9, resource: { buffer: state } },
        ],
      });
      computeGroups.set(f.vel0, g);
    }
    return g;
  }

  function reset(): void {
    device.queue.writeBuffer(state, 0, new Float32Array(N_TREES * STATE_PER_TREE));
  }
  reset();

  function writeUniform(a: DrawArgs, vp: Viewports): void {
    const L = SCENE_LAYOUT;
    const p = a.params;
    const g = a.fields.grid;
    const h = spacing(g);
    const yard = vp.yard ?? [0, 0, a.width, a.height];
    const cam = a.camera.viewProj(yard[2] / Math.max(1, yard[3]));
    new Float32Array(writer.buffer, L.viewProj, 16).set(cam.viewProj);
    new Float32Array(writer.buffer, L.invViewProj, 16).set(cam.invViewProj);
    cam.eye.forEach((v, i) => writer.f32(L.camPos + i * 4, v));
    writer.f32(L.time, a.time);
    [VOLUME_DOMAIN.w, VOLUME_DOMAIN.h, VOLUME_DOMAIN.d].forEach((v, i) => writer.f32(L.domain + i * 4, v));
    writer.f32(L.tAmb, p.tAmb);
    h.forEach((v, i) => writer.f32(L.h + i * 4, v));
    writer.f32(L.contrast, Math.max(0.2, p.volumeContrast));
    [g.nx, g.ny, g.nz].forEach((v, i) => writer.u32(L.n + i * 4, v));
    let flags = 0;
    if (p.showVolume) flags |= F_VOLUME;
    if (p.showDroplets) flags |= F_DROPS;
    if (p.showGrass) flags |= F_GRASS;
    if (vp.yard) flags |= F_YARD;
    if (p.sideField === 'speed') flags |= F_SIDE_SPEED;
    writer.u32(L.flags, flags);
    (vp.side ?? [0, 0, 1, 1]).forEach((v, i) => writer.f32(L.sideRect + i * 4, v));
    [p.sideZ, yard[2], yard[3], 0].forEach((v, i) => writer.f32(L.side + i * 4, v));
    writer.u32(L.nBlades, p.showGrass ? BLADES : 0);
    writer.u32(L.maxDrops, p.maxDroplets);
    writer.f32(L.dt, a.dt);
    const ems = p.emitters.slice(0, MAX_EMITTERS);
    writer.u32(L.emitterCount, ems.length);
    [0.45, 0.85, 0.3, 0].forEach((v, i) => writer.f32(L.light + i * 4, v));
    ems.forEach((em, i) => {
      [em.x, em.y, em.z, em.type === 'air' ? 1 : 0].forEach((v, k) => writer.f32(L.emitters + i * 16 + k * 4, v));
    });
    device.queue.writeBuffer(uniform, 0, writer.buffer);
  }

  return {
    state,
    reset,
    destroy() {
      for (const b of [uniform, state, scene]) b.destroy();
      depthTex?.destroy();
    },
    draw(a) {
      const f = a.fields;
      if (uploaded !== f.solids) {
        uploaded = f.solids;
        device.queue.writeBuffer(scene, 0, packVolumeSolids(f.solids.boxes));
      }
      const vp = viewports(a.params, a.width, a.height);
      writeUniform(a, vp);
      ensureDepth(a.width, a.height);

      const cpass = a.enc.beginComputePass({
        label: 'scene_update',
        timestampWrites: a.timing && { querySet: a.timing.querySet, beginningOfPassWriteIndex: a.timing.begin },
      });
      cpass.setPipeline(pipes.update);
      cpass.setBindGroup(0, computeGroup(f));
      cpass.dispatchWorkgroups(1);
      cpass.end();

      const group = renderGroup(f);
      const pass = a.enc.beginRenderPass({
        label: 'yard opaque',
        colorAttachments: [{ view: a.target, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0.03, g: 0.04, b: 0.06, a: 1 } }],
        depthStencilAttachment: { view: depthView!, depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1 },
      });
      if (vp.yard) {
        pass.setViewport(...vp.yard, 0, 1);
        pass.setBindGroup(0, group);
        pass.setPipeline(pipes.sky);
        pass.draw(3);
        pass.setPipeline(pipes.ground);
        pass.draw(6);
        if (a.params.showGrass) {
          pass.setPipeline(pipes.grass);
          pass.draw(BLADE_VERTS, BLADES);
        }
        pass.setPipeline(pipes.boxes);
        pass.draw(BOX_VERTS, N_SOLIDS + N_TREES);
        pass.setPipeline(pipes.blobs);
        pass.draw(6, N_TREES * CROWN_BLOBS + MAX_EMITTERS);
        if (a.params.showDroplets) {
          pass.setPipeline(pipes.drops);
          pass.draw(6, a.params.maxDroplets);
        }
      }
      pass.end();

      const over = a.enc.beginRenderPass({
        label: 'yard overlay',
        timestampWrites: a.timing && { querySet: a.timing.querySet, endOfPassWriteIndex: a.timing.end },
        colorAttachments: [{ view: a.target, loadOp: 'load', storeOp: 'store' }],
      });
      over.setBindGroup(0, group);
      over.setBindGroup(1, depthGroup!);
      if (vp.yard && a.params.showVolume) {
        over.setViewport(...vp.yard, 0, 1);
        over.setPipeline(pipes.volume);
        over.draw(3);
      }
      if (vp.side) {
        over.setViewport(...vp.side, 0, 1);
        over.setPipeline(pipes.side);
        over.draw(3);
      }
      over.end();
    },
  };
}
