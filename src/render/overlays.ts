import { cellSize, qAmb, type Params } from '../sim/params';
import { MAX_EMITTERS } from '../sim/emitters';
import { tracerPoolSize, type TracerSystem } from '../sim/tracers';

import commonSrc from '../shaders/common.wgsl?raw';
import renderSrc from '../shaders/render.wgsl?raw';

const RPARAM_BYTES = 56;
const EMITTER_VIS_BYTES = 32;
const ARROW_STRIDE = 8;
const EMITTER_VERTS = 93;

export interface Renderer {
  rebuildBindGroups: (f: any) => void;
  draw: (view: GPUTextureView, p: Params, f: any, canvasW: number, canvasH: number) => void;
}

export async function createRenderer(
  device: GPUDevice,
  format: GPUTextureFormat,
  fields: any,
  drops: any,
  tracers: TracerSystem,
  _params: Params,
): Promise<Renderer> {
  const module = device.createShaderModule({
    label: 'render',
    code: commonSrc + '\n' + renderSrc,
  });

  const blend: GPUBlendState = {
    color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
  };

  const overlayPipe = device.createRenderPipeline({
    label: 'overlay',
    layout: 'auto',
    vertex: { module, entryPoint: 'vs_fullscreen' },
    fragment: { module, entryPoint: 'fs_overlay', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  const arrowPipe = device.createRenderPipeline({
    label: 'arrows',
    layout: 'auto',
    vertex: { module, entryPoint: 'vs_arrows' },
    fragment: { module, entryPoint: 'fs_arrows', targets: [{ format, blend: blend as any }] },
    primitive: { topology: 'line-list' },
  });
  const dropPipe = device.createRenderPipeline({
    label: 'droplets',
    layout: 'auto',
    vertex: { module, entryPoint: 'vs_drops' },
    fragment: { module, entryPoint: 'fs_drops', targets: [{ format, blend: blend as any }] },
    primitive: { topology: 'triangle-list' },
  });
  const emitterPipe = device.createRenderPipeline({
    label: 'emitter',
    layout: 'auto',
    vertex: { module, entryPoint: 'vs_emitter' },
    fragment: { module, entryPoint: 'fs_emitter', targets: [{ format, blend: blend as any }] },
    primitive: { topology: 'triangle-list' },
  });
  const streakPipe = device.createRenderPipeline({
    label: 'tracer_streaks',
    layout: 'auto',
    vertex: { module, entryPoint: 'vs_tracer_streaks' },
    fragment: { module, entryPoint: 'fs_tracers', targets: [{ format, blend: blend as any }] },
    primitive: { topology: 'line-list' },
  });
  const tracerDotPipe = device.createRenderPipeline({
    label: 'tracer_dots',
    layout: 'auto',
    vertex: { module, entryPoint: 'vs_tracer_dots' },
    fragment: { module, entryPoint: 'fs_tracers', targets: [{ format, blend: blend as any }] },
    primitive: { topology: 'triangle-list' },
  });

  const uniform = device.createBuffer({
    size: RPARAM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const emitterBuf = device.createBuffer({
    size: MAX_EMITTERS * EMITTER_VIS_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  let overlayBG;
  let arrowBG;
  let dropBG;
  let emitterBG;
  let streakBG;
  let tracerDotBG;

  function rebuildBindGroups(f) {
    const tracerEntries = [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 1, resource: { buffer: f.vel0 } },
      { binding: 7, resource: { buffer: tracers.pool } },
    ];
    overlayBG = device.createBindGroup({
      layout: overlayPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: f.vel0 } },
        { binding: 2, resource: { buffer: f.T0 } },
        { binding: 3, resource: { buffer: f.q0 } },
        { binding: 6, resource: { buffer: f.wet } },
      ],
    });
    arrowBG = device.createBindGroup({
      layout: arrowPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: f.vel0 } },
      ],
    });
    dropBG = device.createBindGroup({
      layout: dropPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 4, resource: { buffer: drops.pool } },
      ],
    });
    emitterBG = device.createBindGroup({
      layout: emitterPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 5, resource: { buffer: emitterBuf } },
      ],
    });
    streakBG = device.createBindGroup({
      layout: streakPipe.getBindGroupLayout(0),
      entries: tracerEntries,
    });
    tracerDotBG = device.createBindGroup({
      layout: tracerDotPipe.getBindGroupLayout(0),
      entries: tracerEntries,
    });
  }

  rebuildBindGroups(fields);

  const data = new ArrayBuffer(RPARAM_BYTES);
  const dv = new DataView(data);
  const emitVis = new ArrayBuffer(MAX_EMITTERS * EMITTER_VIS_BYTES);
  const ev = new DataView(emitVis);

  return {
    rebuildBindGroups,

    draw(view, p, f, canvasW, canvasH) {
      rebuildBindGroups(f);

      let o = 0;
      dv.setUint32(o, p.nx, true); o += 4;
      dv.setUint32(o, p.ny, true); o += 4;
      dv.setFloat32(o, cellSize(p), true); o += 4;
      dv.setUint32(o, p.overlay, true); o += 4;
      dv.setFloat32(o, p.tAmb, true); o += 4;
      dv.setFloat32(o, qAmb(p), true); o += 4;
      dv.setFloat32(o, canvasW, true); o += 4;
      dv.setFloat32(o, canvasH, true); o += 4;
      dv.setFloat32(o, p.showWetGround ? 1 : 0, true); o += 4;
      dv.setFloat32(o, performance.now() / 1000, true); o += 4;
      const pool = tracerPoolSize(p);
      dv.setUint32(o, pool, true); o += 4;
      dv.setFloat32(o, p.tracerLifetime, true); o += 4;
      dv.setFloat32(o, p.showTracers ? 1 : 0, true); o += 4;
      dv.setFloat32(o, p.showTracerStreaks ? 1 : 0, true); o += 4;
      device.queue.writeBuffer(uniform, 0, data);

      for (let i = 0; i < MAX_EMITTERS; i++) {
        const em = p.emitters[i];
        let eo = i * EMITTER_VIS_BYTES;
        if (em) {
          ev.setFloat32(eo, em.x, true); eo += 4;
          ev.setFloat32(eo, em.y, true); eo += 4;
          ev.setFloat32(eo, (em.angleDeg * Math.PI) / 180, true); eo += 4;
          ev.setFloat32(eo, (em.spreadDeg * Math.PI) / 180, true); eo += 4;
          ev.setFloat32(eo, em.id === p.selectedEmitterId ? 1 : 0, true); eo += 4;
          ev.setFloat32(eo, em.type === 'air' ? 1 : 0, true);
        }
      }
      device.queue.writeBuffer(emitterBuf, 0, emitVis);

      const enc = device.createCommandEncoder();
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0.03, g: 0.04, b: 0.06, a: 1 },
        }],
      });

      pass.setPipeline(overlayPipe);
      pass.setBindGroup(0, overlayBG);
      pass.draw(3);

      if (p.showTracers) {
        const n = tracerPoolSize(p);
        if (p.showTracerStreaks) {
          pass.setPipeline(streakPipe);
          pass.setBindGroup(0, streakBG);
          pass.draw(n * 2);
        }
        pass.setPipeline(tracerDotPipe);
        pass.setBindGroup(0, tracerDotBG);
        pass.draw(n * 6);
      }

      if (p.showDroplets) {
        pass.setPipeline(dropPipe);
        pass.setBindGroup(0, dropBG);
        pass.draw(6 * p.maxDroplets);
      }

      if (p.showArrows) {
        const instances = (p.nx / ARROW_STRIDE) * (p.ny / ARROW_STRIDE);
        pass.setPipeline(arrowPipe);
        pass.setBindGroup(0, arrowBG);
        pass.draw(6, instances);
      }

      if (p.emitters.length > 0) {
        pass.setPipeline(emitterPipe);
        pass.setBindGroup(0, emitterBG);
        pass.draw(EMITTER_VERTS, p.emitters.length);
      }

      pass.end();
      device.queue.submit([enc.finish()]);
    },
  };
}
