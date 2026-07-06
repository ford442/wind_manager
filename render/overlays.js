import { cellSize, qAmb } from '../sim/params.js';

const RPARAM_BYTES = 32;
const ARROW_STRIDE = 8;

export async function createRenderer(device, format, fields, drops, params) {
  const common = await (await fetch('shaders/common.wgsl')).text();
  const src = await (await fetch('shaders/render.wgsl')).text();
  const module = device.createShaderModule({
    label: 'render',
    code: common + '\n' + src,
  });

  const blend = {
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
    fragment: { module, entryPoint: 'fs_arrows', targets: [{ format, blend }] },
    primitive: { topology: 'line-list' },
  });
  const dropPipe = device.createRenderPipeline({
    label: 'droplets',
    layout: 'auto',
    vertex: { module, entryPoint: 'vs_drops' },
    fragment: { module, entryPoint: 'fs_drops', targets: [{ format, blend }] },
    primitive: { topology: 'triangle-list' },
  });

  const uniform = device.createBuffer({
    size: RPARAM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  let overlayBG;
  let arrowBG;
  let dropBG;

  function rebuildBindGroups(f) {
    overlayBG = device.createBindGroup({
      layout: overlayPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: f.vel0 } },
        { binding: 2, resource: { buffer: f.T0 } },
        { binding: 3, resource: { buffer: f.q0 } },
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
  }

  rebuildBindGroups(fields);

  const data = new ArrayBuffer(RPARAM_BYTES);
  const dv = new DataView(data);

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
      device.queue.writeBuffer(uniform, 0, data);

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

      pass.end();
      device.queue.submit([enc.finish()]);
    },
  };
}
