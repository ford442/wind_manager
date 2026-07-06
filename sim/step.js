import { cellSize, qAmb } from './params.js';

const PARAM_BYTES = 80;

async function loadShader(name) {
  const res = await fetch(`shaders/${name}`);
  if (!res.ok) throw new Error(`failed to load shaders/${name}`);
  return res.text();
}

export async function createSim(device, fields, drops, params) {
  const common = await loadShader('common.wgsl');
  const [srcDrops, srcApply, srcAdvect, srcBuoy, srcPressure] =
    await Promise.all([
      loadShader('droplets.wgsl'),
      loadShader('apply_scatter.wgsl'),
      loadShader('advect.wgsl'),
      loadShader('buoyancy.wgsl'),
      loadShader('pressure.wgsl'),
    ]);

  const mod = (src, label) =>
    device.createShaderModule({ label, code: common + '\n' + src });
  const modDrops = mod(srcDrops, 'droplets');
  const modApply = mod(srcApply, 'apply_scatter');
  const modAdvect = mod(srcAdvect, 'advect');
  const modBuoy = mod(srcBuoy, 'buoyancy');
  const modPressure = mod(srcPressure, 'pressure');

  const pipe = (module, entryPoint) =>
    device.createComputePipeline({
      label: entryPoint,
      layout: 'auto',
      compute: { module, entryPoint },
    });

  const emitPipe = pipe(modDrops, 'emit');
  const updatePipe = pipe(modDrops, 'update');
  const applyPipe = pipe(modApply, 'apply');
  const advectPipe = pipe(modAdvect, 'advect');
  const buoyPipe = pipe(modBuoy, 'buoyancy');
  const boundPipe = pipe(modBuoy, 'boundaries');
  const divPipe = pipe(modPressure, 'divergence');
  const jacobiPipe = pipe(modPressure, 'jacobi');
  const projPipe = pipe(modPressure, 'project');

  const uniform = device.createBuffer({
    size: PARAM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const paramData = new ArrayBuffer(PARAM_BYTES);
  const dv = new DataView(paramData);

  let emitBG;
  let updateBG;
  let applyBG;
  let advectBG;
  let buoyBG;
  let boundBG;
  let divBG;
  let jacobiA;
  let jacobiB;
  let projBG;

  function rebuildBindGroups() {
    emitBG = device.createBindGroup({
      layout: emitPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: drops.pool } },
        { binding: 2, resource: { buffer: drops.counter } },
      ],
    });
    updateBG = device.createBindGroup({
      layout: updatePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: drops.pool } },
        { binding: 3, resource: { buffer: fields.vel0 } },
        { binding: 4, resource: { buffer: fields.T0 } },
        { binding: 5, resource: { buffer: fields.q0 } },
        { binding: 6, resource: { buffer: fields.accQ } },
        { binding: 7, resource: { buffer: fields.accT } },
        { binding: 8, resource: { buffer: fields.accM } },
      ],
    });
    applyBG = device.createBindGroup({
      layout: applyPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: fields.vel0 } },
        { binding: 2, resource: { buffer: fields.T0 } },
        { binding: 3, resource: { buffer: fields.q0 } },
        { binding: 4, resource: { buffer: fields.accQ } },
        { binding: 5, resource: { buffer: fields.accT } },
        { binding: 6, resource: { buffer: fields.accM } },
      ],
    });
    advectBG = device.createBindGroup({
      layout: advectPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: fields.vel0 } },
        { binding: 2, resource: { buffer: fields.T0 } },
        { binding: 3, resource: { buffer: fields.q0 } },
        { binding: 4, resource: { buffer: fields.vel1 } },
        { binding: 5, resource: { buffer: fields.T1 } },
        { binding: 6, resource: { buffer: fields.q1 } },
      ],
    });
    buoyBG = device.createBindGroup({
      layout: buoyPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: fields.vel0 } },
        { binding: 2, resource: { buffer: fields.T0 } },
        { binding: 3, resource: { buffer: fields.q0 } },
      ],
    });
    boundBG = device.createBindGroup({
      layout: boundPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: fields.vel0 } },
        { binding: 2, resource: { buffer: fields.T0 } },
        { binding: 3, resource: { buffer: fields.q0 } },
      ],
    });
    divBG = device.createBindGroup({
      layout: divPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: fields.vel0 } },
        { binding: 2, resource: { buffer: fields.div } },
        { binding: 4, resource: { buffer: fields.p0 } },
      ],
    });
    jacobiA = device.createBindGroup({
      layout: jacobiPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 2, resource: { buffer: fields.div } },
        { binding: 3, resource: { buffer: fields.p0 } },
        { binding: 4, resource: { buffer: fields.p1 } },
      ],
    });
    jacobiB = device.createBindGroup({
      layout: jacobiPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 2, resource: { buffer: fields.div } },
        { binding: 3, resource: { buffer: fields.p1 } },
        { binding: 4, resource: { buffer: fields.p0 } },
      ],
    });
    projBG = device.createBindGroup({
      layout: projPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: fields.vel0 } },
        { binding: 3, resource: { buffer: fields.p0 } },
      ],
    });
  }

  rebuildBindGroups();

  function swapAdvectBuffers() {
    const tv = fields.vel0;
    fields.vel0 = fields.vel1;
    fields.vel1 = tv;
    const tt = fields.T0;
    fields.T0 = fields.T1;
    fields.T1 = tt;
    const tq = fields.q0;
    fields.q0 = fields.q1;
    fields.q1 = tq;
    rebuildBindGroups();
  }

  function packParams(p, emitCount, seed, time) {
    const h = cellSize(p);
    let o = 0;
    const u32 = (v) => { dv.setUint32(o, v >>> 0, true); o += 4; };
    const f32 = (v) => { dv.setFloat32(o, v, true); o += 4; };
    u32(p.nx); u32(p.ny); f32(h); f32(p.dt);
    f32(p.tAmb); f32(qAmb(p)); f32(p.latentOn ? 1 : 0); f32(time);
    f32(p.emitX); f32(p.emitY);
    f32((p.emitAngleDeg * Math.PI) / 180);
    f32((p.emitSpreadDeg * Math.PI) / 180);
    f32(p.emitSpeed); u32(emitCount);
    f32(p.rMinUm * 1e-6); f32(p.rMaxUm * 1e-6);
    u32(p.maxDroplets); u32(seed); f32(p.relax); f32(p.damp);
  }

  let time = 0;
  let frame = 0;
  let seed = 1;

  const wg = (n, size) => Math.ceil(n / size);
  const wg2d = (nx, ny, sx, sy) => [
    Math.ceil(nx / sx),
    Math.ceil(ny / sy),
  ];

  function substep(p, emitCount) {
    packParams(p, emitCount, seed++, time);
    device.queue.writeBuffer(uniform, 0, paramData);

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();

    if (emitCount > 0) {
      pass.setPipeline(emitPipe);
      pass.setBindGroup(0, emitBG);
      pass.dispatchWorkgroups(wg(emitCount, 64));
    }

    pass.setPipeline(updatePipe);
    pass.setBindGroup(0, updateBG);
    pass.dispatchWorkgroups(wg(p.maxDroplets, 128));

    pass.setPipeline(applyPipe);
    pass.setBindGroup(0, applyBG);
    pass.dispatchWorkgroups(wg(fields.n, 64));

    pass.setPipeline(advectPipe);
    pass.setBindGroup(0, advectBG);
    pass.dispatchWorkgroups(...wg2d(p.nx, p.ny, 8, 8));

    pass.setPipeline(buoyPipe);
    pass.setBindGroup(0, buoyBG);
    pass.dispatchWorkgroups(...wg2d(p.nx, p.ny, 8, 8));

    pass.setPipeline(divPipe);
    pass.setBindGroup(0, divBG);
    pass.dispatchWorkgroups(...wg2d(p.nx, p.ny, 8, 8));

    for (let i = 0; i < p.jacobiIters; i++) {
      pass.setPipeline(jacobiPipe);
      pass.setBindGroup(0, i % 2 === 0 ? jacobiA : jacobiB);
      pass.dispatchWorkgroups(...wg2d(p.nx, p.ny, 8, 8));
    }

    const pFinal = p.jacobiIters % 2 === 0 ? fields.p0 : fields.p1;
    if (p.jacobiIters % 2 !== 0) {
      projBG = device.createBindGroup({
        layout: projPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: { buffer: fields.vel0 } },
          { binding: 3, resource: { buffer: pFinal } },
        ],
      });
    }

    pass.setPipeline(projPipe);
    pass.setBindGroup(0, projBG);
    pass.dispatchWorkgroups(...wg2d(p.nx, p.ny, 8, 8));

    pass.setPipeline(boundPipe);
    pass.setBindGroup(0, boundBG);
    pass.dispatchWorkgroups(...wg2d(p.nx, p.ny, 8, 8));

    pass.end();
    device.queue.submit([enc.finish()]);

    swapAdvectBuffers();
    time += p.dt;
  }

  return {
    time: 0,
    frame: 0,
    getFields: () => fields,
    rebuildBindGroups,

    reset(p) {
      time = 0;
      frame = 0;
      seed = 1;
      fields.reset(device.queue, p);
      drops.reset(device.queue);
      rebuildBindGroups();
    },

    step(p) {
      const emitPerSubstep = Math.max(0, Math.round(p.emitRate * p.dt));
      for (let s = 0; s < p.substeps; s++) {
        substep(p, emitPerSubstep);
      }
      this.time = time;
      this.frame = ++frame;
    },
  };
}
