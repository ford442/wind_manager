// step.js — orchestrates the per-substep compute pass order:
//   emit -> update_droplets (scatter) -> apply_scatter -> advect ->
//   buoyancy -> divergence -> jacobi x30 -> project -> boundaries
// Accumulator clearing is folded into apply_scatter (atomicExchange with 0).

import { cellSize, qAmb } from './params.js';

const PARAM_BYTES = 80; // must match struct Params in common.wgsl

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

  const bg = (pipeline, entries) =>
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: entries.map((buffer, i) => ({
        binding: i,
        resource: { buffer },
      })).filter((e) => e.resource.buffer !== null),
    });

  // Binding orders must match the @binding indices in each shader.
  const emitBG = device.createBindGroup({
    layout: emitPipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 1, resource: { buffer: drops.pool } },
      { binding: 2, resource: { buffer: drops.counter } },
    ],
  });
  const updateBG = device.createBindGroup({
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
  const applyBG = bg(applyPipe, [
    uniform, fields.vel0, fields.T0, fields.q0,
    fields.accQ, fields.accT, fields.accM,
  ]);
  const advectBG = bg(advectPipe, [
    uniform, fields.vel0, fields.T0, fields.q0,
    fields.vel1, fields.T1, fields.q1,
  ]);
  const buoyBG = bg(buoyPipe, [uniform, fields.vel0, fields.T0, fields.q0]);
  const boundBG = bg(boundPipe, [uniform, fields.vel0, fields.T0, fields.q0]);
  const divBG = device.createBindGroup({
    layout: divPipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 1, resource: { buffer: fields.vel0 } },
      { binding: 2, resource: { buffer: fields.div } },
      { binding: 4, resource: { buffer: fields.p0 } },
    ],
  });
  const jacobiA = device.createBindGroup({
    layout: jacobiPipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 2, resource: { buffer: fields.div } },
      { binding: 3, resource: { buffer: fields.p0 } },
      { binding: 4, resource: { buffer: fields.p1 } },
    ],
  });
  const jacobiB = device.createBindGroup({
    layout: jacobiPipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 2, resource: { buffer: fields.div } },
      { binding: 3, resource: { buffer: fields.p1 } },
      { binding: 4, resource: { buffer: fields.p0 } },
    ],
  });
  const projBG = device.createBindGroup({
    layout: projPipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 1, resource: { buffer: fields.vel0 } },
      { binding: 3, resource: { buffer: fields.p0 } },
    ],
  });

  const paramData = new ArrayBuffer(PARAM_BYTES);
  const dv = new DataView(paramData);

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

  const sim = {
    time: 0,
    frame: 0,
    emitCarry: 0, // fractional droplets carried between substeps
  };

  const gx = Math.ceil(params.nx / 8);
  const gy = Math.ceil(params.ny / 8);
  const n = params.nx * params.ny;

  function encodeSubstep(emitCount) {
    const enc = device.createCommandEncoder();
    let cp = enc.beginComputePass();

    if (emitCount > 0) {
      cp.setPipeline(emitPipe);
      cp.setBindGroup(0, emitBG);
      cp.dispatchWorkgroups(Math.ceil(emitCount / 64));
    }
    cp.setPipeline(updatePipe);
    cp.setBindGroup(0, updateBG);
    cp.dispatchWorkgroups(Math.ceil(params.maxDroplets / 128));

    cp.setPipeline(applyPipe);
    cp.setBindGroup(0, applyBG);
    cp.dispatchWorkgroups(Math.ceil(n / 64));

    cp.setPipeline(advectPipe);
    cp.setBindGroup(0, advectBG);
    cp.dispatchWorkgroups(gx, gy);
    cp.end();

    // Copy advection results back so "current" is always the 0-buffers.
    enc.copyBufferToBuffer(fields.vel1, 0, fields.vel0, 0, n * 8);
    enc.copyBufferToBuffer(fields.T1, 0, fields.T0, 0, n * 4);
    enc.copyBufferToBuffer(fields.q1, 0, fields.q0, 0, n * 4);

    cp = enc.beginComputePass();
    cp.setPipeline(buoyPipe);
    cp.setBindGroup(0, buoyBG);
    cp.dispatchWorkgroups(gx, gy);

    cp.setPipeline(divPipe);
    cp.setBindGroup(0, divBG);
    cp.dispatchWorkgroups(gx, gy);

    cp.setPipeline(jacobiPipe);
    for (let it = 0; it < params.jacobiIters; it++) {
      cp.setBindGroup(0, it % 2 === 0 ? jacobiA : jacobiB);
      cp.dispatchWorkgroups(gx, gy);
    }
    // jacobiIters is even, so the final pressure lands in p0 (what project reads)

    cp.setPipeline(projPipe);
    cp.setBindGroup(0, projBG);
    cp.dispatchWorkgroups(gx, gy);

    cp.setPipeline(boundPipe);
    cp.setBindGroup(0, boundBG);
    cp.dispatchWorkgroups(gx, gy);
    cp.end();

    return enc.finish();
  }

  sim.step = (p) => {
    for (let s = 0; s < p.substeps; s++) {
      sim.emitCarry += p.emitRate * p.dt;
      const emitCount = Math.floor(sim.emitCarry);
      sim.emitCarry -= emitCount;

      packParams(p, emitCount, (sim.frame * 7 + s) * 2654435761 % 4294967295, sim.time);
      device.queue.writeBuffer(uniform, 0, paramData);
      device.queue.submit([encodeSubstep(emitCount)]);
      sim.time += p.dt;
    }
    sim.frame++;
  };

  sim.reset = (p) => {
    sim.time = 0;
    sim.frame = 0;
    sim.emitCarry = 0;
    fields.reset(device.queue, p);
    drops.reset(device.queue);
  };

  return sim;
}
