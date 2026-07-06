// params.js — simulation constants and UI-tunable state.
// Physical constants live in shaders/common.wgsl; only qSat is mirrored here
// because the CPU needs it to compute the ambient humidity for field init.

export function qSat(tC) {
  const es = 610.94 * Math.exp((17.625 * tC) / (tC + 243.04)); // Pa
  const p = 101325.0;
  return (0.622 * es) / (p - 0.378 * es);
}

export function defaultParams() {
  return {
    // grid / domain
    nx: 256,
    ny: 192,
    domainW: 8.0, // m
    domainH: 6.0, // m
    dt: 1 / 120,  // s per substep
    substeps: 2,  // per rendered frame (=> 1/60 s of sim time per frame)
    jacobiIters: 30,

    // ambient
    tAmb: 30.0, // degC
    rhAmb: 20,  // percent — dry air, strong evaporation, strong downdraft

    // the proof-of-mechanism toggle
    latentOn: true,

    // emitter
    emitX: 4.0,
    emitY: 0.3,
    emitAngleDeg: 0,   // from straight up
    emitSpreadDeg: 15, // half-angle
    emitSpeed: 10.0,   // m/s
    emitRate: 6000,    // droplets / s
    rMinUm: 50,        // µm
    rMaxUm: 800,       // µm
    maxDroplets: 200000,

    // long-run safeguards
    relax: 1 / 120, // 1/s, T & q relaxation toward ambient
    damp: 0.02,     // 1/s, velocity damping

    // run state / render
    paused: false,
    overlay: 2, // 0 none, 1 velocity, 2 temperature, 3 humidity
    showArrows: true,
    showDroplets: true,
  };
}

export function cellSize(p) {
  return p.domainW / p.nx;
}

export function qAmb(p) {
  return (p.rhAmb / 100) * qSat(p.tAmb);
}
