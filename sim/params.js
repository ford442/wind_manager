export function qSat(tC) {
  const es = 610.94 * Math.exp((17.625 * tC) / (tC + 243.04));
  const p = 101325.0;
  return (0.622 * es) / (p - 0.378 * es);
}

export function defaultParams() {
  return {
    nx: 256,
    ny: 192,
    domainW: 8.0,
    domainH: 6.0,
    dt: 1 / 120,
    substeps: 2,
    jacobiIters: 30,

    tAmb: 30.0,
    rhAmb: 20,

    latentOn: true,

    emitX: 4.0,
    emitY: 0.3,
    emitAngleDeg: 0,
    emitSpreadDeg: 15,
    emitSpeed: 10.0,
    emitRate: 6000,
    rMinUm: 50,
    rMaxUm: 800,
    maxDroplets: 200000,

    relax: 1 / 120,
    damp: 0.02,

    paused: false,
    overlay: 2,
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
