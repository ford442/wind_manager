const s = { w: 32, h: 8 }, i = {
  skyTop: "#0f172a",
  skyMid: "#1e3a5f",
  skyHorizon: "#334155",
  ground: "#166534",
  groundDark: "#14532d",
  grass: "#4ade80",
  grassStroke: "#4ade80",
  trunk: "#3f2a1f",
  foliage: "#166534",
  foliageMid: "#15803d",
  foliageHi: "rgba(74, 222, 128, 0.16)",
  houseBody: "#334155",
  houseRoof: "#1e2937",
  cloud: "rgba(226, 232, 240, 0.72)"
}, l = {
  emitterLeft: 0.032,
  emitterCenter: 0.5,
  emitterRight: 0.965,
  trees: [
    { xFrac: 0.06, height: 4.5, depth: 0.92, phase: 1.1, sampleYFrac: 0.62, trunkTau: 1.8, foliageTau: 0.55, windAmp: 1.05 },
    { xFrac: 0.19, height: 3.1, depth: 0.78, phase: 0.35, sampleYFrac: 0.58, trunkTau: 1.2, foliageTau: 0.42, windAmp: 0.88 },
    { xFrac: 0.36, height: 4.2, depth: 0.88, phase: 2.1, sampleYFrac: 0.64, trunkTau: 2, foliageTau: 0.58, windAmp: 1.15 },
    { xFrac: 0.56, height: 3.4, depth: 0.72, phase: 1.75, sampleYFrac: 0.56, trunkTau: 1.1, foliageTau: 0.45, windAmp: 0.95 },
    { xFrac: 0.91, height: 5.2, depth: 1, phase: 2.7, sampleYFrac: 0.68, trunkTau: 2.6, foliageTau: 0.72, windAmp: 1.55 }
  ],
  houses: [
    { xFrac: 0.28, wFrac: 0.032, hFrac: 0.11, chimney: !1 },
    { xFrac: 0.52, wFrac: 0.026, hFrac: 0.095, chimney: !0 },
    { xFrac: 0.74, wFrac: 0.028, hFrac: 0.1, chimney: !1 }
  ],
  clouds: [
    { xFrac: 0.06, yFrac: 0.88, s: 0.85, phase: 0 },
    { xFrac: 0.28, yFrac: 0.92, s: 1.05, phase: 1.4 },
    { xFrac: 0.52, yFrac: 0.86, s: 0.75, phase: 2.8 },
    { xFrac: 0.76, yFrac: 0.9, s: 0.95, phase: 4.2 },
    { xFrac: 0.93, yFrac: 0.87, s: 0.8, phase: 5.5 }
  ]
}, c = 0.48;
function g(r, o, e, n, a, t) {
  const u = r / e * a, p = (1 - o / n) * t;
  return [u, p];
}
function f() {
  return 0.09;
}
function h(r) {
  return r * f();
}
function d(r) {
  return r - h(r);
}
function y(r, o, e, n = 0.42) {
  const a = d(e), t = r.createLinearGradient(0, 0, 0, a + e * 0.04);
  t.addColorStop(0, i.skyTop), t.addColorStop(0.45, i.skyMid), t.addColorStop(0.78, i.skyHorizon), r.save(), r.globalAlpha = n, r.fillStyle = t, r.fillRect(0, 0, o, a + e * 0.05), r.restore();
}
function m(r, o, e, n = 0.88) {
  const a = h(e), t = e - a;
  r.save(), r.globalAlpha = n, r.fillStyle = i.ground, r.fillRect(0, t, o, a), r.fillStyle = i.groundDark;
  for (let u = 0; u < 5; u++)
    r.fillRect(0, t + 4 + u * (a / 6), o, 3);
  r.strokeStyle = i.grass, r.globalAlpha = n * 0.18, r.beginPath(), r.moveTo(0, t), r.lineTo(o, t), r.stroke(), r.restore();
}
function F(r) {
  return r >= 4.5 ? "tall" : r >= 3.5 ? "medium" : "small";
}
function w(r, o) {
  const n = d(o);
  return l.trees.map((a) => ({
    x: a.xFrac * r,
    h: a.height / s.h * n * c * a.depth,
    type: F(a.height),
    phase: a.phase,
    sway: 0,
    windAmp: a.windAmp,
    depth: a.depth
  }));
}
function k(r, o) {
  const e = d(o);
  return l.houses.map((n) => ({
    x: n.xFrac * r,
    w: n.wFrac * s.w * (r / s.w),
    h: n.hFrac * s.h * (o / s.h) * 6.2,
    groundY: e,
    chimney: n.chimney,
    flagSway: 0
  }));
}
function T(r, o) {
  return l.clouds.map((e) => {
    const n = e.yFrac * s.h, a = g(0, n, s.w, s.h, r, o)[1];
    return {
      x: e.xFrac * r,
      y: a,
      s: e.s,
      phase: e.phase
    };
  });
}
function S(r, o) {
  const e = d(o);
  return {
    x: l.emitterLeft * r,
    y: e - 28,
    angle: -4
  };
}
export {
  s as DOMAIN,
  l as LAYOUT,
  c as PLAYGROUND_TREE_SCALE,
  i as SCENE_COLORS,
  S as defaultPlaygroundEmitter,
  m as drawStylizedGroundBase,
  y as drawStylizedSky,
  d as groundLineY,
  h as groundStripPx,
  T as layoutCloudsForCanvas,
  k as layoutHousesForCanvas,
  w as layoutTreesForCanvas,
  F as treeHeightClass,
  g as worldToPxDomain
};
