//#region src/render/sceneData.ts
var e = {
	w: 32,
	h: 8
}, t = {
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
}, n = {
	emitterLeft: .032,
	emitterCenter: .5,
	emitterRight: .965,
	trees: [
		{
			xFrac: .06,
			height: 4.5,
			depth: .92,
			phase: 1.1,
			sampleYFrac: .62,
			trunkTau: 1.8,
			foliageTau: .55,
			windAmp: 1.05
		},
		{
			xFrac: .19,
			height: 3.1,
			depth: .78,
			phase: .35,
			sampleYFrac: .58,
			trunkTau: 1.2,
			foliageTau: .42,
			windAmp: .88
		},
		{
			xFrac: .36,
			height: 4.2,
			depth: .88,
			phase: 2.1,
			sampleYFrac: .64,
			trunkTau: 2,
			foliageTau: .58,
			windAmp: 1.15
		},
		{
			xFrac: .56,
			height: 3.4,
			depth: .72,
			phase: 1.75,
			sampleYFrac: .56,
			trunkTau: 1.1,
			foliageTau: .45,
			windAmp: .95
		},
		{
			xFrac: .91,
			height: 5.2,
			depth: 1,
			phase: 2.7,
			sampleYFrac: .68,
			trunkTau: 2.6,
			foliageTau: .72,
			windAmp: 1.55
		}
	],
	houses: [
		{
			xFrac: .28,
			wFrac: .032,
			hFrac: .11,
			chimney: !1
		},
		{
			xFrac: .52,
			wFrac: .026,
			hFrac: .095,
			chimney: !0
		},
		{
			xFrac: .74,
			wFrac: .028,
			hFrac: .1,
			chimney: !1
		}
	],
	clouds: [
		{
			xFrac: .06,
			yFrac: .88,
			s: .85,
			phase: 0
		},
		{
			xFrac: .28,
			yFrac: .92,
			s: 1.05,
			phase: 1.4
		},
		{
			xFrac: .52,
			yFrac: .86,
			s: .75,
			phase: 2.8
		},
		{
			xFrac: .76,
			yFrac: .9,
			s: .95,
			phase: 4.2
		},
		{
			xFrac: .93,
			yFrac: .87,
			s: .8,
			phase: 5.5
		}
	]
}, r = .48;
//#endregion
//#region src/render/sceneCanvas.ts
function i(e, t, n, r, i, a) {
	return [e / n * i, (1 - t / r) * a];
}
function a() {
	return .09;
}
function o(e) {
	return e * a();
}
function s(e) {
	return e - o(e);
}
function c(e, n, r, i = .42) {
	let a = s(r), o = e.createLinearGradient(0, 0, 0, a + r * .04);
	o.addColorStop(0, t.skyTop), o.addColorStop(.45, t.skyMid), o.addColorStop(.78, t.skyHorizon), e.save(), e.globalAlpha = i, e.fillStyle = o, e.fillRect(0, 0, n, a + r * .05), e.restore();
}
function l(e, n, r, i = .88) {
	let a = o(r), s = r - a;
	e.save(), e.globalAlpha = i, e.fillStyle = t.ground, e.fillRect(0, s, n, a), e.fillStyle = t.groundDark;
	for (let t = 0; t < 5; t++) e.fillRect(0, s + 4 + a / 6 * t, n, 3);
	e.strokeStyle = t.grass, e.globalAlpha = i * .18, e.beginPath(), e.moveTo(0, s), e.lineTo(n, s), e.stroke(), e.restore();
}
//#endregion
//#region src/render/sceneShared.ts
function u(e) {
	return e >= 4.5 ? "tall" : e >= 3.5 ? "medium" : "small";
}
function d(t, i) {
	let a = s(i);
	return n.trees.map((n) => ({
		x: n.xFrac * t,
		h: n.height / e.h * a * r * n.depth,
		type: u(n.height),
		phase: n.phase,
		sway: 0,
		windAmp: n.windAmp,
		depth: n.depth
	}));
}
function f(t, r) {
	let i = s(r);
	return n.houses.map((n) => ({
		x: n.xFrac * t,
		w: n.wFrac * e.w * (t / e.w),
		h: n.hFrac * e.h * (r / e.h) * 6.2,
		groundY: i,
		chimney: n.chimney,
		flagSway: 0
	}));
}
function p(t, r) {
	return n.clouds.map((n) => {
		let a = i(0, n.yFrac * e.h, e.w, e.h, t, r)[1];
		return {
			x: n.xFrac * t,
			y: a,
			s: n.s,
			phase: n.phase
		};
	});
}
function m(e, t) {
	let r = s(t);
	return {
		x: n.emitterLeft * e,
		y: r - 28,
		angle: -4
	};
}
//#endregion
export { e as DOMAIN, n as LAYOUT, r as PLAYGROUND_TREE_SCALE, t as SCENE_COLORS, m as defaultPlaygroundEmitter, l as drawStylizedGroundBase, c as drawStylizedSky, s as groundLineY, o as groundStripPx, p as layoutCloudsForCanvas, f as layoutHousesForCanvas, d as layoutTreesForCanvas, u as treeHeightClass, i as worldToPxDomain };
