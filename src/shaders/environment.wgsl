// environment.wgsl — stylized backyard (sky, ground, grass, trees, houses,
// clouds, mist) drawn in WebGPU, sampling the sim's velocity buffer directly.
//
// Assembled as: generated (MAX_EMITTERS) + common.wgsl + this file.
// Consumer: src/render/environment.ts (uniform layout ENV_PARAMS_LAYOUT there).
//
// All drawing math is in canvas pixels with y pointing down, matching the 2D
// canvas overlay this replaces, then converted to clip space at the end.
//
// Frame flow (one command buffer, after the sim step):
//   1. `env_update` (compute) advances the stateful low-pass filters — blade
//      angles, tree trunk/foliage sway (the hero-tree lag), flag sway, cloud
//      drift — into `state`.
//   2. In the field render pass: `fs_env_backdrop` (sky, ground strip,
//      ground moisture), `vs_env_shapes` (clouds, mist, houses, foliage) and
//      `vs_env_strokes` (blades, trunks, flags) read `vel` and `state`.

const N_TREES  : u32 = 5u;
const N_HOUSES : u32 = 3u;
const N_CLOUDS : u32 = 5u;
const N_MIST   : u32 = 18u;
const MAX_BLADES : u32 = 1024u;

// `state` layout (f32 indices).
const S_TREE  : u32 = 0u;   // [trunk, foliage] per tree
const S_FLAG  : u32 = 10u;  // one per house
const S_CLOUD : u32 = 13u;  // drift (m) per cloud
const S_BLADE : u32 = 32u;  // angle (rad) per blade
// Threads [0, OBJ_THREADS) update objects; the rest update blades.
const OBJ_THREADS : u32 = 16u;

// Shape instance ranges (vs_env_shapes).
const SH_CLOUD : u32 = 0u;                       // 4 ellipses per cloud
const SH_MIST  : u32 = 20u;                      // 1 gradient ellipse per sample
const SH_HOUSE : u32 = 38u;                      // HOUSE_PRIMS per house
const HOUSE_PRIMS : u32 = 10u;
const SH_TREE  : u32 = 68u;                      // TREE_PRIMS per tree
const TREE_PRIMS : u32 = 11u;

// Stroke instance ranges (vs_env_strokes).
const ST_TRUNK : u32 = 0u;
const ST_FLAG  : u32 = 5u;
const ST_BLADE : u32 = 8u;
const STROKE_SEGS : u32 = 12u;

// EnvParams.flags
const F_GRASS    : u32 = 1u;
const F_TREES    : u32 = 2u;
const F_HOUSES   : u32 = 4u;
const F_CLOUDS   : u32 = 8u;
const F_MIST     : u32 = 16u;
const F_WET      : u32 = 32u;
const F_MOISTURE : u32 = 64u;
const F_ACTIVE   : u32 = 128u;

struct EnvParams {
  nx         : u32,
  ny         : u32,
  h          : f32,
  n_blades   : u32,
  domain_w   : f32,
  domain_h   : f32,
  canvas_w   : f32,
  canvas_h   : f32,
  sim_time   : f32,
  dt         : f32,
  sky_alpha  : f32,
  flags      : u32,
  grass_resp : f32,
  rh_amb     : f32,
  water_count: u32,
  _pad0      : u32,
  // xy: water emitter position (m).
  water      : array<vec4f, MAX_EMITTERS>,
};

// Static backyard layout from src/render/sceneData.ts (LAYOUT), uploaded once.
struct TreeDef {
  a : vec4f, // xFrac, height (m), depth, phase
  b : vec4f, // sampleYFrac, trunkTau, foliageTau, windAmp
};

struct SceneLayout {
  trees  : array<TreeDef, N_TREES>,
  houses : array<vec4f, N_HOUSES>, // xFrac, wFrac, hFrac, chimney (0/1)
  clouds : array<vec4f, N_CLOUDS>, // xFrac, yFrac, s, phase
};

@group(0) @binding(0) var<uniform> E : EnvParams;
@group(0) @binding(1) var<storage, read> vel : array<vec2f>;
@group(0) @binding(2) var<storage, read> state : array<f32>;
@group(0) @binding(3) var<storage, read> scene : SceneLayout;
@group(0) @binding(4) var<storage, read> tf : array<f32>;
@group(0) @binding(5) var<storage, read> qf : array<f32>;
@group(0) @binding(6) var<storage, read> wet : array<f32>;
@group(0) @binding(7) var<storage, read> q_dep : array<f32>;
@group(0) @binding(8) var<storage, read_write> state_rw : array<f32>;

// --- Field sampling ----------------------------------------------------------

fn sample_vel(pos: vec2f) -> vec2f {
  let nx = i32(E.nx);
  let b = bilin_at(pos, E.h, nx, i32(E.ny));
  return bilin_blend(
    b,
    vel[cell_index(b.i0, b.j0, nx)], vel[cell_index(b.i1, b.j0, nx)],
    vel[cell_index(b.i0, b.j1, nx)], vel[cell_index(b.i1, b.j1, nx)],
  );
}

struct Col1 { i0 : i32, i1 : i32, fx : f32 };

fn col_at(world_x: f32) -> Col1 {
  let nx = i32(E.nx);
  let gx = clamp(world_x / E.h - 0.5, 0.0, f32(nx - 1) - 1.0e-4);
  let i0 = i32(floor(gx));
  return Col1(i0, min(i0 + 1, nx - 1), gx - f32(i0));
}

// Ground moisture index from 1D wet/q_dep and the bottom two T/q rows only.
fn ground_moisture(world_x: f32) -> vec3f {
  let c = col_at(world_x);
  let nx = i32(E.nx);
  let wv = mix(wet[c.i0], wet[c.i1], c.fx);
  let qd = mix(q_dep[c.i0], q_dep[c.i1], c.fx);
  let k00 = cell_index(c.i0, 0, nx);
  let k10 = cell_index(c.i1, 0, nx);
  let k01 = cell_index(c.i0, 1, nx);
  let k11 = cell_index(c.i1, 1, nx);
  let t = mix(mix(tf[k00], tf[k01], 0.55), mix(tf[k10], tf[k11], 0.55), c.fx);
  let q = mix(mix(qf[k00], qf[k01], 0.55), mix(qf[k10], qf[k11], 0.55), c.fx);
  let qs = q_sat(t);
  var rh_s = 0.0;
  if (qs > 1.0e-6) { rh_s = clamp(q / qs, 0.0, 1.0); }
  let rh_a = E.rh_amb;
  let vapor = max(0.0, (rh_s - rh_a) / max(1.0 - rh_a, 0.06));
  let gm = min(1.0, max(wv, max(qd * 0.95, vapor * 0.68)));
  return vec3f(gm, wv, qd);
}

// --- Helpers -----------------------------------------------------------------

fn has(f: u32) -> bool { return (E.flags & f) != 0u; }

fn px_per_m() -> f32 { return E.canvas_w / E.domain_w; }

fn world_to_px(p: vec2f) -> vec2f {
  return vec2f(p.x / E.domain_w * E.canvas_w, (1.0 - p.y / E.domain_h) * E.canvas_h);
}

fn px_to_clip(p: vec2f) -> vec4f {
  return vec4f(p.x / E.canvas_w * 2.0 - 1.0, 1.0 - p.y / E.canvas_h * 2.0, 0.0, 1.0);
}

fn low_pass(s: f32, goal: f32, tau: f32, dt: f32) -> f32 {
  return s + (goal - s) * (1.0 - exp(-dt / max(0.05, tau)));
}

fn low_pass_rate(s: f32, goal: f32, rate: f32, dt: f32) -> f32 {
  return s + (goal - s) * (1.0 - exp(-rate * dt));
}

fn hash01(i: u32, k: u32) -> f32 {
  return f32(pcg(i * 4u + k + 0x9e3779b9u)) / 4294967296.0;
}

fn rgb(hex: u32) -> vec3f {
  return vec3f(f32((hex >> 16u) & 255u), f32((hex >> 8u) & 255u), f32(hex & 255u)) / 255.0;
}

struct Blade { x : f32, height : f32, phase : f32, thickness : f32 };

// Static blade attributes from the index (replaces Math.random in the old rebuild).
fn blade_def(i: u32) -> Blade {
  let n = E.n_blades;
  let margin = 0.08;
  var t = 0.5;
  if (n > 1u) { t = f32(i) / f32(n - 1u); }
  var b : Blade;
  b.x = margin + t * (E.domain_w - 2.0 * margin) + (hash01(i, 0u) - 0.5) * 0.06;
  b.height = 0.22 + hash01(i, 1u) * 0.38;
  b.phase = hash01(i, 2u) * 2.0 * PI;
  b.thickness = 0.9 + hash01(i, 3u) * 0.7;
  return b;
}

fn house_px(hd: vec4f) -> vec4f {
  // hx (px), ground y (px), wPx, hPx
  let wx = hd.x * E.domain_w;
  let hw = hd.y * E.domain_w;
  let hh = hd.z * E.domain_h;
  return vec4f(world_to_px(vec2f(wx, 0.0)), hw / E.domain_w * E.canvas_w, hh / E.domain_h * E.canvas_h);
}

fn house_vx(hd: vec4f) -> f32 {
  return sample_vel(vec2f(hd.x * E.domain_w, hd.z * E.domain_h * 0.35)).x;
}

// --- 1. State update ---------------------------------------------------------

@compute @workgroup_size(64)
fn env_update(@builtin(global_invocation_id) gid: vec3u) {
  let g = gid.x;
  let dt = E.dt;
  let time = E.sim_time;

  if (g < N_TREES) {
    let td = scene.trees[g];
    let wx = td.a.x * E.domain_w;
    let sample_y = td.a.y * td.b.x;
    let v = sample_vel(vec2f(wx, sample_y));
    let sway_scale = px_per_m() * 0.11 * td.a.z;
    let wind_x = v.x * 0.82 + v.y * 0.14;
    let gust = wind_x * td.b.w * sway_scale;
    let phase = td.a.w;
    let idle = sin(time * 1.35 + phase) * sway_scale * 0.35;
    let k = S_TREE + g * 2u;
    state_rw[k] = low_pass(state_rw[k], gust * 0.42 + idle * 0.25, td.b.y, dt);
    state_rw[k + 1u] = low_pass(
      state_rw[k + 1u],
      gust * 1.15 + idle * 0.9 + sin(time * 2.4 + phase * 1.7) * sway_scale * 0.2,
      td.b.z,
      dt,
    );
    return;
  }

  if (g < N_TREES + N_HOUSES) {
    let i = g - N_TREES;
    let vx = house_vx(scene.houses[i]);
    state_rw[S_FLAG + i] = low_pass_rate(state_rw[S_FLAG + i], vx * px_per_m() * 0.55, 5.5, dt);
    return;
  }

  if (g < N_TREES + N_HOUSES + N_CLOUDS) {
    let i = g - N_TREES - N_HOUSES;
    let cd = scene.clouds[i];
    let base_x = cd.x * E.domain_w;
    let cy = cd.y * E.domain_h;
    var drift = state_rw[S_CLOUD + i];
    let vx = sample_vel(vec2f(base_x + drift, cy)).x;
    drift += vx * dt * 0.22 + 0.012 * dt;
    let span = E.domain_w * 0.08;
    if (drift > span) { drift = -span; }
    if (drift < -span) { drift = span; }
    state_rw[S_CLOUD + i] = drift;
    return;
  }

  if (g < OBJ_THREADS) { return; }
  let bi = g - OBJ_THREADS;
  if (bi >= E.n_blades || bi >= MAX_BLADES) { return; }
  let b = blade_def(bi);
  let vx = sample_vel(vec2f(b.x, 0.05)).x;
  let goal = clamp(vx * 18.0, -1.15, 1.15) + sin(time * 2.1 + b.phase) * 0.04;
  let step = min(1.0, E.grass_resp * dt);
  let k = S_BLADE + bi;
  state_rw[k] = state_rw[k] + (goal - state_rw[k]) * step;
}

// --- 2a. Backdrop: sky, ground strip, ground moisture ------------------------

struct BackOut {
  @builtin(position) pos : vec4f,
};

@vertex
fn vs_env_backdrop(@builtin(vertex_index) vi: u32) -> BackOut {
  var pts = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(3.0, 1.0), vec2f(-1.0, 1.0));
  var o : BackOut;
  o.pos = vec4f(pts[vi], 0.0, 1.0);
  return o;
}

// Premultiplied "source over" (what the 2D canvas did, layer by layer).
fn over(dst: vec4f, col: vec3f, a: f32) -> vec4f {
  return vec4f(col * a + dst.rgb * (1.0 - a), a + dst.a * (1.0 - a));
}

fn sky_color(y: f32, ground_y: f32) -> vec3f {
  let t = clamp(y / (ground_y + E.canvas_h * 0.04), 0.0, 1.0);
  let c0 = rgb(0x0f172au);
  let c1 = rgb(0x1e3a5fu);
  let c2 = rgb(0x334155u);
  if (t < 0.45) { return mix(c0, c1, t / 0.45); }
  if (t < 0.78) { return mix(c1, c2, (t - 0.45) / 0.33); }
  return c2;
}

@fragment
fn fs_env_backdrop(in: BackOut) -> @location(0) vec4f {
  let x = in.pos.x;
  let y = in.pos.y;
  let H = E.canvas_h;
  let strip = H * 0.09;
  let ground_y = H - strip;
  var c = vec4f(0.0);

  if (y < ground_y + H * 0.05) {
    c = over(c, sky_color(y, ground_y), E.sky_alpha);
  }

  if (y >= ground_y) {
    let a = 0.9;
    c = over(c, rgb(0x166534u), a);
    for (var i = 0; i < 5; i++) {
      let y0 = ground_y + 4.0 + f32(i) * (strip / 6.0);
      if (y >= y0 && y < y0 + 3.0) { c = over(c, rgb(0x14532du), a); }
    }
    if (y < ground_y + 1.0) { c = over(c, rgb(0x4ade80u), a * 0.18); }

    if (has(F_ACTIVE)) {
      let moisture_vis = has(F_WET) || has(F_MOISTURE);
      let any_ground = has(F_GRASS) || has(F_TREES) || has(F_HOUSES) || moisture_vis;
      var ga = 0.35;
      if (has(F_GRASS)) { ga = 0.55; }
      if (any_ground && moisture_vis) {
        // 96 flat columns, sampled at column centres like the canvas version.
        let cols = 96.0;
        let col = floor(x / E.canvas_w * cols);
        let wx = (col + 0.5) / cols * E.domain_w;
        let m = ground_moisture(wx);
        let stain = min(1.0, m.z * 1.05);
        let puddle = min(1.0, m.y * 1.1);
        let rgb8 = vec3f(
          round(34.0 - stain * 22.0 - puddle * 8.0 + m.x * 4.0),
          round(96.0 - stain * 48.0 - puddle * 28.0),
          round(42.0 + stain * 42.0 + puddle * 22.0),
        );
        c = over(c, clamp(rgb8, vec3f(0.0), vec3f(255.0)) / 255.0, ga);
        let st = (y - ground_y) / strip;
        var sheen : vec4f;
        if (st < 0.55) {
          sheen = mix(vec4f(90.0, 180.0, 210.0, 0.08 * 255.0), vec4f(30.0, 90.0, 120.0, 0.10 * 255.0), st / 0.55);
        } else {
          sheen = mix(vec4f(30.0, 90.0, 120.0, 0.10 * 255.0), vec4f(12.0, 40.0, 58.0, 0.18 * 255.0), (st - 0.55) / 0.45);
        }
        c = over(c, sheen.rgb / 255.0, sheen.a / 255.0);
      } else if (has(F_GRASS)) {
        c = over(c, vec3f(22.0, 83.0, 52.0) / 255.0, ga * 0.85);
      }
    }
  }

  if (c.a <= 0.0) { discard; }
  return vec4f(c.rgb / c.a, c.a);
}

// --- 2b. Shapes: ellipses, rings, rects, triangles, gradient ellipses --------

const K_NONE    : u32 = 0u;
const K_ELLIPSE : u32 = 1u;
const K_RING    : u32 = 2u;
const K_RECT    : u32 = 3u;
const K_TRI     : u32 = 4u;  // origin at base centre, apex at -y
const K_MIST    : u32 = 5u;

struct Shape {
  center : vec2f, // px
  half   : vec2f, // radii / half extents (px); for K_TRI: half base, height
  rot    : f32,
  kind   : u32,
  param  : f32,   // ring width (K_RING) or gradient radius (K_MIST)
  color  : vec4f,
};

fn no_shape() -> Shape {
  return Shape(vec2f(0.0), vec2f(0.0), 0.0, K_NONE, 0.0, vec4f(0.0));
}

fn ellipse(c: vec2f, r: vec2f, rot: f32, col: vec4f) -> Shape {
  return Shape(c, r, rot, K_ELLIPSE, 0.0, col);
}

// Canvas fillRect(x, y, w, h) with top-left origin.
fn rect_tl(x: f32, y: f32, w: f32, h: f32, col: vec4f) -> Shape {
  return Shape(vec2f(x + w * 0.5, y + h * 0.5), vec2f(w * 0.5, h * 0.5), 0.0, K_RECT, 0.0, col);
}

fn cloud_shape(ci: u32, k: u32) -> Shape {
  if (!has(F_CLOUDS)) { return no_shape(); }
  let cd = scene.clouds[ci];
  let wx = cd.x * E.domain_w + state[S_CLOUD + ci];
  let cy = cd.y * E.domain_h;
  let vy = sample_vel(vec2f(wx, cy)).y;
  let p = world_to_px(vec2f(wx, cy));
  let bob = sin(E.sim_time * 0.45 + cd.w) * 2.0 + vy * 1.5;
  let s = cd.z * px_per_m() * 0.028;
  let col = vec4f(226.0 / 255.0, 232.0 / 255.0, 240.0 / 255.0, 0.72);
  switch (k) {
    case 0u: { return ellipse(vec2f(p.x, p.y + bob), vec2f(26.0, 11.0) * s, 0.0, col); }
    case 1u: { return ellipse(vec2f(p.x - 18.0 * s, p.y - 1.0 + bob), vec2f(17.0, 12.0) * s, 0.0, col); }
    case 2u: { return ellipse(vec2f(p.x + 19.0 * s, p.y + 1.0 + bob), vec2f(15.0, 9.5) * s, 0.0, col); }
    default: { return ellipse(vec2f(p.x - 5.0 * s, p.y - 7.0 + bob), vec2f(12.0, 9.0) * s, 0.0, col); }
  }
}

fn mist_shape(i: u32) -> Shape {
  if (!has(F_MIST)) { return no_shape(); }
  let t = (f32(i) + 0.5) / f32(N_MIST);
  let wx = 0.2 + t * (E.domain_w - 0.4);
  let wy = 0.12 + sin(f32(i) * 1.3) * 0.04;
  let vx = sample_vel(vec2f(wx, wy)).x;
  var spray = 0.0;
  for (var e = 0u; e < E.water_count && e < MAX_EMITTERS; e++) {
    let d = distance(vec2f(wx, wy), E.water[e].xy);
    spray = max(spray, max(0.0, 1.0 - d / 3.5) * 0.35);
  }
  let gust = min(1.0, abs(vx) / 1.8) * 0.5;
  let alpha = min(0.42, spray + gust * 0.28);
  if (alpha < 0.06) { return no_shape(); }
  let strip_h = E.canvas_h * 0.14;
  let base_y = E.canvas_h - strip_h * 0.35;
  let px = world_to_px(vec2f(wx, 0.0)).x;
  let wobble = sin(E.sim_time * 1.6 + f32(i) * 0.9) * strip_h * 0.08;
  let drift = vx * px_per_m() * 8.0;
  let r = strip_h * (0.22 + f32(i % 3u) * 0.04);
  return Shape(
    vec2f(px + drift, base_y + wobble), vec2f(r * 1.35, r * 0.55), vx * 0.04,
    K_MIST, r, vec4f(0.0, 0.0, 0.0, alpha),
  );
}

fn house_shape(hi: u32, k: u32) -> Shape {
  if (!has(F_HOUSES)) { return no_shape(); }
  let hd = scene.houses[hi];
  let hp = house_px(hd);
  let hx = hp.x;
  let gy = hp.y;
  let w = hp.z;
  let h = hp.w;
  let chimney = hd.w > 0.5;
  let win_w = w * 0.11;
  let win_h = h * 0.14;
  let chim_x = hx + w * 0.22;
  let chim_base = gy - h - h * 0.42;
  switch (k) {
    case 0u: { return rect_tl(hx - w * 0.5, gy - h, w, h, vec4f(rgb(0x334155u), 1.0)); }
    case 1u: {
      return Shape(vec2f(hx, gy - h), vec2f(w * 0.5 + 5.0, h * 0.55), 0.0, K_TRI, 0.0, vec4f(rgb(0x1e2937u), 1.0));
    }
    case 2u: { return rect_tl(hx - w * 0.28, gy - h * 0.68, win_w, win_h, vec4f(rgb(0x64748bu), 1.0)); }
    case 3u: { return rect_tl(hx + w * 0.12, gy - h * 0.68, win_w, win_h, vec4f(rgb(0x64748bu), 1.0)); }
    case 4u: {
      if (w <= 55.0) { return no_shape(); }
      return rect_tl(hx - w * 0.28, gy - h * 0.42, win_w, win_h, vec4f(rgb(0x64748bu), 1.0));
    }
    case 5u: { return rect_tl(hx - w * 0.06, gy - h * 0.22, w * 0.12, h * 0.22, vec4f(rgb(0x1e2937u), 1.0)); }
    case 6u: {
      if (!chimney) { return no_shape(); }
      return rect_tl(chim_x, chim_base - h * 0.18, w * 0.08, h * 0.18, vec4f(rgb(0x475569u), 1.0));
    }
    default: {
      if (!chimney) { return no_shape(); }
      let i = k - 7u;
      let t = f32(i) * 0.5;
      let smoke_drift = state[S_FLAG + hi] * 0.35 + house_vx(hd) * px_per_m() * 0.2;
      let sy = chim_base - h * (0.22 + t * 0.2);
      let sx = chim_x + w * 0.04 + smoke_drift * (0.4 + t * 0.5);
      let wobble = sin(E.sim_time * 2.2 + f32(hi) * 1.7 + f32(i)) * 3.0;
      return Shape(
        vec2f(sx + wobble, sy), vec2f(4.0 + t * 3.0, 3.0 + t * 2.0), 0.0, K_RING, 1.2,
        vec4f(203.0 / 255.0, 213.0 / 255.0, 225.0 / 255.0, 0.55),
      );
    }
  }
}

struct TreePx {
  ground : vec2f,
  h_px   : f32,
  trunk  : f32,
  foliage: f32,
};

fn tree_px(ti: u32) -> TreePx {
  let td = scene.trees[ti];
  var o : TreePx;
  o.ground = world_to_px(vec2f(td.a.x * E.domain_w, 0.0));
  o.h_px = (td.a.y / E.domain_h) * E.canvas_h * td.a.z;
  o.trunk = state[S_TREE + ti * 2u];
  o.foliage = state[S_TREE + ti * 2u + 1u];
  return o;
}

fn tree_shape(ti: u32, k: u32) -> Shape {
  if (!has(F_TREES)) { return no_shape(); }
  let td = scene.trees[ti];
  let height = td.a.y;
  let depth = td.a.z;
  let phase = td.a.w;
  let tp = tree_px(ti);
  var base = vec4f(rgb(0x15803du), 1.0);
  if (depth > 0.85) { base = vec4f(rgb(0x166534u), 1.0); }
  var layers = 3u;
  if (height > 4.2) { layers = 5u; } else if (height > 3.2) { layers = 4u; }

  if (k == TREE_PRIMS - 1u) {
    let top_sway = tp.foliage * 1.18;
    return ellipse(
      vec2f(tp.ground.x + top_sway * 0.55, tp.ground.y - tp.h_px - 5.0 * depth),
      vec2f(10.0, 8.0) * depth, top_sway * 0.015, base,
    );
  }
  let i = k / 2u;
  if (i >= layers) { return no_shape(); }
  let t = f32(i) / f32(max(1u, layers - 1u));
  let ly = tp.ground.y - tp.h_px * (0.16 + t * 0.72);
  let layer_sway = tp.trunk * (1.0 - t) * 0.45 + tp.foliage * (0.55 + t * 0.55);
  let lx = tp.ground.x + layer_sway * (0.65 + t * 0.35);
  let r = (14.0 + (tp.h_px / 7.0) * (0.8 + sin(f32(i) + phase) * 0.12) - f32(i) * 1.2) * depth;
  if (k % 2u == 0u) {
    return ellipse(vec2f(lx, ly), vec2f(r * 0.92, r * (0.68 + t * 0.08)), layer_sway * 0.008, base);
  }
  return ellipse(vec2f(lx - r * 0.22, ly - r * 0.18), vec2f(r * 0.42, r * 0.3), 0.0, vec4f(74.0 / 255.0, 222.0 / 255.0, 128.0 / 255.0, 0.16));
}

fn shape_for(id: u32) -> Shape {
  if (id < SH_MIST) { return cloud_shape((id - SH_CLOUD) / 4u, (id - SH_CLOUD) % 4u); }
  if (id < SH_HOUSE) { return mist_shape(id - SH_MIST); }
  if (id < SH_TREE) { return house_shape((id - SH_HOUSE) / HOUSE_PRIMS, (id - SH_HOUSE) % HOUSE_PRIMS); }
  let k = id - SH_TREE;
  if (k < N_TREES * TREE_PRIMS) { return tree_shape(k / TREE_PRIMS, k % TREE_PRIMS); }
  return no_shape();
}

struct ShapeOut {
  @builtin(position) pos : vec4f,
  @location(0) local : vec2f,                       // px, shape frame
  @location(1) @interpolate(flat) half : vec2f,
  @location(2) @interpolate(flat) kp : vec2f,       // kind, param
  @location(3) @interpolate(flat) color : vec4f,
};

@vertex
fn vs_env_shapes(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> ShapeOut {
  let s = shape_for(ii);
  var o : ShapeOut;
  o.half = s.half;
  o.kp = vec2f(f32(s.kind), s.param);
  o.color = s.color;
  if (s.kind == K_NONE) {
    o.pos = vec4f(2.0, 2.0, 0.0, 1.0); // off-screen, zero area
    o.local = vec2f(0.0);
    return o;
  }
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  );
  let pad = 1.5 + s.param;
  var ext = s.half + vec2f(pad);
  var origin = vec2f(0.0);
  if (s.kind == K_TRI) {
    ext = vec2f(s.half.x, s.half.y * 0.5) + vec2f(pad);
    origin = vec2f(0.0, -s.half.y * 0.5);
  }
  let local = origin + corners[vi] * ext;
  let c = cos(s.rot);
  let sn = sin(s.rot);
  let world = s.center + vec2f(local.x * c - local.y * sn, local.x * sn + local.y * c);
  o.pos = px_to_clip(world);
  o.local = local;
  return o;
}

fn ellipse_sd(p: vec2f, r: vec2f) -> f32 {
  let rr = max(r, vec2f(1.0e-3));
  return (length(p / rr) - 1.0) * min(rr.x, rr.y);
}

@fragment
fn fs_env_shapes(in: ShapeOut) -> @location(0) vec4f {
  let kind = u32(in.kp.x + 0.5);
  let p = in.local;
  let r = in.half;
  var cov = 0.0;
  var col = in.color;
  switch (kind) {
    case K_ELLIPSE: { cov = clamp(0.5 - ellipse_sd(p, r), 0.0, 1.0); }
    case K_RING: { cov = clamp(0.5 - (abs(ellipse_sd(p, r)) - in.kp.y * 0.5), 0.0, 1.0); }
    case K_RECT: {
      let d = abs(p) - r;
      cov = clamp(0.5 - max(d.x, d.y), 0.0, 1.0);
    }
    case K_TRI: {
      let hw = r.x;
      let ht = r.y;
      let d_edge = (abs(p.x) * ht - p.y * hw - hw * ht) / sqrt(ht * ht + hw * hw);
      cov = clamp(0.5 - max(p.y, d_edge), 0.0, 1.0);
    }
    case K_MIST: {
      cov = clamp(0.5 - ellipse_sd(p, r), 0.0, 1.0);
      let g = length(p) / max(in.kp.y, 1.0e-3);
      let a = in.color.a;
      if (g < 0.55) {
        col = mix(vec4f(140.0, 200.0, 230.0, a * 255.0), vec4f(100.0, 170.0, 210.0, a * 0.45 * 255.0), g / 0.55) / 255.0;
      } else {
        col = mix(vec4f(100.0, 170.0, 210.0, a * 0.45 * 255.0), vec4f(80.0, 140.0, 180.0, 0.0), min(1.0, (g - 0.55) / 0.45)) / 255.0;
      }
    }
    default: { }
  }
  let alpha = col.a * cov;
  if (alpha <= 0.0) { discard; }
  return vec4f(col.rgb, alpha);
}

// --- 2c. Strokes: tapered quadratic Bezier ribbons ---------------------------

struct Stroke {
  p0 : vec2f,
  p1 : vec2f, // control point
  p2 : vec2f,
  w0 : f32,   // width at p0 (px)
  w1 : f32,   // width at p2 (px)
  color : vec4f,
};

fn no_stroke() -> Stroke {
  return Stroke(vec2f(0.0), vec2f(0.0), vec2f(0.0), 0.0, 0.0, vec4f(0.0));
}

fn trunk_stroke(ti: u32) -> Stroke {
  if (!has(F_TREES)) { return no_stroke(); }
  let depth = scene.trees[ti].a.z;
  let tp = tree_px(ti);
  let top_sway = tp.foliage * 1.18;
  let mid_sway = tp.trunk * 0.72 + tp.foliage * 0.35;
  var w = 4.0;
  if (depth > 0.85) { w = 7.0; } else if (depth > 0.7) { w = 5.5; }
  w *= depth;
  let g = tp.ground;
  return Stroke(
    g,
    vec2f(g.x + mid_sway * 0.38, g.y - tp.h_px * 0.52),
    vec2f(g.x + top_sway * 0.52, g.y - tp.h_px),
    w, w * 0.8, vec4f(rgb(0x3f2a1fu), 1.0),
  );
}

fn flag_stroke(hi: u32) -> Stroke {
  if (!has(F_HOUSES)) { return no_stroke(); }
  let hd = scene.houses[hi];
  if (hd.w > 0.5) { return no_stroke(); } // chimney houses have smoke instead
  let hp = house_px(hd);
  let fx = hp.x + hp.z * 0.5 + 4.0;
  let fy = hp.y - hp.w - hp.w * 0.35;
  let fs = state[S_FLAG + hi];
  return Stroke(
    vec2f(fx, fy),
    vec2f(fx + 12.0 + fs * 0.8, fy - 2.0 + fs * 0.2),
    vec2f(fx + 22.0 + fs * 1.6, fy + 4.0 + fs * 0.6),
    1.5, 1.5, vec4f(rgb(0xe2e8f0u), 1.0),
  );
}

fn blade_stroke(bi: u32) -> Stroke {
  if (!has(F_GRASS) || bi >= E.n_blades || bi >= MAX_BLADES) { return no_stroke(); }
  let b = blade_def(bi);
  let angle = state[S_BLADE + bi];
  let base = world_to_px(vec2f(b.x, 0.0));
  let len = b.height * E.canvas_h;
  let bend = sin(angle) * len;
  return Stroke(
    base,
    vec2f(base.x + bend * 0.45, base.y - len * 0.55),
    vec2f(base.x + bend, base.y - len),
    b.thickness, b.thickness * 0.4, vec4f(rgb(0x4ade80u), 1.0),
  );
}

fn stroke_for(id: u32) -> Stroke {
  if (id < ST_FLAG) { return trunk_stroke(id - ST_TRUNK); }
  if (id < ST_BLADE) { return flag_stroke(id - ST_FLAG); }
  return blade_stroke(id - ST_BLADE);
}

struct StrokeOut {
  @builtin(position) pos : vec4f,
  @location(0) across : f32,                    // px from centreline
  @location(1) half_w : f32,
  @location(2) @interpolate(flat) color : vec4f,
};

// Triangle strip: 2 * (STROKE_SEGS + 1) vertices per instance.
@vertex
fn vs_env_strokes(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> StrokeOut {
  let s = stroke_for(ii);
  var o : StrokeOut;
  o.color = s.color;
  if (s.w0 <= 0.0) {
    o.pos = vec4f(2.0, 2.0, 0.0, 1.0);
    o.across = 0.0;
    o.half_w = 0.0;
    return o;
  }
  let t = f32(vi / 2u) / f32(STROKE_SEGS);
  let side = f32(vi % 2u) * 2.0 - 1.0;
  let u = 1.0 - t;
  let pos = u * u * s.p0 + 2.0 * u * t * s.p1 + t * t * s.p2;
  var tan = 2.0 * u * (s.p1 - s.p0) + 2.0 * t * (s.p2 - s.p1);
  if (dot(tan, tan) < 1.0e-8) { tan = vec2f(0.0, -1.0); }
  let n = normalize(vec2f(-tan.y, tan.x));
  let half_w = mix(s.w0, s.w1, t) * 0.5;
  let ext = half_w + 1.0; // 1 px for AA
  o.pos = px_to_clip(pos + n * side * ext);
  o.across = side * ext;
  o.half_w = half_w;
  return o;
}

@fragment
fn fs_env_strokes(in: StrokeOut) -> @location(0) vec4f {
  // Thin strokes (< 1 px) keep their area as reduced alpha, like canvas lines.
  let cov = clamp(in.half_w + 0.5 - abs(in.across), 0.0, 1.0) * min(1.0, in.half_w * 2.0 + 0.25);
  let a = in.color.a * cov;
  if (a <= 0.0) { discard; }
  return vec4f(in.color.rgb, a);
}
