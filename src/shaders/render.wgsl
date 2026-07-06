struct RParams {
  nx        : u32,
  ny        : u32,
  h         : f32,
  mode      : u32,
  t_amb     : f32,
  q_amb     : f32,
  canvas_w  : f32,
  canvas_h  : f32,
};

@group(0) @binding(0) var<uniform> R : RParams;
@group(0) @binding(1) var<storage, read> vel : array<vec2f>;
@group(0) @binding(2) var<storage, read> tf : array<f32>;
@group(0) @binding(3) var<storage, read> qf : array<f32>;
@group(0) @binding(4) var<storage, read> drops : array<Droplet>;

fn world_to_clip(p: vec2f) -> vec2f {
  let dom = vec2f(f32(R.nx), f32(R.ny)) * R.h;
  return p / dom * 2.0 - 1.0;
}

struct FSOut {
  @builtin(position) pos : vec4f,
  @location(0) uv : vec2f,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> FSOut {
  var pts = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(3.0, 1.0), vec2f(-1.0, 1.0));
  var o : FSOut;
  o.pos = vec4f(pts[vi], 0.0, 1.0);
  o.uv = pts[vi] * 0.5 + 0.5;
  return o;
}

fn ramp_speed(t: f32) -> vec3f {
  let c0 = vec3f(0.045, 0.06, 0.10);
  let c1 = vec3f(0.05, 0.42, 0.52);
  let c2 = vec3f(0.95, 0.92, 0.65);
  if (t < 0.5) { return mix(c0, c1, t * 2.0); }
  return mix(c1, c2, (t - 0.5) * 2.0);
}

fn ramp_temp(t: f32) -> vec3f {
  let cold = vec3f(0.20, 0.50, 0.98);
  let mid  = vec3f(0.055, 0.07, 0.11);
  let warm = vec3f(0.95, 0.42, 0.13);
  if (t < 0.0) { return mix(mid, cold, -t); }
  return mix(mid, warm, t);
}

fn ramp_rh(t: f32) -> vec3f {
  let c0 = vec3f(0.05, 0.06, 0.10);
  let c1 = vec3f(0.10, 0.35, 0.45);
  let c2 = vec3f(0.75, 0.95, 1.0);
  if (t < 0.5) { return mix(c0, c1, t * 2.0); }
  return mix(c1, c2, (t - 0.5) * 2.0);
}

@fragment
fn fs_overlay(in: FSOut) -> @location(0) vec4f {
  let nx = i32(R.nx);
  let ny = i32(R.ny);
  let world = in.uv * vec2f(f32(nx), f32(R.ny)) * R.h;
  let b = bilin_at(world, R.h, nx, ny);
  let k00 = cell_index(b.i0, b.j0, nx);
  let k10 = cell_index(b.i1, b.j0, nx);
  let k01 = cell_index(b.i0, b.j1, nx);
  let k11 = cell_index(b.i1, b.j1, nx);

  var col = vec3f(0.045, 0.055, 0.085);
  if (R.mode == 1u) {
    let v = mix(mix(vel[k00], vel[k10], b.fx), mix(vel[k01], vel[k11], b.fx), b.fy);
    col = ramp_speed(clamp(length(v) / 3.0, 0.0, 1.0));
  } else if (R.mode == 2u) {
    let t = mix(mix(tf[k00], tf[k10], b.fx), mix(tf[k01], tf[k11], b.fx), b.fy);
    col = ramp_temp(clamp((t - R.t_amb) / 3.0, -1.0, 1.0));
  } else if (R.mode == 3u) {
    let t = mix(mix(tf[k00], tf[k10], b.fx), mix(tf[k01], tf[k11], b.fx), b.fy);
    let q = mix(mix(qf[k00], qf[k10], b.fx), mix(qf[k01], qf[k11], b.fx), b.fy);
    let rh = clamp(q / max(q_sat(t), 1.0e-6), 0.0, 1.0);
    col = ramp_rh(rh);
  }
  return vec4f(col, 1.0);
}

const ARROW_STRIDE : u32 = 8u;

@vertex
fn vs_arrows(@builtin(vertex_index) vi: u32,
             @builtin(instance_index) ii: u32) -> @builtin(position) vec4f {
  let gw = R.nx / ARROW_STRIDE;
  let ci = (ii % gw) * ARROW_STRIDE + ARROW_STRIDE / 2u;
  let cj = (ii / gw) * ARROW_STRIDE + ARROW_STRIDE / 2u;
  let k = i32(cj * R.nx + ci);
  let v = vel[k];
  let sp = length(v);
  let base = (vec2f(f32(ci), f32(cj)) + 0.5) * R.h;
  if (sp < 0.03) { return vec4f(0.0, 0.0, 2.0, 1.0); }

  let dir = v / sp;
  let len = min(0.06 + sp * 0.11, 0.42);
  let tip = base + dir * len;
  let perp = vec2f(-dir.y, dir.x);
  let hl = dir * len * 0.30;
  let hw = perp * len * 0.18;

  var p = base;
  switch (vi) {
    case 0u: { p = base; }
    case 1u: { p = tip; }
    case 2u: { p = tip; }
    case 3u: { p = tip - hl + hw; }
    case 4u: { p = tip; }
    default: { p = tip - hl - hw; }
  }
  return vec4f(world_to_clip(p), 0.0, 1.0);
}

@fragment
fn fs_arrows() -> @location(0) vec4f {
  return vec4f(0.92, 0.96, 1.0, 0.55);
}

struct DropOut {
  @builtin(position) pos : vec4f,
  @location(0) col : vec4f,
};

@vertex
fn vs_drops(@builtin(vertex_index) vi: u32) -> DropOut {
  let di = vi / 6u;
  let corner = vi % 6u;
  var o : DropOut;
  o.pos = vec4f(0.0, 0.0, 2.0, 1.0);
  o.col = vec4f(0.0);
  if (di >= arrayLength(&drops)) { return o; }
  let d = drops[di];
  if (d.alive == 0u) { return o; }

  let px = 0.003 + d.radius * 1.8;
  let c = world_to_clip(d.pos);
  var offset = vec2f(0.0);
  switch (corner) {
    case 0u: { offset = vec2f(-px, -px); }
    case 1u: { offset = vec2f( px, -px); }
    case 2u: { offset = vec2f( px,  px); }
    case 3u: { offset = vec2f(-px, -px); }
    case 4u: { offset = vec2f( px,  px); }
    default: { offset = vec2f(-px,  px); }
  }
  let dom = vec2f(f32(R.nx), f32(R.ny)) * R.h;
  let clipOff = offset / dom * 2.0;
  o.pos = vec4f(c + clipOff, 0.0, 1.0);
  let alpha = clamp(0.35 + d.radius * 800.0, 0.25, 0.85);
  o.col = vec4f(0.55, 0.82, 0.98, alpha);
  return o;
}

@fragment
fn fs_drops(in: DropOut) -> @location(0) vec4f {
  return in.col;
}
