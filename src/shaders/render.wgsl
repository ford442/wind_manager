struct RParams {
  nx          : u32,
  ny          : u32,
  h           : f32,
  mode        : u32,
  t_amb       : f32,
  q_amb       : f32,
  canvas_w    : f32,
  canvas_h    : f32,
  show_wet    : f32,
  time        : f32,
  max_tracers : u32,
  tracer_life : f32,
  show_tracers: f32,
  show_streaks: f32,
};

struct EmitterVis {
  pos     : vec2f,
  angle   : f32,
  spread  : f32,
  selected: f32,
  typ     : f32,
  _pad0   : f32,
  _pad1   : f32,
};

@group(0) @binding(0) var<uniform> R : RParams;
@group(0) @binding(1) var<storage, read> vel : array<vec2f>;
@group(0) @binding(2) var<storage, read> tf : array<f32>;
@group(0) @binding(3) var<storage, read> qf : array<f32>;
@group(0) @binding(4) var<storage, read> drops : array<Droplet>;
@group(0) @binding(5) var<storage, read> emitters : array<EmitterVis>;
@group(0) @binding(6) var<storage, read> wet : array<f32>;
@group(0) @binding(7) var<storage, read> tracers : array<Tracer>;

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

fn sample_wet(world_x: f32, nx: i32, h: f32) -> f32 {
  let gx = clamp(world_x / h - 0.5, 0.0, f32(nx - 1) - 1.0e-4);
  let i0 = i32(floor(gx));
  let i1 = min(i0 + 1, nx - 1);
  let fx = gx - f32(i0);
  return mix(wet[i0], wet[i1], fx);
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

  if (R.show_wet > 0.5) {
    let world_y = in.uv.y * f32(ny) * R.h;
    if (world_y < R.h * 2.2) {
      let wv = sample_wet(world.x, nx, R.h);
      let band = smoothstep(R.h * 2.2, 0.0, world_y);
      let wet_col = vec3f(0.04, 0.10, 0.20);
      col = mix(col, col * 0.62 + wet_col, wv * band * 0.82);
      let sparkle = sin(world.x * 48.0 + R.time * 9.0) * sin(world.x * 31.0 - R.time * 6.0);
      let speck = max(0.0, sparkle) * wv * band * 0.18;
      col += vec3f(0.35, 0.55, 0.75) * speck;
    }
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

struct EmitterOut {
  @builtin(position) pos : vec4f,
  @location(0) col : vec4f,
};

@vertex
fn vs_emitter(@builtin(vertex_index) vi: u32,
              @builtin(instance_index) ii: u32) -> EmitterOut {
  var o : EmitterOut;
  let em = emitters[ii];
  let ex = em.pos.x;
  let ey = em.pos.y;
  let ang = em.angle;
  let spr = em.spread;
  let is_air = em.typ > 0.5;
  let sel = em.selected > 0.5;
  let dir = vec2f(sin(ang), cos(ang));
  let perp = vec2f(-dir.y, dir.x);
  let center = vec2f(ex, ey);

  let glow_r = 0.24;
  let core_r = 0.10;
  let cone_len = 0.58;
  let shaft_len = 0.42;
  let shaft_hw = 0.018;
  let line_hw = 0.008;

  let water_glow = vec3f(0.22, 0.74, 0.97);
  let air_glow   = vec3f(0.20, 0.82, 0.55);
  let base_rgb = select(water_glow, air_glow, is_air);
  let sel_boost = select(0.0, 0.18, sel);

  var p = center;
  var col = vec4f(0.0);

  if (vi < 24u) {
    let seg = vi / 3u;
    let corner = vi % 3u;
    let a0 = f32(seg) * PI * 2.0 / 8.0 - PI * 0.5;
    let a1 = f32(seg + 1u) * PI * 2.0 / 8.0 - PI * 0.5;
    if (corner == 0u) { p = center; }
    else if (corner == 1u) { p = center + vec2f(cos(a0), sin(a0)) * glow_r; }
    else { p = center + vec2f(cos(a1), sin(a1)) * glow_r; }
    col = vec4f(base_rgb, 0.22 + sel_boost);
  } else if (vi < 48u) {
    let vi2 = vi - 24u;
    let seg = vi2 / 3u;
    let corner = vi2 % 3u;
    let a0 = f32(seg) * PI * 2.0 / 8.0;
    let a1 = f32(seg + 1u) * PI * 2.0 / 8.0;
    if (corner == 0u) { p = center; }
    else if (corner == 1u) { p = center + vec2f(cos(a0), sin(a0)) * core_r; }
    else { p = center + vec2f(cos(a1), sin(a1)) * core_r; }
    col = vec4f(base_rgb + vec3f(sel_boost), 0.95);
  } else if (is_air) {
    let viA = vi - 48u;
    let arrow_idx = viA / 12u;
    let corner = viA % 12u;
    let stream_len = 0.68;
    let stream_hw = 0.013;

    if (arrow_idx < 3u) {
      var spread_dir = -spr;
      if (arrow_idx == 1u) { spread_dir = 0.0; }
      if (arrow_idx == 2u) { spread_dir = spr; }
      let ad = dir * cos(spread_dir) + perp * sin(spread_dir);
      let an = vec2f(-ad.y, ad.x);
      let tip = center + ad * stream_len;
      let head = center + ad * (stream_len + 0.14);
      let back = tip - ad * 0.1;
      if (corner < 6u) {
        switch (corner) {
          case 0u: { p = center + an * stream_hw; }
          case 1u: { p = center - an * stream_hw; }
          case 2u: { p = tip - an * stream_hw; }
          case 3u: { p = center + an * stream_hw; }
          case 4u: { p = tip + an * stream_hw; }
          default: { p = tip - an * stream_hw; }
        }
        col = vec4f(mix(base_rgb, vec3f(0.95), 0.45), 0.78);
      } else {
        let c2 = corner - 6u;
        if (c2 == 0u) { p = tip; }
        else if (c2 == 1u) { p = head; }
        else if (c2 == 2u) { p = back + an * 0.06; }
        else if (c2 == 3u) { p = tip; }
        else if (c2 == 4u) { p = back - an * 0.06; }
        else { p = head; }
        col = vec4f(0.92, 1.0, 0.95, select(0.88, 0.98, sel));
      }
    } else {
      let handle_c = center + dir * (stream_len + 0.14);
      let handle_r = 0.065;
      let seg = (viA - 36u) / 3u;
      let hc = (viA - 36u) % 3u;
      let a0 = f32(seg) * PI * 2.0 / 3.0;
      let a1 = f32(seg + 1u) * PI * 2.0 / 3.0;
      if (hc == 0u) { p = handle_c; }
      else if (hc == 1u) { p = handle_c + vec2f(cos(a0), sin(a0)) * handle_r; }
      else { p = handle_c + vec2f(cos(a1), sin(a1)) * handle_r; }
      col = vec4f(0.92, 1.0, 0.95, select(0.7, 0.9, sel));
    }
  } else if (vi < 51u) {
    let left = dir * cos(spr) + perp * sin(spr);
    let right = dir * cos(spr) - perp * sin(spr);
    switch (vi - 48u) {
      case 0u: { p = center; }
      case 1u: { p = center + left * cone_len; }
      default: { p = center + right * cone_len; }
    }
    col = vec4f(mix(base_rgb, vec3f(1.0), 0.35), 0.14 + sel_boost * 0.5);
  } else if (vi < 69u) {
    let vi3 = vi - 51u;
    let part = vi3 / 6u;
    let corner = vi3 % 6u;
    if (part < 2u) {
      let spread_dir = select(spr, -spr, part == 1u);
      let edge_dir = dir * cos(spread_dir) + perp * sin(spread_dir);
      let en = vec2f(-edge_dir.y, edge_dir.x);
      let tip = center + edge_dir * cone_len;
      switch (corner) {
        case 0u: { p = center + en * line_hw; }
        case 1u: { p = center - en * line_hw; }
        case 2u: { p = tip - en * line_hw; }
        case 3u: { p = center + en * line_hw; }
        case 4u: { p = tip + en * line_hw; }
        default: { p = tip - en * line_hw; }
      }
      col = vec4f(mix(base_rgb, vec3f(0.95), 0.4), 0.55);
    } else {
      let tip = center + dir * shaft_len;
      switch (corner) {
        case 0u: { p = center + perp * shaft_hw; }
        case 1u: { p = center - perp * shaft_hw; }
        case 2u: { p = tip - perp * shaft_hw; }
        case 3u: { p = center + perp * shaft_hw; }
        case 4u: { p = tip + perp * shaft_hw; }
        default: { p = tip - perp * shaft_hw; }
      }
      col = vec4f(mix(base_rgb, vec3f(0.95), 0.5), 0.85);
    }
  } else if (vi < 75u) {
    let tip = center + dir * shaft_len;
    let head = center + dir * (shaft_len + 0.12);
    let back = tip - dir * 0.09;
    let tri = vi - 69u;
    if (tri == 0u) { p = tip; }
    else if (tri == 1u) { p = head; }
    else if (tri == 2u) { p = back + perp * 0.07; }
    else if (tri == 3u) { p = tip; }
    else if (tri == 4u) { p = back - perp * 0.07; }
    else { p = head; }
    col = vec4f(0.95, 0.98, 1.0, select(0.82, 0.95, sel));
  } else {
    let vi4 = vi - 75u;
    let handle_c = center + dir * (shaft_len + 0.12);
    let handle_r = 0.07;
    let seg = vi4 / 3u;
    let corner = vi4 % 3u;
    let a0 = f32(seg) * PI * 2.0 / 6.0;
    let a1 = f32(seg + 1u) * PI * 2.0 / 6.0;
    if (corner == 0u) { p = handle_c; }
    else if (corner == 1u) { p = handle_c + vec2f(cos(a0), sin(a0)) * handle_r; }
    else { p = handle_c + vec2f(cos(a1), sin(a1)) * handle_r; }
    col = vec4f(0.95, 0.98, 1.0, select(0.75, 0.92, sel));
  }

  o.pos = vec4f(world_to_clip(p), 0.0, 1.0);
  o.col = col;
  return o;
}

@fragment
fn fs_emitter(in: EmitterOut) -> @location(0) vec4f {
  return in.col;
}

struct TracerOut {
  @builtin(position) pos : vec4f,
  @location(0) col : vec4f,
};

fn tracer_speed_col(sp: f32, tint: f32) -> vec3f {
  let t = clamp(sp / 4.0, 0.0, 1.0);
  let cool = vec3f(0.72, 0.88, 1.0);
  let fast = vec3f(1.0, 0.95, 0.72);
  let base = mix(cool, fast, t);
  let hue = vec3f(
    0.55 + tint * 0.25,
    0.78 + tint * 0.12,
    0.98 - tint * 0.2,
  );
  return mix(hue, base, 0.55);
}

fn tracer_fade(age: f32) -> f32 {
  let t = clamp(age / max(R.tracer_life, 0.1), 0.0, 1.0);
  let fade_in = smoothstep(0.0, 0.06, t);
  let fade_out = 1.0 - smoothstep(0.72, 1.0, t);
  return fade_in * fade_out;
}

@vertex
fn vs_tracer_streaks(@builtin(vertex_index) vi: u32) -> TracerOut {
  let ti = vi / 2u;
  let endpt = vi % 2u;
  var o : TracerOut;
  o.pos = vec4f(0.0, 0.0, 2.0, 1.0);
  o.col = vec4f(0.0);
  if (ti >= R.max_tracers) { return o; }

  let tr = tracers[ti];
  if (tr.alive == 0u) { return o; }

  let nx = i32(R.nx);
  let ny = i32(R.ny);
  let sp = length(sample_vel(tr.pos, R.h, nx, ny, &vel));
  let fade = tracer_fade(tr.age);
  let p = select(tr.prev_pos, tr.pos, endpt == 1u);
  o.pos = vec4f(world_to_clip(p), 0.0, 1.0);
  let alpha = select(0.08, 0.55, endpt == 1u) * fade * (0.45 + 0.55 * clamp(sp / 3.0, 0.0, 1.0));
  o.col = vec4f(tracer_speed_col(sp, tr.tint), alpha);
  return o;
}

@vertex
fn vs_tracer_dots(@builtin(vertex_index) vi: u32) -> TracerOut {
  let ti = vi / 6u;
  let corner = vi % 6u;
  var o : TracerOut;
  o.pos = vec4f(0.0, 0.0, 2.0, 1.0);
  o.col = vec4f(0.0);
  if (ti >= R.max_tracers) { return o; }

  let tr = tracers[ti];
  if (tr.alive == 0u) { return o; }

  let nx = i32(R.nx);
  let ny = i32(R.ny);
  let sp = length(sample_vel(tr.pos, R.h, nx, ny, &vel));
  let fade = tracer_fade(tr.age);
  let px = 0.0022 + clamp(sp * 0.00035, 0.0, 0.004);
  let c = world_to_clip(tr.pos);
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
  o.pos = vec4f(c + offset / dom * 2.0, 0.0, 1.0);
  o.col = vec4f(tracer_speed_col(sp, tr.tint), 0.35 + fade * 0.5);
  return o;
}

@fragment
fn fs_tracers(in: TracerOut) -> @location(0) vec4f {
  return in.col;
}
