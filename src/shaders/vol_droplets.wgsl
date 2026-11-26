// vol_droplets.wgsl — emit, droplet update and air injection for the yard volume.
// The slice version is droplets.wgsl; the physics per drop is the same (d² law,
// latent ΔT, semi-implicit Stokes drag, ground and wall contact), with z added.

@group(0) @binding(0) var<uniform> P : VParams;
@group(0) @binding(1) var<storage, read_write> drops : array<VDroplet>;
@group(0) @binding(2) var<storage, read_write> emit_counter : atomic<u32>;
@group(0) @binding(3) var<storage, read> vel : array<vec4f>;
@group(0) @binding(4) var<storage, read> tf : array<f32>;
@group(0) @binding(5) var<storage, read> qf : array<f32>;
// [q, T] per cell, fixed point.
@group(0) @binding(6) var<storage, read_write> acc_qt : array<atomic<i32>>;
// Eight storage buffers in `update`, the WebGPU default per-stage limit.
@group(0) @binding(7) var<storage, read> solid : array<u32>;
// [vx, vy, vz] per cell, fixed point.
@group(0) @binding(8) var<storage, read_write> acc_m : array<atomic<i32>>;
// Ground wetness, one per floor cell: k * nx + i.
@group(0) @binding(9) var<storage, read_write> acc_wet : array<atomic<i32>>;

// Per-emitter parameters for this substep. Layout: VOL_EMITTER_LAYOUT in
// src/volume/uniforms.ts (64-byte stride). An emitter is a disk of radius
// `size` in the x-z plane at `pos`, spraying a cone of half-angle `spread`
// around the direction tilted `angle` from vertical toward azimuth `azimuth`
// (0 = +x, pi/2 = +z).
struct VEmitter {
  pos     : vec3f,
  size    : f32,
  angle   : f32,
  azimuth : f32,
  spread  : f32,
  speed   : f32,
  count   : u32,
  kind    : u32,
  r_min   : f32,
  r_max   : f32,
  seed    : u32,
  // Real drops per parcel: one computational drop stands for `mult` drops of
  // the same size, so a hose's water flow is not bound by the droplet cap.
  mult    : f32,
  _pad1   : u32,
  _pad2   : u32,
};
@group(0) @binding(10) var<uniform> E : array<VEmitter, MAX_EMITTERS>;

fn dims() -> vec3i { return vec3i(i32(P.nx), i32(P.ny), i32(P.nz)); }

fn fixed_pt(v: f32) -> i32 {
  return i32(clamp(round(v), -2.0e9, 2.0e9));
}

fn is_solid_cell(c: vec3i) -> bool {
  let n = dims();
  if (!vinside(c.x, c.y, c.z, n)) { return false; }
  return solid[vcell(c.x, c.y, c.z, n)] != 0u;
}

fn solid_at(pos: vec3f) -> bool {
  return is_solid_cell(vec3i(floor(pos / P.h)));
}

// Trilinear weights with solid corners dropped and the rest renormalized.
fn fluid_tri(t: Tri) -> array<f32, 8> {
  let n = dims();
  var w : array<f32, 8>;
  var s = 0.0;
  var cnt = 0.0;
  for (var c = 0u; c < 8u; c++) {
    let fluid = solid[tri_corner(t, c, n)] == 0u;
    w[c] = select(0.0, tri_weight(t, c), fluid);
    s += w[c];
    cnt += select(0.0, 1.0, fluid);
  }
  for (var c = 0u; c < 8u; c++) {
    if (s > 1.0e-6) {
      w[c] = w[c] / s;
    } else if (cnt > 0.0) {
      // Only near-zero-weight corners are fluid: split evenly among them.
      w[c] = select(0.0, 1.0 / cnt, solid[tri_corner(t, c, n)] == 0u);
    }
  }
  return w;
}

fn deposit(k: i32, w: f32, dq: f32, dtemp: f32, dvel: vec3f) {
  if (w <= 0.0) { return; }
  atomicAdd(&acc_qt[2 * k], fixed_pt(w * dq * SCALE_Q));
  atomicAdd(&acc_qt[2 * k + 1], fixed_pt(w * dtemp * SCALE_T));
  atomicAdd(&acc_m[3 * k], fixed_pt(w * dvel.x * SCALE_M));
  atomicAdd(&acc_m[3 * k + 1], fixed_pt(w * dvel.y * SCALE_M));
  atomicAdd(&acc_m[3 * k + 2], fixed_pt(w * dvel.z * SCALE_M));
}

fn deposit_fluid(pos: vec3f, dq: f32, dtemp: f32, dvel: vec3f) {
  let n = dims();
  let t = tri_at(pos, P.h, n);
  let w = fluid_tri(t);
  for (var c = 0u; c < 8u; c++) {
    deposit(tri_corner(t, c, n), w[c], dq, dtemp, dvel);
  }
}

// Momentum only (air sources): touches acc_m, never acc_qt.
fn deposit_m_fluid(pos: vec3f, dvel: vec3f) {
  let n = dims();
  let t = tri_at(pos, P.h, n);
  let w = fluid_tri(t);
  for (var c = 0u; c < 8u; c++) {
    let k = tri_corner(t, c, n);
    if (w[c] <= 0.0) { continue; }
    atomicAdd(&acc_m[3 * k], fixed_pt(w[c] * dvel.x * SCALE_M));
    atomicAdd(&acc_m[3 * k + 1], fixed_pt(w[c] * dvel.y * SCALE_M));
    atomicAdd(&acc_m[3 * k + 2], fixed_pt(w[c] * dvel.z * SCALE_M));
  }
}

// Ground impact at (x, z): splash momentum into the bottom row and wetness on
// the floor cells, bilinear in x-z. The water evaporates later from the wet
// ground (vol_apply_wet), limited by the air's saturation deficit; the slice
// instead turns the whole drop into vapor and 12 % of its latent heat on
// impact, which a volume hose's small wet patch cools without bound.
fn deposit_ground(pos: vec3f, splash: vec3f, wet_amt: f32) {
  let n = dims();
  let g = clamp(pos.xz / P.h.xz - 0.5, vec2f(0.0), vec2f(f32(n.x - 1), f32(n.z - 1)) - 1.0e-4);
  let i0 = vec2i(floor(g));
  let f = g - vec2f(i0);
  for (var c = 0u; c < 4u; c++) {
    let ox = i32(c & 1u);
    let oz = i32(c >> 1u);
    let i = min(i0.x + ox, n.x - 1);
    let k = min(i0.y + oz, n.z - 1);
    let w = select(1.0 - f.x, f.x, ox == 1) * select(1.0 - f.y, f.y, oz == 1);
    if (w <= 0.0) { continue; }
    if (solid[vcell(i, 0, k, n)] == 0u) {
      deposit(vcell(i, 0, k, n), w, 0.0, 0.0, splash);
    }
    atomicAdd(&acc_wet[k * n.x + i], fixed_pt(w * wet_amt * SCALE_WET));
  }
}

// Unit vector in a cone of half-angle `spread` around the emitter direction.
fn cone_dir(em: VEmitter, seed: ptr<function, u32>) -> vec3f {
  let axis = vec3f(sin(em.angle) * cos(em.azimuth), cos(em.angle), sin(em.angle) * sin(em.azimuth));
  let ct = mix(cos(em.spread), 1.0, rand01(seed));
  let st = sqrt(max(0.0, 1.0 - ct * ct));
  let phi = 2.0 * PI * rand01(seed);
  var ref_v = vec3f(1.0, 0.0, 0.0);
  if (abs(axis.x) > 0.9) { ref_v = vec3f(0.0, 0.0, 1.0); }
  let u = normalize(cross(axis, ref_v));
  let v = cross(axis, u);
  return normalize(axis * ct + (u * cos(phi) + v * sin(phi)) * st);
}

// A point on the nozzle disk (radius `size`, horizontal), uniform by area.
fn disk_point(em: VEmitter, seed: ptr<function, u32>) -> vec3f {
  let r = em.size * sqrt(rand01(seed));
  let a = 2.0 * PI * rand01(seed);
  return em.pos + vec3f(r * cos(a), (rand01(seed) - 0.5) * 0.04, r * sin(a));
}

@compute @workgroup_size(64)
fn vol_emit(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.y >= P.emitter_count) { return; }
  let em = E[gid.y];
  if (em.kind != 0u || gid.x >= em.count) { return; }
  var seed = pcg(em.seed ^ (gid.x * 0x9E3779B9u + 1u));

  let slot = atomicAdd(&emit_counter, 1u) % P.max_droplets;
  let dir = cone_dir(em, &seed);
  let speed = em.speed * (0.8 + 0.4 * rand01(&seed));
  let r = em.r_min * pow(em.r_max / em.r_min, rand01(&seed));

  var d : VDroplet;
  d.pos_r = vec4f(disk_point(em, &seed), r);
  d.vel_a = vec4f(dir * speed, max(em.mult, 1.0e-3));
  drops[slot] = d;
}

@compute @workgroup_size(128)
fn vol_update(@builtin(global_invocation_id) gid: vec3u) {
  let di = gid.x;
  if (di >= P.max_droplets) { return; }
  var d = drops[di];
  // vel_a.w: parcel multiplicity, 0 when dead.
  let mult = d.vel_a.w;
  if (mult <= 0.0) { return; }

  let n = dims();
  let pos = d.pos_r.xyz;
  let vd = d.vel_a.xyz;
  let radius = d.pos_r.w;

  let t = tri_at(pos, P.h, n);
  var ua = vec3f(0.0);
  var ta = 0.0;
  var qa = 0.0;
  for (var c = 0u; c < 8u; c++) {
    let kk = tri_corner(t, c, n);
    let w = tri_weight(t, c);
    ua += w * vel[kk].xyz;
    ta += w * tf[kk];
    qa += w * qf[kk];
  }

  let rel = length(ua - vd);
  let re = 2.0 * radius * rel / NU_AIR;
  let sh = 2.0 + 0.6 * sqrt(re) * pow(SC_AIR, 1.0 / 3.0);
  let deficit = max(0.0, q_sat(ta) - qa);
  let dr2 = -(2.0 * DV_H2O * sh * RHO_AIR / RHO_W) * deficit;
  let r_new = sqrt(max(0.0, radius * radius + dr2 * P.dt));
  let dm = RHO_W * (4.0 / 3.0) * PI * max(0.0, radius * radius * radius - r_new * r_new * r_new);

  var tau = RHO_W * (2.0 * radius) * (2.0 * radius) / (18.0 * MU_AIR);
  tau = tau / (1.0 + 0.15 * pow(max(re, 0.0), 0.687));
  tau = max(tau, 1.0e-5);
  let gvec = vec3f(0.0, -G_ACC, 0.0);
  let vel_new = (vd + P.dt * (ua / tau + gvec)) / (1.0 + P.dt / tau);

  // A cell is a real volume now (the slice used h² per metre of depth).
  let mcell = RHO_AIR * P.h.x * P.h.y * P.h.z;
  let dq = mult * dm / mcell;
  var dtemp = 0.0;
  if (P.latent_on > 0.5) {
    dtemp = -(L_VAP * mult * dm) / (mcell * CP_AIR);
  }
  let m_drop = RHO_W * (4.0 / 3.0) * PI * radius * radius * radius;
  let dvel = -(mult * m_drop * (vel_new - vd - gvec * P.dt)) / mcell;

  let w = fluid_tri(t);
  for (var c = 0u; c < 8u; c++) {
    deposit(tri_corner(t, c, n), w[c], dq, dtemp, dvel);
  }

  let pos_new = pos + vel_new * P.dt;
  d.vel_a = vec4f(vel_new, mult);
  d.pos_r = vec4f(pos_new, r_new);

  let m_liquid = mult * RHO_W * (4.0 / 3.0) * PI * r_new * r_new * r_new;
  let wet_amt = clamp((m_liquid / mcell) / 2.0e-4, 0.02, 0.55);

  // Solid contact (house or trunk). The midpoint is tested too so a fast drop
  // cannot skip a one-cell trunk. The water runs down the wall and wets the
  // ground at its foot; the drop dies.
  let mid = 0.5 * (pos + pos_new);
  if (pos_new.y > 0.0 && (solid_at(mid) || solid_at(pos_new))) {
    var last_fluid = pos;
    if (!solid_at(mid)) { last_fluid = mid; }
    if (m_liquid > 1.0e-15) { deposit_ground(last_fluid, vec3f(0.0), wet_amt); }
    d.vel_a.w = 0.0;
    drops[di] = d;
    return;
  }

  let dom = vec3f(n) * P.h;
  let oob = pos_new.y >= dom.y || pos_new.x <= 0.0 || pos_new.x >= dom.x
    || pos_new.z <= 0.0 || pos_new.z >= dom.z;

  if (pos_new.y <= 0.0) {
    if (m_liquid > 1.0e-15) {
      // Splash: a share of the drop's horizontal momentum per unit cell air
      // mass. (The slice adds GROUND_SPLASH * v per drop regardless of mass,
      // which at a volume hose's drop rate runs away into a ground jet.)
      let splash = vec3f(vel_new.x, 0.0, vel_new.z) * GROUND_SPLASH * m_liquid / mcell;
      deposit_ground(pos_new, splash, wet_amt);
    }
    d.vel_a.w = 0.0;
  } else if (r_new < R_KILL || oob) {
    d.vel_a.w = 0.0;
  }
  drops[di] = d;
}

// Pure momentum injection: no droplets, mass, humidity or latent heat.
@compute @workgroup_size(64)
fn vol_air_inject(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.y >= P.emitter_count) { return; }
  let em = E[gid.y];
  if (em.kind != 1u || gid.x >= em.count) { return; }
  var seed = pcg(em.seed ^ (gid.x * 0x9E3779B9u + 77u));

  let dir = cone_dir(em, &seed);
  let speed = em.speed * (0.75 + 0.5 * rand01(&seed));
  let reach = 0.08 + rand01(&seed) * 0.38;
  let origin = disk_point(em, &seed);
  if (solid_at(origin)) { return; }

  // Walk out along the jet and stop in front of the first solid.
  let step_len = 0.5 * min(P.h.x, min(P.h.y, P.h.z));
  var at = origin;
  let steps = u32(ceil(reach / step_len));
  for (var s = 1u; s <= steps; s++) {
    let next = origin + dir * min(reach, f32(s) * step_len);
    if (solid_at(next)) { break; }
    at = next;
  }
  // Same velocity kick per impulse as the slice's air_inject.
  deposit_m_fluid(at, dir * speed * 0.22);
}
