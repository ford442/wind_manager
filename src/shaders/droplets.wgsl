@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read_write> drops : array<Droplet>;
@group(0) @binding(2) var<storage, read_write> emit_counter : atomic<u32>;
@group(0) @binding(3) var<storage, read> vel : array<vec2f>;
@group(0) @binding(4) var<storage, read> tf : array<f32>;
@group(0) @binding(5) var<storage, read> qf : array<f32>;
// Humidity and temperature scatter, interleaved [q, T] per cell.
@group(0) @binding(6) var<storage, read_write> acc_qt : array<atomic<i32>>;
// Solid mask (0 fluid, id + 1 solid). `update` binds 8 storage buffers, the
// WebGPU default per-stage limit (see requiredLimits in render/gpuDevice.ts).
@group(0) @binding(7) var<storage, read> solid : array<u32>;
@group(0) @binding(8) var<storage, read_write> acc_m : array<atomic<i32>>;
// Wetness: nx ground columns, then N_SOLIDS * N_SOLID_FACES wall faces.
@group(0) @binding(9) var<storage, read_write> acc_wet : array<atomic<i32>>;

// Per-emitter parameters for this substep. Layout: EMITTER_PARAMS_LAYOUT in
// src/sim/simParamsUniform.ts (48-byte stride). MAX_EMITTERS is injected by
// step.ts from src/sim/emitters.ts. `emit` / `air_inject` dispatch with
// workgroup_id.y = emitter index, so every emitter in one pass reads its own row.
struct EmitterParams {
  x      : f32,
  y      : f32,
  angle  : f32,
  spread : f32,
  speed  : f32,
  count  : u32,
  kind   : u32,
  r_min  : f32,
  r_max  : f32,
  seed   : u32,
  _pad0  : u32,
  _pad1  : u32,
};
@group(0) @binding(10) var<uniform> E : array<EmitterParams, MAX_EMITTERS>;

fn fixed_pt(v: f32) -> i32 {
  return i32(clamp(round(v), -2.0e9, 2.0e9));
}

fn deposit(k: i32, w: f32, dq: f32, dtemp: f32, dvel: vec2f) {
  if (w <= 0.0) { return; }
  atomicAdd(&acc_qt[2 * k], fixed_pt(w * dq * SCALE_Q));
  atomicAdd(&acc_qt[2 * k + 1], fixed_pt(w * dtemp * SCALE_T));
  atomicAdd(&acc_m[2 * k], fixed_pt(w * dvel.x * SCALE_M));
  atomicAdd(&acc_m[2 * k + 1], fixed_pt(w * dvel.y * SCALE_M));
}

fn deposit_m(k: i32, w: f32, dvel: vec2f) {
  if (w <= 0.0) { return; }
  atomicAdd(&acc_m[2 * k], fixed_pt(w * dvel.x * SCALE_M));
  atomicAdd(&acc_m[2 * k + 1], fixed_pt(w * dvel.y * SCALE_M));
}

fn deposit_wet(i: i32, w: f32, amount: f32) {
  atomicAdd(&acc_wet[i], fixed_pt(w * amount * SCALE_WET));
}

// Solid id at cell (i, j), or -1 for fluid / outside the grid.
fn solid_id(i: i32, j: i32) -> i32 {
  let nx = i32(P.nx);
  if (i < 0 || j < 0 || i >= nx || j >= i32(P.ny)) { return -1; }
  return i32(solid[cell_index(i, j, nx)]) - 1;
}

fn solid_at_pos(pos: vec2f) -> i32 {
  return solid_id(i32(floor(pos.x / P.h)), i32(floor(pos.y / P.h)));
}

fn corner_fluid(b: Bilin) -> vec4f {
  return vec4f(
    select(1.0, 0.0, solid_id(b.i0, b.j0) >= 0), select(1.0, 0.0, solid_id(b.i1, b.j0) >= 0),
    select(1.0, 0.0, solid_id(b.i0, b.j1) >= 0), select(1.0, 0.0, solid_id(b.i1, b.j1) >= 0),
  );
}

// Bilinear scatter that only lands in fluid cells: air inside a house never
// receives heat, vapor or momentum.
fn deposit_fluid(pos: vec2f, dq: f32, dtemp: f32, dvel: vec2f) {
  let nx = i32(P.nx);
  let b = bilin_at(pos, P.h, nx, i32(P.ny));
  let w = fluid_weights(b, corner_fluid(b));
  deposit(cell_index(b.i0, b.j0, nx), w.x, dq, dtemp, dvel);
  deposit(cell_index(b.i1, b.j0, nx), w.y, dq, dtemp, dvel);
  deposit(cell_index(b.i0, b.j1, nx), w.z, dq, dtemp, dvel);
  deposit(cell_index(b.i1, b.j1, nx), w.w, dq, dtemp, dvel);
}

fn deposit_m_fluid(pos: vec2f, dvel: vec2f) {
  let nx = i32(P.nx);
  let b = bilin_at(pos, P.h, nx, i32(P.ny));
  let w = fluid_weights(b, corner_fluid(b));
  deposit_m(cell_index(b.i0, b.j0, nx), w.x, dvel);
  deposit_m(cell_index(b.i1, b.j0, nx), w.y, dvel);
  deposit_m(cell_index(b.i0, b.j1, nx), w.z, dvel);
  deposit_m(cell_index(b.i1, b.j1, nx), w.w, dvel);
}

// A drop that ends a substep inside a solid: leave its water on the face it
// struck, like the floor does in `update` (vapor, a little latent cooling,
// along-surface splash momentum), run some down to the ground, then die.
// `p_from` is the last fluid position, `p_to` where the drop entered solid `id`.
fn wall_contact(p_from: vec2f, p_to: vec2f, id: i32, v: vec2f, r: f32, mcell: f32) {
  let m_liquid = RHO_W * (4.0 / 3.0) * PI * r * r * r;
  if (m_liquid <= 1.0e-15) { return; }
  let nx = i32(P.nx);
  let fi = i32(floor(p_from.x / P.h));
  let fj = i32(floor(p_from.y / P.h));
  let ti = i32(floor(p_to.x / P.h));
  let tj = i32(floor(p_to.y / P.h));
  // Which face: the axis whose step alone enters the solid. A drop dropping
  // onto a roof or trunk top crosses a horizontal face; one flying into a
  // wall crosses a vertical face.
  var side_hit = abs(p_to.x - p_from.x) * 1.0001 >= abs(p_to.y - p_from.y);
  if (ti != fi && tj == fj) { side_hit = true; }
  if (tj != fj && ti == fi) { side_hit = false; }
  if (ti != fi && tj != fj) {
    side_hit = solid_id(ti, fj) >= 0 && solid_id(fi, tj) < 0;
  }

  let dq_w = m_liquid / mcell;
  var dtemp_w = 0.0;
  if (P.latent_on > 0.5) {
    dtemp_w = -(L_VAP * m_liquid * GROUND_TEMP) / (mcell * CP_AIR);
  }
  let wet_amt = clamp(dq_w / 2.0e-4, 0.02, 0.55);
  var splash = vec2f(v.x * GROUND_SPLASH, 0.0);
  let sid = u32(id);
  if (side_hit) {
    // Water runs up/down the wall; the cooled, splashed air leaves along it.
    splash = vec2f(0.0, v.y * WALL_SPLASH);
    let face = select(1u, 0u, p_to.x > p_from.x);
    deposit_wet(solid_face_slot(nx, sid, face), 1.0 - WALL_DRAIN, wet_amt);
  } else {
    // Roof or trunk top: water sheds off both eaves.
    let half_wet = 0.5 * (1.0 - WALL_DRAIN);
    deposit_wet(solid_face_slot(nx, sid, 0u), half_wet, wet_amt);
    deposit_wet(solid_face_slot(nx, sid, 1u), half_wet, wet_amt);
  }
  deposit_fluid(p_from, dq_w, dtemp_w, splash);
  deposit_wet(clamp(fi, 0, nx - 1), WALL_DRAIN, wet_amt);
}

fn deposit_ground(
  i: i32, nx: i32, w: f32,
  dq: f32, dtemp: f32, dvel_h: vec2f, wet_amt: f32,
) {
  let k0 = cell_index(i, 0, nx);
  deposit(k0, w, dq, dtemp, dvel_h);
  let k1 = cell_index(i, 1, nx);
  deposit(k1, w * 0.4, dq * 0.4, dtemp * 0.4, dvel_h * 0.4);
  deposit_wet(i, w, wet_amt);
}

@compute @workgroup_size(64)
fn emit(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.y >= P.emitter_count) { return; }
  let em = E[gid.y];
  if (em.kind != 0u || gid.x >= em.count) { return; }
  var seed = pcg(em.seed ^ (gid.x * 0x9E3779B9u + 1u));

  let slot = atomicAdd(&emit_counter, 1u) % P.max_droplets;

  let a = em.angle + (rand01(&seed) - 0.5) * 2.0 * em.spread;
  let dir = vec2f(sin(a), cos(a));
  let speed = em.speed * (0.8 + 0.4 * rand01(&seed));
  let r = em.r_min * pow(em.r_max / em.r_min, rand01(&seed));

  let jx = (rand01(&seed) - 0.5) * 0.04;
  let jy = (rand01(&seed) - 0.5) * 0.04;

  var d : Droplet;
  d.pos = vec2f(em.x + jx, em.y + jy);
  d.vel = dir * speed;
  d.radius = r;
  d.alive = 1u;
  drops[slot] = d;
}

@compute @workgroup_size(128)
fn update(@builtin(global_invocation_id) gid: vec3u) {
  let di = gid.x;
  if (di >= P.max_droplets) { return; }
  var d = drops[di];
  if (d.alive == 0u) { return; }

  let nx = i32(P.nx);
  let ny = i32(P.ny);

  let b = bilin_at(d.pos, P.h, nx, ny);
  let k00 = cell_index(b.i0, b.j0, nx);
  let k10 = cell_index(b.i1, b.j0, nx);
  let k01 = cell_index(b.i0, b.j1, nx);
  let k11 = cell_index(b.i1, b.j1, nx);
  let ua = mix(mix(vel[k00], vel[k10], b.fx), mix(vel[k01], vel[k11], b.fx), b.fy);
  let ta = mix(mix(tf[k00], tf[k10], b.fx), mix(tf[k01], tf[k11], b.fx), b.fy);
  let qa = mix(mix(qf[k00], qf[k10], b.fx), mix(qf[k01], qf[k11], b.fx), b.fy);

  let rel = length(ua - d.vel);
  let re = 2.0 * d.radius * rel / NU_AIR;

  let sh = 2.0 + 0.6 * sqrt(re) * pow(SC_AIR, 1.0 / 3.0);
  let deficit = max(0.0, q_sat(ta) - qa);
  let dr2 = -(2.0 * DV_H2O * sh * RHO_AIR / RHO_W) * deficit;
  let r_new = sqrt(max(0.0, d.radius * d.radius + dr2 * P.dt));
  let dm = RHO_W * (4.0 / 3.0) * PI
         * max(0.0, d.radius * d.radius * d.radius - r_new * r_new * r_new);

  var tau = RHO_W * (2.0 * d.radius) * (2.0 * d.radius) / (18.0 * MU_AIR);
  tau = tau / (1.0 + 0.15 * pow(max(re, 0.0), 0.687));
  tau = max(tau, 1.0e-5);
  let gvec = vec2f(0.0, -G_ACC);
  let vel_new = (d.vel + P.dt * (ua / tau + gvec)) / (1.0 + P.dt / tau);

  let mcell = RHO_AIR * P.h * P.h;
  let dq = dm / mcell;
  var dtemp = 0.0;
  if (P.latent_on > 0.5) {
    dtemp = -(L_VAP * dm) / (mcell * CP_AIR);
  }

  let m_drop = RHO_W * (4.0 / 3.0) * PI * d.radius * d.radius * d.radius;
  let dvel = -(m_drop * (vel_new - d.vel - gvec * P.dt)) / mcell;

  let w = fluid_weights(b, corner_fluid(b));
  deposit(k00, w.x, dq, dtemp, dvel);
  deposit(k10, w.y, dq, dtemp, dvel);
  deposit(k01, w.z, dq, dtemp, dvel);
  deposit(k11, w.w, dq, dtemp, dvel);

  let pos_prev = d.pos;
  d.vel = vel_new;
  d.pos = d.pos + vel_new * P.dt;
  d.radius = r_new;

  // Solid contact. Test the midpoint too so a fast drop cannot skip a
  // two-cell trunk in one substep.
  var hit = -1;
  var hit_at = d.pos;
  let mid = 0.5 * (pos_prev + d.pos);
  let hit_mid = solid_at_pos(mid);
  if (hit_mid >= 0) {
    hit = hit_mid;
    hit_at = mid;
  } else {
    hit = solid_at_pos(d.pos);
  }
  if (hit >= 0 && d.pos.y > 0.0) {
    var last_fluid = pos_prev;
    if (hit_mid < 0) { last_fluid = mid; }
    wall_contact(last_fluid, hit_at, hit, vel_new, r_new, mcell);
    d.alive = 0u;
    drops[di] = d;
    return;
  }

  let dom_w = f32(nx) * P.h;
  let ht = f32(ny) * P.h;
  let oob = d.pos.y >= ht || d.pos.x <= 0.0 || d.pos.x >= dom_w;

  if (d.pos.y <= 0.0) {
    let m_liquid = RHO_W * (4.0 / 3.0) * PI * r_new * r_new * r_new;
    if (m_liquid > 1.0e-15) {
      let dq_g = m_liquid / mcell;
      var dtemp_g = 0.0;
      if (P.latent_on > 0.5) {
        dtemp_g = -(L_VAP * m_liquid * GROUND_TEMP) / (mcell * CP_AIR);
      }
      let splash = vec2f(vel_new.x * GROUND_SPLASH, 0.0);
      let wet_amt = clamp(dq_g / 2.0e-4, 0.02, 0.55);

      let gx = clamp(d.pos.x / P.h - 0.5, 0.0, f32(nx - 1) - 1.0e-4);
      let i0 = i32(floor(gx));
      let i1 = min(i0 + 1, nx - 1);
      let fx = gx - f32(i0);
      deposit_ground(i0, nx, 1.0 - fx, dq_g, dtemp_g, splash * (1.0 - fx), wet_amt * (1.0 - fx));
      deposit_ground(i1, nx, fx, dq_g, dtemp_g, splash * fx, wet_amt * fx);
    }
    d.alive = 0u;
  } else if (r_new < R_KILL || oob) {
    d.alive = 0u;
  }

  drops[di] = d;
}

@compute @workgroup_size(64)
// Pure momentum injection — no droplets, mass, humidity, or latent heat.
fn air_inject(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.y >= P.emitter_count) { return; }
  let em = E[gid.y];
  if (em.kind != 1u || gid.x >= em.count) { return; }
  var seed = pcg(em.seed ^ (gid.x * 0x9E3779B9u + 77u));

  let a = em.angle + (rand01(&seed) - 0.5) * 2.0 * em.spread;
  let dir = vec2f(sin(a), cos(a));
  let speed = em.speed * (0.75 + 0.5 * rand01(&seed));
  let reach = 0.08 + rand01(&seed) * 0.38;
  let jx = (rand01(&seed) - 0.5) * 0.05;
  let jy = (rand01(&seed) - 0.5) * 0.05;
  let origin = vec2f(em.x + jx, em.y + jy);
  if (solid_at_pos(origin) >= 0) { return; } // nozzle inside a house

  // Walk out along the jet in half-cell steps and stop in front of the first
  // solid: an air stream pushes on a wall, it does not reach the far side.
  var pos = origin;
  let steps = u32(ceil(reach / (0.5 * P.h)));
  for (var s = 1u; s <= steps; s++) {
    let next = origin + dir * min(reach, f32(s) * 0.5 * P.h);
    if (solid_at_pos(next) >= 0) { break; }
    pos = next;
  }

  deposit_m_fluid(pos, dir * speed * 0.22);
}
