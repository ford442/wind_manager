@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read_write> drops : array<Droplet>;
@group(0) @binding(2) var<storage, read_write> emit_counter : atomic<u32>;
@group(0) @binding(3) var<storage, read> vel : array<vec2f>;
@group(0) @binding(4) var<storage, read> tf : array<f32>;
@group(0) @binding(5) var<storage, read> qf : array<f32>;
@group(0) @binding(6) var<storage, read_write> acc_q : array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> acc_t : array<atomic<i32>>;
@group(0) @binding(8) var<storage, read_write> acc_m : array<atomic<i32>>;
@group(0) @binding(9) var<storage, read_write> acc_wet : array<atomic<i32>>;

fn fixed_pt(v: f32) -> i32 {
  return i32(clamp(round(v), -2.0e9, 2.0e9));
}

fn deposit(k: i32, w: f32, dq: f32, dtemp: f32, dvel: vec2f) {
  atomicAdd(&acc_q[k], fixed_pt(w * dq * SCALE_Q));
  atomicAdd(&acc_t[k], fixed_pt(w * dtemp * SCALE_T));
  atomicAdd(&acc_m[2 * k], fixed_pt(w * dvel.x * SCALE_M));
  atomicAdd(&acc_m[2 * k + 1], fixed_pt(w * dvel.y * SCALE_M));
}

fn deposit_m(k: i32, w: f32, dvel: vec2f) {
  atomicAdd(&acc_m[2 * k], fixed_pt(w * dvel.x * SCALE_M));
  atomicAdd(&acc_m[2 * k + 1], fixed_pt(w * dvel.y * SCALE_M));
}

fn deposit_wet(i: i32, w: f32, amount: f32) {
  atomicAdd(&acc_wet[i], fixed_pt(w * amount * SCALE_WET));
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
  if (gid.x >= P.emit_count) { return; }
  var seed = pcg(P.seed ^ (gid.x * 0x9E3779B9u + 1u));

  let slot = atomicAdd(&emit_counter, 1u) % P.max_droplets;

  let a = P.emit_angle + (rand01(&seed) - 0.5) * 2.0 * P.emit_spread;
  let dir = vec2f(sin(a), cos(a));
  let speed = P.emit_speed * (0.8 + 0.4 * rand01(&seed));
  let r = P.r_min * pow(P.r_max / P.r_min, rand01(&seed));

  let jx = (rand01(&seed) - 0.5) * 0.04;
  let jy = (rand01(&seed) - 0.5) * 0.04;

  var d : Droplet;
  d.pos = vec2f(P.emit_x + jx, P.emit_y + jy);
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

  let w00 = (1.0 - b.fx) * (1.0 - b.fy);
  let w10 = b.fx * (1.0 - b.fy);
  let w01 = (1.0 - b.fx) * b.fy;
  let w11 = b.fx * b.fy;
  deposit(k00, w00, dq, dtemp, dvel);
  deposit(k10, w10, dq, dtemp, dvel);
  deposit(k01, w01, dq, dtemp, dvel);
  deposit(k11, w11, dq, dtemp, dvel);

  d.vel = vel_new;
  d.pos = d.pos + vel_new * P.dt;
  d.radius = r_new;

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
  if (gid.x >= P.emit_count) { return; }
  var seed = pcg(P.seed ^ (gid.x * 0x9E3779B9u + 77u));

  let a = P.emit_angle + (rand01(&seed) - 0.5) * 2.0 * P.emit_spread;
  let dir = vec2f(sin(a), cos(a));
  let speed = P.emit_speed * (0.75 + 0.5 * rand01(&seed));
  let reach = 0.08 + rand01(&seed) * 0.38;
  let jx = (rand01(&seed) - 0.5) * 0.05;
  let jy = (rand01(&seed) - 0.5) * 0.05;
  let pos = vec2f(P.emit_x + jx, P.emit_y + jy) + dir * reach;

  let nx = i32(P.nx);
  let ny = i32(P.ny);
  let b = bilin_at(pos, P.h, nx, ny);
  let k00 = cell_index(b.i0, b.j0, nx);
  let k10 = cell_index(b.i1, b.j0, nx);
  let k01 = cell_index(b.i0, b.j1, nx);
  let k11 = cell_index(b.i1, b.j1, nx);

  let impulse = dir * speed * 0.22;
  let w00 = (1.0 - b.fx) * (1.0 - b.fy);
  let w10 = b.fx * (1.0 - b.fy);
  let w01 = (1.0 - b.fx) * b.fy;
  let w11 = b.fx * b.fy;
  deposit_m(k00, w00, impulse);
  deposit_m(k10, w10, impulse);
  deposit_m(k01, w01, impulse);
  deposit_m(k11, w11, impulse);
}
