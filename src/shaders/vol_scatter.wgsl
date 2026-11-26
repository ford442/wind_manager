// vol_scatter.wgsl — apply the droplet scatter to u, T, q (apply_scatter.wgsl in 3D).

@group(0) @binding(0) var<uniform> P : VParams;
@group(0) @binding(1) var<storage, read_write> vel : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> tf : array<f32>;
@group(0) @binding(3) var<storage, read_write> qf : array<f32>;
@group(0) @binding(4) var<storage, read_write> acc_qt : array<atomic<i32>>;
@group(0) @binding(5) var<storage, read> solid : array<u32>;
@group(0) @binding(6) var<storage, read_write> acc_m : array<atomic<i32>>;
// Ground wetness per floor cell (k * nx + i) and its accumulator.
@group(0) @binding(7) var<storage, read_write> wet : array<f32>;
@group(0) @binding(8) var<storage, read_write> acc_wet : array<atomic<i32>>;

@compute @workgroup_size(64)
fn vol_apply(@builtin(global_invocation_id) gid: vec3u) {
  let k = i32(gid.x);
  if (gid.x >= P.nx * P.ny * P.nz) { return; }

  let dq = f32(atomicExchange(&acc_qt[2 * k], 0)) / SCALE_Q;
  let dtemp = f32(atomicExchange(&acc_qt[2 * k + 1], 0)) / SCALE_T;
  let dvx = f32(atomicExchange(&acc_m[3 * k], 0)) / SCALE_M;
  let dvy = f32(atomicExchange(&acc_m[3 * k + 1], 0)) / SCALE_M;
  let dvz = f32(atomicExchange(&acc_m[3 * k + 2], 0)) / SCALE_M;
  if (solid[k] != 0u) { return; }

  qf[k] = clamp(qf[k] + clamp(dq, 0.0, 5.0e-3), 0.0, 0.05);
  tf[k] = clamp(tf[k] + clamp(dtemp, -10.0, 10.0), P.t_amb - 40.0, P.t_amb + 40.0);
  let dv = clamp(vec3f(dvx, dvy, dvz), vec3f(-5.0), vec3f(5.0));
  vel[k] = vec4f(clamp(vel[k].xyz + dv, vec3f(-40.0), vec3f(40.0)), 0.0);
}

// Wet ground evaporates into the cell above it at a surface transfer velocity
// (m/s), driven by the saturation deficit: the floor can cool the air to its
// wet-bulb temperature and no further.
const WET_EVAP_VEL : f32 = 0.005;

// One thread per floor cell.
@compute @workgroup_size(64)
fn vol_apply_wet(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= P.nx * P.nz) { return; }
  let i = i32(gid.x);
  let dw = f32(atomicExchange(&acc_wet[i], 0)) / SCALE_WET;
  let w = clamp(wet[i] * (1.0 - WET_DECAY * P.dt) + clamp(dw, 0.0, 0.65), 0.0, 1.0);
  wet[i] = w;
  let k = (i / i32(P.nx)) * i32(P.ny) * i32(P.nx) + i % i32(P.nx); // cell (i % nx, 0, i / nx)
  if (solid[k] != 0u || w <= 0.0) { return; }
  let deficit = max(0.0, q_sat(tf[k]) - qf[k]);
  let dq = min(deficit, w * WET_EVAP_VEL / P.h.y * deficit * P.dt);
  qf[k] += dq;
  if (P.latent_on > 0.5) {
    tf[k] -= L_VAP * dq / CP_AIR;
  }
}
