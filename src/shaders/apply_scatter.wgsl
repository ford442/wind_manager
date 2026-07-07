@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read_write> vel : array<vec2f>;
@group(0) @binding(2) var<storage, read_write> tf : array<f32>;
@group(0) @binding(3) var<storage, read_write> qf : array<f32>;
@group(0) @binding(4) var<storage, read_write> acc_q : array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> acc_t : array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> acc_m : array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> wet : array<f32>;
@group(0) @binding(8) var<storage, read_write> acc_wet : array<atomic<i32>>;
@group(0) @binding(9) var<storage, read_write> q_dep : array<f32>;

@compute @workgroup_size(64)
fn apply(@builtin(global_invocation_id) gid: vec3u) {
  let k = i32(gid.x);
  if (gid.x >= P.nx * P.ny) { return; }

  let dq = f32(atomicExchange(&acc_q[k], 0)) / SCALE_Q;
  let dtemp = f32(atomicExchange(&acc_t[k], 0)) / SCALE_T;
  let dvx = f32(atomicExchange(&acc_m[2 * k], 0)) / SCALE_M;
  let dvy = f32(atomicExchange(&acc_m[2 * k + 1], 0)) / SCALE_M;

  qf[k] = clamp(qf[k] + clamp(dq, 0.0, 5.0e-3), 0.0, 0.05);
  tf[k] = clamp(tf[k] + clamp(dtemp, -10.0, 10.0), P.t_amb - 40.0, P.t_amb + 40.0);
  let dv = clamp(vec2f(dvx, dvy), vec2f(-5.0), vec2f(5.0));
  vel[k] = clamp(vel[k] + dv, vec2f(-40.0), vec2f(40.0));
}

@compute @workgroup_size(64)
fn apply_wet(@builtin(global_invocation_id) gid: vec3u) {
  let i = i32(gid.x);
  if (gid.x >= P.nx) { return; }

  let nx = i32(P.nx);
  let dw = f32(atomicExchange(&acc_wet[i], 0)) / SCALE_WET;
  var w = wet[i] * (1.0 - WET_DECAY * P.dt) + clamp(dw, 0.0, 0.65);
  wet[i] = clamp(w, 0.0, 1.0);

  let k0 = cell_index(i, 0, nx);
  let k1 = cell_index(i, 1, nx);
  let q_surf = mix(qf[k0], qf[k1], 0.55);
  let q_ex = max(0.0, q_surf - P.q_amb);
  let splash = clamp(dw * 2.2, 0.0, 0.4);
  var dep = q_dep[i] * (1.0 - QDEP_DECAY * P.dt);
  dep += (q_ex * QDEP_INTEG + splash) * P.dt;
  q_dep[i] = clamp(dep, 0.0, 1.0);
}
