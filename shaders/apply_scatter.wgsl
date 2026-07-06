@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read_write> vel : array<vec2f>;
@group(0) @binding(2) var<storage, read_write> tf : array<f32>;
@group(0) @binding(3) var<storage, read_write> qf : array<f32>;
@group(0) @binding(4) var<storage, read_write> acc_q : array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> acc_t : array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> acc_m : array<atomic<i32>>;

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
