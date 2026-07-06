// buoyancy.wgsl — thermal (virtual-temperature) buoyancy plus the gentle
// long-run safeguards, and the boundary-condition pass.

@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read_write> vel : array<vec2f>;
@group(0) @binding(2) var<storage, read_write> tf : array<f32>;
@group(0) @binding(3) var<storage, read_write> qf : array<f32>;

@compute @workgroup_size(8, 8)
fn buoyancy(@builtin(global_invocation_id) gid: vec3u) {
  let nx = i32(P.nx);
  let ny = i32(P.ny);
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (i >= nx || j >= ny) { return; }
  let k = cell_index(i, j, nx);

  // Virtual temperature handles both effects with the right signs:
  // vapor makes the parcel lighter (+0.61*q), latent cooling makes it
  // heavier — and cooling wins by ~13x, so the net spray column SINKS.
  let tv = virt_temp(tf[k], qf[k]);
  let tv_amb = virt_temp(P.t_amb, P.q_amb);
  var v = vel[k];
  v.y += P.dt * G_ACC * (tv - tv_amb) / tv_amb;

  // Gentle safeguards against long-run drift.
  v *= 1.0 - P.damp * P.dt;
  vel[k] = v;
  tf[k] += (P.t_amb - tf[k]) * P.relax * P.dt;
  qf[k] += (P.q_amb - qf[k]) * P.relax * P.dt;
}

// Boundary pass: only edge cells act, and they read interior neighbors that
// no thread writes in this pass, so there is no race.
//  - floor (j=0): free-slip — v=0, horizontal u preserved (lets the cold
//    column spread sideways into the outflow gust)
//  - side walls: zero-gradient (copy interior) — approximates open sides
//  - top: open / ambient inflow — zero-gradient velocity, T,q nudged toward
//    ambient so fresh air enters at ambient conditions
@compute @workgroup_size(8, 8)
fn boundaries(@builtin(global_invocation_id) gid: vec3u) {
  let nx = i32(P.nx);
  let ny = i32(P.ny);
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (i >= nx || j >= ny) { return; }
  let edge = (i == 0) || (i == nx - 1) || (j == 0) || (j == ny - 1);
  if (!edge) { return; }
  let k = cell_index(i, j, nx);

  var v = vel[k];
  var t = tf[k];
  var q = qf[k];

  if (i == 0) {
    let kk = cell_index(1, j, nx);
    v = vel[kk]; t = tf[kk]; q = qf[kk];
  } else if (i == nx - 1) {
    let kk = cell_index(nx - 2, j, nx);
    v = vel[kk]; t = tf[kk]; q = qf[kk];
  }
  if (j == ny - 1) {
    let kk = cell_index(i, ny - 2, nx);
    v = vel[kk];
    t = mix(tf[kk], P.t_amb, 0.5);
    q = mix(qf[kk], P.q_amb, 0.5);
  }
  if (j == 0) {
    v.y = 0.0; // free-slip floor
  }

  vel[k] = v;
  tf[k] = t;
  qf[k] = q;
}
