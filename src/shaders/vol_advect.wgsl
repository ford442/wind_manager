// vol_advect.wgsl — semi-Lagrangian advection of u, T, q (advect.wgsl in 3D).

@group(0) @binding(0) var<uniform> P : VParams;
@group(0) @binding(1) var<storage, read> vel_src : array<vec4f>;
@group(0) @binding(2) var<storage, read> t_src : array<f32>;
@group(0) @binding(3) var<storage, read> q_src : array<f32>;
@group(0) @binding(4) var<storage, read_write> vel_dst : array<vec4f>;
@group(0) @binding(5) var<storage, read_write> t_dst : array<f32>;
@group(0) @binding(6) var<storage, read_write> q_dst : array<f32>;
@group(0) @binding(7) var<storage, read> solid : array<u32>;

@compute @workgroup_size(4, 4, 4)
fn vol_advect(@builtin(global_invocation_id) gid: vec3u) {
  let n = vec3i(i32(P.nx), i32(P.ny), i32(P.nz));
  let c = vec3i(gid);
  if (!vinside(c.x, c.y, c.z, n)) { return; }
  let k = vcell(c.x, c.y, c.z, n);

  if (solid[k] != 0u) {
    // No flow inside a solid; `vol_boundaries` refills T/q from fluid neighbours.
    vel_dst[k] = vec4f(0.0);
    t_dst[k] = t_src[k];
    q_dst[k] = q_src[k];
    return;
  }

  let pos = (vec3f(c) + 0.5) * P.h;
  let back = pos - vel_src[k].xyz * P.dt;
  let t = tri_at(back, P.h, n);

  // Solid corners hold zero velocity, so plain trilinear velocity is the wall's
  // no-slip value. Scalars blend fluid corners only.
  var v = vec3f(0.0);
  var ts = 0.0;
  var qs = 0.0;
  var ws = 0.0;
  for (var cn = 0u; cn < 8u; cn++) {
    let kk = tri_corner(t, cn, n);
    let w = tri_weight(t, cn);
    v += w * vel_src[kk].xyz;
    if (solid[kk] == 0u) {
      ts += w * t_src[kk];
      qs += w * q_src[kk];
      ws += w;
    }
  }
  vel_dst[k] = vec4f(v, 0.0);
  var t_new = t_src[k];
  var q_new = q_src[k];
  if (ws > 1.0e-6) {
    t_new = ts / ws;
    q_new = qs / ws;
  }
  // Relaxation toward ambient lives here (not in vol_buoyancy) so buoyancy
  // can read T/q neighbours that no thread is writing.
  t_dst[k] = t_new + (P.t_amb - t_new) * P.relax * P.dt;
  q_dst[k] = q_new + (P.q_amb - q_new) * P.relax * P.dt;
}
