@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read> vel_src : array<vec2f>;
@group(0) @binding(2) var<storage, read> t_src : array<f32>;
@group(0) @binding(3) var<storage, read> q_src : array<f32>;
@group(0) @binding(4) var<storage, read_write> vel_dst : array<vec2f>;
@group(0) @binding(5) var<storage, read_write> t_dst : array<f32>;
@group(0) @binding(6) var<storage, read_write> q_dst : array<f32>;
@group(0) @binding(7) var<storage, read> solid : array<u32>;

fn fluid01(k: i32) -> f32 {
  return select(0.0, 1.0, solid[k] == 0u);
}

@compute @workgroup_size(8, 8)
fn advect(@builtin(global_invocation_id) gid: vec3u) {
  let nx = i32(P.nx);
  let ny = i32(P.ny);
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (i >= nx || j >= ny) { return; }
  let k = cell_index(i, j, nx);

  if (solid[k] != 0u) {
    // Not advected: no flow inside a solid. `boundaries` refills T/q from the
    // fluid neighbours so samplers near the wall see air values.
    vel_dst[k] = vec2f(0.0);
    t_dst[k] = t_src[k];
    q_dst[k] = q_src[k];
    return;
  }

  let pos = (vec2f(f32(i), f32(j)) + 0.5) * P.h;
  let back = pos - vel_src[k] * P.dt;

  let b = bilin_at(back, P.h, nx, ny);
  let k00 = cell_index(b.i0, b.j0, nx);
  let k10 = cell_index(b.i1, b.j0, nx);
  let k01 = cell_index(b.i0, b.j1, nx);
  let k11 = cell_index(b.i1, b.j1, nx);

  // Solid corners hold zero velocity, so plain bilinear velocity is the wall's
  // no-slip value. Scalars only blend fluid corners: T/q never come out of a solid.
  vel_dst[k] = mix(mix(vel_src[k00], vel_src[k10], b.fx),
                   mix(vel_src[k01], vel_src[k11], b.fx), b.fy);
  let w = fluid_weights(b, vec4f(fluid01(k00), fluid01(k10), fluid01(k01), fluid01(k11)));
  if (w.x + w.y + w.z + w.w <= 0.0) {
    t_dst[k] = t_src[k];
    q_dst[k] = q_src[k];
    return;
  }
  t_dst[k] = w.x * t_src[k00] + w.y * t_src[k10] + w.z * t_src[k01] + w.w * t_src[k11];
  q_dst[k] = w.x * q_src[k00] + w.y * q_src[k10] + w.z * q_src[k01] + w.w * q_src[k11];
}
