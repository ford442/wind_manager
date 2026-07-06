@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read> vel_src : array<vec2f>;
@group(0) @binding(2) var<storage, read> t_src : array<f32>;
@group(0) @binding(3) var<storage, read> q_src : array<f32>;
@group(0) @binding(4) var<storage, read_write> vel_dst : array<vec2f>;
@group(0) @binding(5) var<storage, read_write> t_dst : array<f32>;
@group(0) @binding(6) var<storage, read_write> q_dst : array<f32>;

@compute @workgroup_size(8, 8)
fn advect(@builtin(global_invocation_id) gid: vec3u) {
  let nx = i32(P.nx);
  let ny = i32(P.ny);
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (i >= nx || j >= ny) { return; }
  let k = cell_index(i, j, nx);

  let pos = (vec2f(f32(i), f32(j)) + 0.5) * P.h;
  let back = pos - vel_src[k] * P.dt;

  let b = bilin_at(back, P.h, nx, ny);
  let k00 = cell_index(b.i0, b.j0, nx);
  let k10 = cell_index(b.i1, b.j0, nx);
  let k01 = cell_index(b.i0, b.j1, nx);
  let k11 = cell_index(b.i1, b.j1, nx);

  vel_dst[k] = mix(mix(vel_src[k00], vel_src[k10], b.fx),
                   mix(vel_src[k01], vel_src[k11], b.fx), b.fy);
  t_dst[k] = mix(mix(t_src[k00], t_src[k10], b.fx),
                 mix(t_src[k01], t_src[k11], b.fx), b.fy);
  q_dst[k] = mix(mix(q_src[k00], q_src[k10], b.fx),
                 mix(q_src[k01], q_src[k11], b.fx), b.fy);
}
