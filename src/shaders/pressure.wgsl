@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read_write> vel : array<vec2f>;
@group(0) @binding(2) var<storage, read_write> div : array<f32>;
@group(0) @binding(3) var<storage, read> p_in : array<f32>;
@group(0) @binding(4) var<storage, read_write> p_out : array<f32>;

fn vel_at(i: i32, j: i32, nx: i32, ny: i32) -> vec2f {
  return vel[cell_index(clamp(i, 0, nx - 1), clamp(j, 0, ny - 1), nx)];
}

fn p_at(i: i32, j: i32, k_center: i32, nx: i32, ny: i32) -> f32 {
  if (j >= ny) { return 0.0; }
  if (i < 0 || i >= nx || j < 0) { return p_in[k_center]; }
  return p_in[cell_index(i, j, nx)];
}

@compute @workgroup_size(8, 8)
fn divergence(@builtin(global_invocation_id) gid: vec3u) {
  let nx = i32(P.nx);
  let ny = i32(P.ny);
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (i >= nx || j >= ny) { return; }
  let k = cell_index(i, j, nx);

  div[k] = -0.5 * P.h * ((vel_at(i + 1, j, nx, ny).x - vel_at(i - 1, j, nx, ny).x)
                       + (vel_at(i, j + 1, nx, ny).y - vel_at(i, j - 1, nx, ny).y));
  p_out[k] = 0.0;
}

@compute @workgroup_size(8, 8)
fn jacobi(@builtin(global_invocation_id) gid: vec3u) {
  let nx = i32(P.nx);
  let ny = i32(P.ny);
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (i >= nx || j >= ny) { return; }
  let k = cell_index(i, j, nx);

  p_out[k] = (div[k]
            + p_at(i + 1, j, k, nx, ny) + p_at(i - 1, j, k, nx, ny)
            + p_at(i, j + 1, k, nx, ny) + p_at(i, j - 1, k, nx, ny)) * 0.25;
}

@compute @workgroup_size(8, 8)
fn project(@builtin(global_invocation_id) gid: vec3u) {
  let nx = i32(P.nx);
  let ny = i32(P.ny);
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (i >= nx || j >= ny) { return; }
  let k = cell_index(i, j, nx);

  var v = vel[k];
  v.x -= 0.5 * (p_at(i + 1, j, k, nx, ny) - p_at(i - 1, j, k, nx, ny)) / P.h;
  v.y -= 0.5 * (p_at(i, j + 1, k, nx, ny) - p_at(i, j - 1, k, nx, ny)) / P.h;
  vel[k] = v;
}
