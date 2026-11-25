struct Level { nx: u32, ny: u32, _pad: vec2u }
struct PressureNode { edges: vec4f, diagonal: f32, fluid: u32, _pad: vec2u }

@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read_write> vel : array<vec2f>;
@group(0) @binding(2) var<storage, read_write> rhs : array<f32>;
@group(0) @binding(3) var<storage, read> p_in : array<f32>;
@group(0) @binding(4) var<storage, read_write> p_out : array<f32>;
@group(0) @binding(5) var<storage, read> solid : array<u32>;
@group(0) @binding(6) var<uniform> L : Level;
@group(0) @binding(7) var<storage, read> graph : array<PressureNode>;
@group(0) @binding(8) var<storage, read> coarse_p : array<f32>;
@group(0) @binding(9) var<storage, read_write> coarse_rhs : array<f32>;
@group(0) @binding(10) var<storage, read_write> coarse_zero : array<f32>;

fn fluid(i: i32, j: i32) -> bool {
  if (i < 0 || j < 0 || i >= i32(P.nx) || j >= i32(P.ny)) { return false; }
  return solid[cell_index(i, j, i32(P.nx))] == 0u;
}

fn free_x(i: i32, j: i32) -> bool {
  return fluid(i, j) && (i == 0 || fluid(i - 1, j)) &&
    (i == i32(P.nx) - 1 || fluid(i + 1, j));
}

fn free_y(i: i32, j: i32) -> bool {
  return fluid(i, j) && j > 0 && fluid(i, j - 1) &&
    (j == i32(P.ny) - 1 || fluid(i, j + 1));
}

fn vx(i: i32, j: i32) -> f32 {
  if (!free_x(i, j)) { return 0.0; }
  return vel[cell_index(i, j, i32(P.nx))].x;
}

fn vy(i: i32, j: i32) -> f32 {
  if (!free_y(i, j)) { return 0.0; }
  return vel[cell_index(i, j, i32(P.nx))].y;
}

@compute @workgroup_size(8, 8)
fn divergence(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= P.nx || gid.y >= P.ny) { return; }
  let i = i32(gid.x);
  let j = i32(gid.y);
  let k = cell_index(i, j, i32(P.nx));
  p_out[k] = 0.0;
  rhs[k] = 0.0;
  if (!fluid(i, j)) { return; }
  // h Gᵀu: the negative divergence, including side-gradient boundary rows.
  var d = vx(i - 1, j) - vx(i + 1, j) + vy(i, j - 1) - vy(i, j + 1);
  if (i == 0) { d -= vx(i, j); }
  if (i == i32(P.nx) - 1) { d += vx(i, j); }
  rhs[k] = 0.5 * P.h * d;
}

fn neighbour_sum(k: u32) -> f32 {
  let offsets = array<u32, 4>(1u, L.nx, 2u, 2u * L.nx);
  var sum = 0.0;
  for (var d = 0u; d < 4u; d++) {
    let w = graph[k].edges[d];
    if (w > 0.0) { sum += w * p_in[k + offsets[d]]; }
    if (k >= offsets[d]) {
      let n = k - offsets[d];
      let back = graph[n].edges[d];
      if (back > 0.0) { sum += back * p_in[n]; }
    }
  }
  return sum;
}

@compute @workgroup_size(8, 8)
fn pressure_smooth(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= L.nx || gid.y >= L.ny) { return; }
  let k = gid.y * L.nx + gid.x;
  let diagonal = graph[k].diagonal;
  p_out[k] = 0.0;
  if (graph[k].fluid == 0u || diagonal <= 0.0) { return; }
  // Weighted Jacobi damps the high-frequency error before restriction.
  p_out[k] = mix(p_in[k], (rhs[k] + neighbour_sum(k)) / diagonal, 2.0 / 3.0);
}

@compute @workgroup_size(8, 8)
fn restrict_residual(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= L.nx / 2u || gid.y >= L.ny / 2u) { return; }
  let ck = gid.y * (L.nx / 2u) + gid.x;
  var residual = 0.0;
  for (var y = 0u; y < 2u; y++) {
    for (var x = 0u; x < 2u; x++) {
      let fx = 4u * (gid.x / 2u) + gid.x % 2u + 2u * x;
      let fy = 4u * (gid.y / 2u) + gid.y % 2u + 2u * y;
      let k = fy * L.nx + fx;
      if (graph[k].fluid != 0u) {
        residual += rhs[k] - graph[k].diagonal * p_in[k] + neighbour_sum(k);
      }
    }
  }
  coarse_rhs[ck] = residual;
  coarse_zero[ck] = 0.0;
}

@compute @workgroup_size(8, 8)
fn prolongate(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= L.nx || gid.y >= L.ny) { return; }
  let k = gid.y * L.nx + gid.x;
  // Inactive coarse aggregates have identically zero correction.
  if (graph[k].fluid != 0u) {
    let cx = 2u * (gid.x / 4u) + gid.x % 2u;
    let cy = 2u * (gid.y / 4u) + gid.y % 2u;
    p_out[k] += coarse_p[cy * (L.nx / 2u) + cx];
  }
}

@compute @workgroup_size(8, 8)
fn project(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= P.nx || gid.y >= P.ny) { return; }
  let i = i32(gid.x);
  let j = i32(gid.y);
  let nx = i32(P.nx);
  let k = cell_index(i, j, nx);
  if (!fluid(i, j)) { vel[k] = vec2f(0.0); return; }
  var v = vec2f(0.0);
  if (free_x(i, j)) {
    let w = cell_index(max(0, i - 1), j, nx);
    let e = cell_index(min(nx - 1, i + 1), j, nx);
    v.x = vel[k].x - 0.5 * (p_in[e] - p_in[w]) / P.h;
  }
  if (free_y(i, j)) {
    var north = 0.0;
    if (j + 1 < i32(P.ny)) { north = p_in[k + nx]; }
    v.y = vel[k].y - 0.5 * (north - p_in[k - nx]) / P.h;
  }
  vel[k] = v;
}
