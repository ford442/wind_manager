// vol_pressure.wgsl — divergence, multigrid smoother / transfers and the
// projection for the yard volume.
//
// Approximate (face) projection, not the slice's GᵀG: face velocities are cell
// averages, A is the compact 7-point face Laplacian (src/volume/pressure.ts
// builds it per level and explains why), and each cell velocity is corrected
// by the average of its two face pressure gradients. Edge weights are in units
// of h.x, so a non-cubic cell only changes the weights.

// Mirrors VOL_PRESSURE_OMEGA / VOL_PRESSURE_OVERCORRECT in src/volume/pressure.ts.
const VOL_OMEGA : f32 = 0.857142857;
const VOL_OVERCORRECT : f32 = 1.8;

struct Level { nx: u32, ny: u32, nz: u32, _pad: u32 }
// edges_a.xyz = faces to (x+1, y+1, z+1), stored on the lower node; edges_a.w
// and edges_b are unused (kept so the node stays 32 bytes, like the slice's).
struct VPressureNode { edges_a: vec4f, edges_b: vec2f, diagonal: f32, fluid: u32 }

@group(0) @binding(0) var<uniform> P : VParams;
@group(0) @binding(1) var<storage, read_write> vel : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> rhs : array<f32>;
@group(0) @binding(3) var<storage, read> p_in : array<f32>;
@group(0) @binding(4) var<storage, read_write> p_out : array<f32>;
@group(0) @binding(5) var<storage, read> solid : array<u32>;
@group(0) @binding(6) var<uniform> L : Level;
@group(0) @binding(7) var<storage, read> graph : array<VPressureNode>;
@group(0) @binding(8) var<storage, read> coarse_p : array<f32>;
@group(0) @binding(9) var<storage, read_write> coarse_rhs : array<f32>;
@group(0) @binding(10) var<storage, read_write> coarse_zero : array<f32>;

fn dims() -> vec3i { return vec3i(i32(P.nx), i32(P.ny), i32(P.nz)); }

fn fluid(c: vec3i) -> bool {
  let n = dims();
  if (!vinside(c.x, c.y, c.z, n)) { return false; }
  return solid[vcell(c.x, c.y, c.z, n)] == 0u;
}

fn axis_unit(a: u32) -> vec3i {
  return select(select(vec3i(0, 0, 1), vec3i(0, 1, 0), a == 1u), vec3i(1, 0, 0), a == 0u);
}

// Face velocity on the `dir` (+1 / -1) face of fluid cell c along axis a:
// the average with a fluid neighbour, 0 on a solid face or the floor, the
// cell's own value on the open sides and top.
fn face_u(c: vec3i, a: u32, dir: i32) -> f32 {
  let n = dims();
  let o = c + axis_unit(a) * dir;
  let u = vel[vcell(c.x, c.y, c.z, n)][a];
  if (!vinside(o.x, o.y, o.z, n)) {
    if (a == 1u && dir < 0) { return 0.0; }
    return u;
  }
  if (!fluid(o)) { return 0.0; }
  return 0.5 * (u + vel[vcell(o.x, o.y, o.z, n)][a]);
}

@compute @workgroup_size(4, 4, 4)
fn vol_divergence(@builtin(global_invocation_id) gid: vec3u) {
  let n = dims();
  let c = vec3i(gid);
  if (!vinside(c.x, c.y, c.z, n)) { return; }
  let k = vcell(c.x, c.y, c.z, n);
  // p0 keeps the previous substep's pressure: the V-cycles warm-start from it.
  rhs[k] = 0.0;
  if (!fluid(c)) { return; }
  var div = 0.0;
  for (var a = 0u; a < 3u; a++) {
    div += (face_u(c, a, 1) - face_u(c, a, -1)) / P.h[a];
  }
  // A p = -h.x² D_f u_f  (A is the positive form of -D_f G_f, in units of h.x).
  rhs[k] = -P.h.x * P.h.x * div;
}

fn neighbour_sum(k: u32) -> f32 {
  let offsets = array<u32, 3>(1u, L.nx, L.nx * L.ny);
  var sum = 0.0;
  for (var d = 0u; d < 3u; d++) {
    let w = graph[k].edges_a[d];
    if (w > 0.0) { sum += w * p_in[k + offsets[d]]; }
    if (k >= offsets[d]) {
      let nb = k - offsets[d];
      let back = graph[nb].edges_a[d];
      if (back > 0.0) { sum += back * p_in[nb]; }
    }
  }
  return sum;
}

fn level_index(g: vec3u) -> u32 {
  return (g.z * L.ny + g.y) * L.nx + g.x;
}

@compute @workgroup_size(4, 4, 4)
fn vol_pressure_smooth(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= L.nx || gid.y >= L.ny || gid.z >= L.nz) { return; }
  let k = level_index(gid);
  let diagonal = graph[k].diagonal;
  p_out[k] = 0.0;
  if (graph[k].fluid == 0u || diagonal <= 0.0) { return; }
  p_out[k] = mix(p_in[k], (rhs[k] + neighbour_sum(k)) / diagonal, VOL_OMEGA);
}

// Coarse node (x, y, z) aggregates the 2×2×2 fine block at 2 (x, y, z).
@compute @workgroup_size(4, 4, 4)
fn vol_restrict_residual(@builtin(global_invocation_id) gid: vec3u) {
  let cn = vec3u(L.nx, L.ny, L.nz) / 2u;
  if (gid.x >= cn.x || gid.y >= cn.y || gid.z >= cn.z) { return; }
  let ck = (gid.z * cn.y + gid.y) * cn.x + gid.x;
  var residual = 0.0;
  for (var c = 0u; c < 8u; c++) {
    let f = 2u * gid + vec3u(c & 1u, (c >> 1u) & 1u, (c >> 2u) & 1u);
    let k = level_index(f);
    if (graph[k].fluid != 0u) {
      residual += rhs[k] - graph[k].diagonal * p_in[k] + neighbour_sum(k);
    }
  }
  coarse_rhs[ck] = residual;
  coarse_zero[ck] = 0.0;
}

@compute @workgroup_size(4, 4, 4)
fn vol_prolongate(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= L.nx || gid.y >= L.ny || gid.z >= L.nz) { return; }
  let k = level_index(gid);
  if (graph[k].fluid != 0u) {
    let cn = vec3u(L.nx, L.ny, L.nz) / 2u;
    let pc = gid / 2u;
    p_out[k] += VOL_OVERCORRECT * coarse_p[(pc.z * cn.y + pc.y) * cn.x + pc.x];
  }
}

// Pressure gradient on the `dir` face of fluid cell c along axis a: zero on
// solid faces, the floor and the open sides (Neumann); zero ghost pressure
// above the top.
fn face_grad(c: vec3i, k: i32, a: u32, dir: i32) -> f32 {
  let n = dims();
  let o = c + axis_unit(a) * dir;
  if (!vinside(o.x, o.y, o.z, n)) {
    if (a == 1u && dir > 0) { return (0.0 - p_in[k]) / P.h.y; }
    return 0.0;
  }
  if (!fluid(o)) { return 0.0; }
  return f32(dir) * (p_in[vcell(o.x, o.y, o.z, n)] - p_in[k]) / P.h[a];
}

@compute @workgroup_size(4, 4, 4)
fn vol_project(@builtin(global_invocation_id) gid: vec3u) {
  let n = dims();
  let c = vec3i(gid);
  if (!vinside(c.x, c.y, c.z, n)) { return; }
  let k = vcell(c.x, c.y, c.z, n);
  if (!fluid(c)) { vel[k] = vec4f(0.0); return; }
  var v = vel[k].xyz;
  for (var a = 0u; a < 3u; a++) {
    v[a] -= 0.5 * (face_grad(c, k, a, 1) + face_grad(c, k, a, -1));
  }
  // No flow into a wall: drop the component pointing into a solid neighbour
  // or the floor (as vol_boundaries does before the solve).
  for (var a = 0u; a < 3u; a++) {
    let e = axis_unit(a);
    if (vinside(c.x + e.x, c.y + e.y, c.z + e.z, n) && !fluid(c + e)) { v[a] = min(v[a], 0.0); }
    if (vinside(c.x - e.x, c.y - e.y, c.z - e.z, n) && !fluid(c - e)) { v[a] = max(v[a], 0.0); }
  }
  if (c.y == 0) { v.y = 0.0; }
  vel[k] = vec4f(v, 0.0);
}
