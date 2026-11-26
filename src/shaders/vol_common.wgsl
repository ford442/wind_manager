// vol_common.wgsl — the yard volume's structs and trilinear helpers.
//
// Assembled after common.wgsl (constants, q_sat, virt_temp, pcg, rand01), so
// every vol_*.wgsl module is: generated + common.wgsl + vol_common.wgsl + file.
// The 2D slice structs in common.wgsl are unused here; the volume adds the `k`
// (z) index and keeps everything else.
//
// Cell (i, j, k) is x (along the yard), y (up), z (across the yard, from the
// back fence). Index = (k * ny + j) * nx + i. Cell centres sit at
// (ijk + 0.5) * h; `h` is per axis so a slab grid (e.g. 256 x 48 x 8) works.

struct VParams {
  // Layout: VOL_PARAMS_LAYOUT in src/volume/uniforms.ts (64 bytes).
  nx           : u32,
  ny           : u32,
  nz           : u32,
  max_droplets : u32,
  h            : vec3f,
  dt           : f32,
  t_amb        : f32,
  q_amb        : f32,
  latent_on    : f32,
  time         : f32,
  seed         : u32,
  emitter_count: u32,
  relax        : f32,
  damp         : f32,
};

// One water parcel. `pos_r` = (x, y, z, radius); `vel_a` = (vx, vy, vz,
// multiplicity): `w` real drops of this radius, 0 when the parcel is dead.
struct VDroplet {
  pos_r : vec4f,
  vel_a : vec4f,
};

fn vcell(i: i32, j: i32, k: i32, n: vec3i) -> i32 {
  return (k * n.y + j) * n.x + i;
}

fn vinside(i: i32, j: i32, k: i32, n: vec3i) -> bool {
  return i >= 0 && j >= 0 && k >= 0 && i < n.x && j < n.y && k < n.z;
}

// Trilinear stencil at a world position (clamped to cell centres, like bilin_at).
struct Tri {
  i0 : vec3i,
  i1 : vec3i,
  f  : vec3f,
};

fn tri_at(pos: vec3f, h: vec3f, n: vec3i) -> Tri {
  let g = clamp(pos / h - 0.5, vec3f(0.0), vec3f(n - 1) - 1.0e-4);
  let i0 = vec3i(floor(g));
  var t : Tri;
  t.i0 = i0;
  t.i1 = min(i0 + 1, n - 1);
  t.f = g - vec3f(i0);
  return t;
}

// Corner c in 0..8: bit 0 = x, bit 1 = y, bit 2 = z.
fn tri_corner(t: Tri, c: u32, n: vec3i) -> i32 {
  let i = select(t.i0.x, t.i1.x, (c & 1u) != 0u);
  let j = select(t.i0.y, t.i1.y, (c & 2u) != 0u);
  let k = select(t.i0.z, t.i1.z, (c & 4u) != 0u);
  return vcell(i, j, k, n);
}

fn tri_weight(t: Tri, c: u32) -> f32 {
  let wx = select(1.0 - t.f.x, t.f.x, (c & 1u) != 0u);
  let wy = select(1.0 - t.f.y, t.f.y, (c & 2u) != 0u);
  let wz = select(1.0 - t.f.z, t.f.z, (c & 4u) != 0u);
  return wx * wy * wz;
}

// Cell containing a world position, or (-1, -1, -1) outside the grid.
fn vcell_of(pos: vec3f, h: vec3f, n: vec3i) -> vec3i {
  let c = vec3i(floor(pos / h));
  if (!vinside(c.x, c.y, c.z, n)) { return vec3i(-1); }
  return c;
}
