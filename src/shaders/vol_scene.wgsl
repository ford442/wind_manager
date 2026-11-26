// vol_scene.wgsl — the yard, drawn from the volume's fields. Consumer:
// src/volume/scene.ts (uniform layout SCENE_LAYOUT there).
//
// Frame flow (one command buffer, after the solver substeps):
//   1. `scene_update` (compute): low-pass tree sway from the wind loading on
//      each tree (velocity sampled up its height) into `state`.
//   2. Opaque pass with depth: sky, ground sheet, grass blades bent by the
//      bottom layer of velocity, solid boxes (exactly the mask's houses and
//      trunks), upper trunks and crowns that sway, emitter markers, drops.
//   3. Overlay pass: temperature raymarch through T (stops at scene depth),
//      then the side view: the z slice drawn like the slice app's overlay.

const N_TREES : u32 = 5u;
const N_HOUSES : u32 = 3u;
const CROWN_BLOBS : u32 = 9u;
const STATE_PER_TREE : u32 = 4u; // trunk sway (x, z), crown sway (x, z), metres

// SceneU.flags
const F_VOLUME : u32 = 1u;
const F_DROPS : u32 = 2u;
const F_GRASS : u32 = 4u;
const F_YARD : u32 = 8u;
const F_SIDE_SPEED : u32 = 16u;

struct SceneU {
  view_proj     : mat4x4f,
  inv_view_proj : mat4x4f,
  cam_pos       : vec3f,
  time          : f32,
  domain        : vec3f,
  t_amb         : f32,
  h             : vec3f,
  contrast      : f32,
  n             : vec3u,
  flags         : u32,
  side_rect     : vec4f, // px: x, y, w, h of the side view
  side          : vec4f, // slice z (m), canvas w, canvas h, side aspect scale
  n_blades      : u32,
  max_drops     : u32,
  dt            : f32,
  emitter_count : u32,
  light         : vec4f,
  emitters      : array<vec4f, MAX_EMITTERS>, // xyz, kind (0 water, 1 air)
};

// One solid box exactly as src/volume/solids.ts rasterized it (packVolumeSolids).
struct VSolid {
  a : vec4f, // x0, top, z0, kind (0 house, 1 trunk)
  b : vec4f, // x1, roof height, z1, layout index
  c : vec4f, // roof apex x, roof half base, 0, 0
};

struct VTree {
  a : vec4f, // x, z, height (m), depth
  b : vec4f, // trunkTau, foliageTau, windAmp, phase
};

struct VScene {
  solids : array<VSolid, N_SOLIDS>,
  trees  : array<VTree, N_TREES>,
};

@group(0) @binding(0) var<uniform> S : SceneU;
@group(0) @binding(1) var<storage, read> vel : array<vec4f>;
@group(0) @binding(2) var<storage, read> tf : array<f32>;
@group(0) @binding(3) var<storage, read> solid : array<u32>;
@group(0) @binding(4) var<storage, read> wet : array<f32>;
@group(0) @binding(5) var<storage, read> state : array<f32>;
@group(0) @binding(6) var<storage, read> scene : VScene;
@group(0) @binding(7) var<storage, read> drops : array<VDroplet>;
@group(0) @binding(9) var<storage, read_write> state_rw : array<f32>;
@group(1) @binding(0) var depth_tex : texture_depth_2d;

fn dims() -> vec3i { return vec3i(S.n); }

fn sample_vel(pos: vec3f) -> vec3f {
  let n = dims();
  let t = tri_at(pos, S.h, n);
  var v = vec3f(0.0);
  for (var c = 0u; c < 8u; c++) { v += tri_weight(t, c) * vel[tri_corner(t, c, n)].xyz; }
  return v;
}

fn sample_t(pos: vec3f) -> f32 {
  let n = dims();
  let t = tri_at(pos, S.h, n);
  var v = 0.0;
  for (var c = 0u; c < 8u; c++) { v += tri_weight(t, c) * tf[tri_corner(t, c, n)]; }
  return v;
}

fn solid_cell(c: vec3i) -> bool {
  let n = dims();
  if (!vinside(c.x, c.y, c.z, n)) { return false; }
  return solid[vcell(c.x, c.y, c.z, n)] != 0u;
}

fn hash01(i: u32, k: u32) -> f32 {
  return f32(pcg(i * 7u + k + 0x9e3779b9u)) / 4294967296.0;
}

fn rgb(hex: u32) -> vec3f {
  return vec3f(f32((hex >> 16u) & 255u), f32((hex >> 8u) & 255u), f32(hex & 255u)) / 255.0;
}

fn has(f: u32) -> bool { return (S.flags & f) != 0u; }

fn lambert(nrm: vec3f) -> f32 {
  return 0.35 + 0.65 * max(0.0, dot(normalize(nrm), normalize(S.light.xyz)));
}

fn trunk_box(ti: u32) -> VSolid { return scene.solids[N_HOUSES + ti]; }

fn crown_center(ti: u32) -> vec3f {
  let td = scene.trees[ti];
  return vec3f(td.a.x, td.a.z * td.a.w * 0.72, td.a.y);
}

// --- 1. Tree sway -------------------------------------------------------------

fn low_pass(s: f32, goal: f32, tau: f32, dt: f32) -> f32 {
  return s + (goal - s) * (1.0 - exp(-dt / max(0.05, tau)));
}

@compute @workgroup_size(8)
fn scene_update(@builtin(global_invocation_id) gid: vec3u) {
  let ti = gid.x;
  if (ti >= N_TREES) { return; }
  let td = scene.trees[ti];
  let trunk = trunk_box(ti);
  let height = td.a.z * td.a.w;
  // Wind loading up the tree: trunk and lower branches near the ground (where
  // an outflow gust runs), the crown above the solid trunk. Sampled just
  // outside the bark on both x sides.
  var wind = vec2f(0.0);
  var wsum = 0.0;
  let heights = array<f32, 6>(0.25, 0.6, 1.0, 0.45 * height, 0.65 * height, 0.85 * height);
  let areas = array<f32, 6>(1.0, 0.8, 0.6, 1.0, 1.0, 0.8);
  for (var s = 0u; s < 6u; s++) {
    let side = 0.5 * (trunk.b.x - trunk.a.x) + S.h.x;
    let p = vec3f(0.5 * (trunk.a.x + trunk.b.x), heights[s], 0.5 * (trunk.a.z + trunk.b.z));
    let v = 0.5 * (sample_vel(p + vec3f(side, 0.0, 0.0)) + sample_vel(p - vec3f(side, 0.0, 0.0)));
    wind += areas[s] * v.xz;
    wsum += areas[s];
  }
  wind /= wsum;
  // Crown displacement: 1 m per m/s of loading times windAmp and depth. A
  // visual gain, as the slice's pixel gain is: a ground gust loads only the
  // lower samples, so the average stays small.
  let goal = wind * td.b.z * td.a.w;
  let k = ti * STATE_PER_TREE;
  state_rw[k] = low_pass(state_rw[k], goal.x * 0.42, td.b.x, S.dt);
  state_rw[k + 1u] = low_pass(state_rw[k + 1u], goal.y * 0.42, td.b.x, S.dt);
  state_rw[k + 2u] = low_pass(state_rw[k + 2u], goal.x * 1.15, td.b.y, S.dt);
  state_rw[k + 3u] = low_pass(state_rw[k + 3u], goal.y * 1.15, td.b.y, S.dt);
}

fn tree_sway(ti: u32, crown: bool) -> vec2f {
  let td = scene.trees[ti];
  let k = ti * STATE_PER_TREE + select(0u, 2u, crown);
  let idle = vec2f(sin(S.time * 1.35 + td.b.w), cos(S.time * 1.1 + td.b.w * 1.7)) * 0.02 * td.a.w;
  return vec2f(state[k], state[k + 1u]) + idle;
}

// --- 2a. Sky ------------------------------------------------------------------

struct FullOut {
  @builtin(position) pos : vec4f,
  @location(0) ndc : vec2f,
};

@vertex
fn vs_full(@builtin(vertex_index) vi: u32) -> FullOut {
  var pts = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(3.0, 1.0), vec2f(-1.0, 1.0));
  var o : FullOut;
  o.pos = vec4f(pts[vi], 0.0, 1.0);
  o.ndc = pts[vi];
  return o;
}

fn ray_dir(ndc: vec2f) -> vec3f {
  let far = S.inv_view_proj * vec4f(ndc, 1.0, 1.0);
  return normalize(far.xyz / far.w - S.cam_pos);
}

@fragment
fn fs_sky(in: FullOut) -> @location(0) vec4f {
  let d = ray_dir(in.ndc);
  let t = clamp(d.y * 1.6 + 0.15, 0.0, 1.0);
  let col = mix(rgb(0x334155u), mix(rgb(0x1e3a5fu), rgb(0x0f172au), clamp(t * 1.4 - 0.4, 0.0, 1.0)), t);
  return vec4f(col, 1.0);
}

// --- 2b. Ground sheet ---------------------------------------------------------

struct GroundOut {
  @builtin(position) pos : vec4f,
  @location(0) world : vec3f,
};

@vertex
fn vs_ground(@builtin(vertex_index) vi: u32) -> GroundOut {
  var c = array<vec2f, 6>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
                          vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0));
  let margin = vec2f(14.0);
  let xz = -margin + c[vi] * (S.domain.xz + 2.0 * margin);
  var o : GroundOut;
  o.world = vec3f(xz.x, 0.0, xz.y);
  o.pos = S.view_proj * vec4f(o.world, 1.0);
  return o;
}

fn sample_wet(xz: vec2f) -> f32 {
  let n = dims();
  let g = clamp(xz / S.h.xz - 0.5, vec2f(0.0), vec2f(f32(n.x - 1), f32(n.z - 1)) - 1.0e-4);
  let i0 = vec2i(floor(g));
  let f = g - vec2f(i0);
  let i1 = min(i0 + 1, vec2i(n.x - 1, n.z - 1));
  let w00 = wet[i0.y * n.x + i0.x];
  let w10 = wet[i0.y * n.x + i1.x];
  let w01 = wet[i1.y * n.x + i0.x];
  let w11 = wet[i1.y * n.x + i1.x];
  return mix(mix(w00, w10, f.x), mix(w01, w11, f.x), f.y);
}

@fragment
fn fs_ground(in: GroundOut) -> @location(0) vec4f {
  let p = in.world.xz;
  let inside = all(p >= vec2f(0.0)) && all(p <= S.domain.xz);
  var col = rgb(0x166534u);
  let stripe = 0.5 + 0.5 * sin(p.x * 1.7 + sin(p.y * 0.9) * 1.3);
  col = mix(col, rgb(0x14532du), stripe * 0.35);
  if (inside) {
    let w = sample_wet(p);
    col = mix(col, vec3f(0.03, 0.16, 0.22), clamp(w, 0.0, 1.0) * 0.85);
    // Bottom-layer wind: the sheet lightens where the gust runs.
    let v = sample_vel(vec3f(p.x, 0.5 * S.h.y, p.y));
    col += vec3f(0.05, 0.10, 0.06) * clamp(length(v.xz) / 1.5, 0.0, 1.0);
  } else {
    col *= 0.55; // beyond the simulated yard
  }
  // Yard edge line.
  let edge = min(min(abs(p.x), abs(p.x - S.domain.x)), min(abs(p.y), abs(p.y - S.domain.z)));
  col = mix(col, vec3f(0.55, 0.62, 0.7), (1.0 - smoothstep(0.0, 0.05, edge)) * 0.6);
  return vec4f(col * lambert(vec3f(0.0, 1.0, 0.0)), 1.0);
}

// --- 2c. Grass blades ---------------------------------------------------------

struct BladeOut {
  @builtin(position) pos : vec4f,
  @location(0) shade : vec3f,
};

const BLADE_VERTS : u32 = 5u; // triangle strip: base L/R, mid L/R, tip

@vertex
fn vs_grass(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> BladeOut {
  var o : BladeOut;
  o.shade = vec3f(0.0);
  o.pos = vec4f(2.0, 2.0, 2.0, 1.0);
  if (ii >= S.n_blades || !has(F_GRASS)) { return o; }
  // Jittered grid over the yard, aspect matched to its footprint.
  let bx = max(1u, u32(round(sqrt(f32(S.n_blades) * S.domain.x / S.domain.z))));
  let bz = max(1u, S.n_blades / bx);
  let gx = ii % bx;
  let gz = ii / bx;
  if (gz >= bz) { return o; }
  let cell = S.domain.xz / vec2f(f32(bx), f32(bz));
  let base_xz = (vec2f(f32(gx), f32(gz)) + vec2f(hash01(ii, 0u), hash01(ii, 1u))) * cell;
  let n = dims();
  let ci = vec3i(vec3f(base_xz.x, 0.0, base_xz.y) / S.h);
  if (solid_cell(vec3i(ci.x, 0, ci.z))) { return o; } // under a house or trunk

  let height = 0.18 + hash01(ii, 2u) * 0.3;
  let width = 0.025 + hash01(ii, 3u) * 0.02;
  // Bottom layer of the volume drives the bend.
  let v = sample_vel(vec3f(base_xz.x, 0.5 * S.h.y, base_xz.y)).xz;
  let phase = hash01(ii, 4u) * 6.28318;
  let idle = vec2f(sin(S.time * 2.1 + phase), cos(S.time * 1.7 + phase)) * 0.05;
  var bend = v * 0.55 + idle;
  let bl = length(bend);
  if (bl > 1.1) { bend *= 1.1 / bl; }

  let seg = vi / 2u;
  let side = select(-1.0, 1.0, (vi % 2u) == 1u);
  let t = select(f32(seg) * 0.5, 1.0, vi == 4u);
  let lean = bend * t * t * height;
  let y = height * t * (1.0 - 0.35 * length(bend) * t);
  let base = vec3f(base_xz.x, 0.0, base_xz.y);
  let to_cam = S.cam_pos.xz - base_xz;
  let across = normalize(vec2f(-to_cam.y, to_cam.x) + vec2f(1.0e-4, 0.0));
  let w = select(width * (1.0 - t), 0.0, vi == 4u);
  let p = base + vec3f(lean.x + across.x * side * w * 0.5, y, lean.y + across.y * side * w * 0.5);
  o.pos = S.view_proj * vec4f(p, 1.0);
  let wv = sample_wet(base_xz);
  o.shade = mix(mix(rgb(0x166534u), rgb(0x4ade80u), t * 0.8 + 0.1), vec3f(0.05, 0.3, 0.25), wv * 0.5);
  return o;
}

@fragment
fn fs_grass(in: BladeOut) -> @location(0) vec4f {
  return vec4f(in.shade * lambert(vec3f(0.2, 1.0, 0.1)), 1.0);
}

// --- 2d. Boxes: solid houses and trunks, plus the drawn upper trunks ---------

struct BoxOut {
  @builtin(position) pos : vec4f,
  @location(0) nrm : vec3f,
  @location(1) @interpolate(flat) color : vec3f,
};

// 36 box vertices (12 triangles) then 24 gable-roof vertices (8 triangles).
const BOX_VERTS : u32 = 60u;

fn box_corner(f: u32, c: u32) -> vec3f {
  // Face f: 0 -x, 1 +x, 2 -y, 3 +y, 4 -z, 5 +z; corner c of a two-triangle quad.
  var q = array<vec2f, 6>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0),
                          vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0));
  let uv = q[c];
  switch (f) {
    case 0u: { return vec3f(0.0, uv.x, uv.y); }
    case 1u: { return vec3f(1.0, uv.y, uv.x); }
    case 2u: { return vec3f(uv.y, 0.0, uv.x); }
    case 3u: { return vec3f(uv.x, 1.0, uv.y); }
    case 4u: { return vec3f(uv.y, uv.x, 0.0); }
    default: { return vec3f(uv.x, uv.y, 1.0); }
  }
}

fn face_normal(f: u32) -> vec3f {
  var nrm = array<vec3f, 6>(vec3f(-1.0, 0.0, 0.0), vec3f(1.0, 0.0, 0.0), vec3f(0.0, -1.0, 0.0),
                            vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, -1.0), vec3f(0.0, 0.0, 1.0));
  return nrm[f];
}

@vertex
fn vs_boxes(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> BoxOut {
  var o : BoxOut;
  o.pos = vec4f(2.0, 2.0, 2.0, 1.0);
  o.nrm = vec3f(0.0, 1.0, 0.0);
  o.color = vec3f(0.0);
  var lo : vec3f;
  var hi : vec3f;
  var top_shift = vec2f(0.0);
  var is_house = false;
  var roof = vec3f(0.0); // apex x, half base, height
  if (ii < N_SOLIDS) {
    let sd = scene.solids[ii];
    lo = vec3f(sd.a.x, 0.0, sd.a.z);
    hi = vec3f(sd.b.x, sd.a.y, sd.b.z);
    is_house = sd.a.w < 0.5;
    roof = vec3f(sd.c.x, sd.c.y, sd.b.y);
    o.color = select(rgb(0x3f2a1fu), rgb(0x334155u), is_house);
  } else {
    // Drawn upper trunk of tree t: from the solid top up into the crown, swaying.
    let ti = ii - N_SOLIDS;
    if (ti >= N_TREES) { return o; }
    let tb = trunk_box(ti);
    let shrink = 0.15 * (tb.b.x - tb.a.x);
    lo = vec3f(tb.a.x + shrink, tb.a.y, tb.a.z + shrink);
    hi = vec3f(tb.b.x - shrink, crown_center(ti).y, tb.b.z - shrink);
    top_shift = tree_sway(ti, false);
    o.color = rgb(0x4a3326u);
  }
  if (vi < 36u) {
    let f = vi / 6u;
    let u = box_corner(f, vi % 6u);
    var p = lo + u * (hi - lo);
    p = vec3f(p.x + top_shift.x * u.y, p.y, p.z + top_shift.y * u.y);
    o.nrm = face_normal(f);
    o.pos = S.view_proj * vec4f(p, 1.0);
    return o;
  }
  if (!is_house || roof.z <= 0.0) { return o; }
  // Gable roof along z: apex ridge at x = roof.x, eaves at roof.x ± roof.y.
  let r = vi - 36u;
  let tri = r / 3u;
  let c = r % 3u;
  let y0 = hi.y;
  let y1 = hi.y + roof.z;
  let xl = roof.x - roof.y;
  let xr = roof.x + roof.y;
  let ridge0 = vec3f(roof.x, y1, lo.z);
  let ridge1 = vec3f(roof.x, y1, hi.z);
  let l0 = vec3f(xl, y0, lo.z);
  let l1 = vec3f(xl, y0, hi.z);
  let r0 = vec3f(xr, y0, lo.z);
  let r1 = vec3f(xr, y0, hi.z);
  var tris = array<vec3f, 24>(
    l0, l1, ridge1, l0, ridge1, ridge0,       // left slope
    r0, ridge0, ridge1, r0, ridge1, r1,       // right slope
    l0, ridge0, r0, l1, r1, ridge1,           // gable ends
    l0, r0, r1, l0, r1, l1,                   // eave underside
  );
  let p = tris[tri * 3u + c];
  let a = tris[tri * 3u];
  let b = tris[tri * 3u + 1u];
  let d = tris[tri * 3u + 2u];
  o.nrm = normalize(cross(b - a, d - a));
  o.color = rgb(0x1e2937u);
  o.pos = S.view_proj * vec4f(p, 1.0);
  return o;
}

@fragment
fn fs_boxes(in: BoxOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  let nrm = select(-in.nrm, in.nrm, front);
  return vec4f(in.color * (0.25 + 0.75 * lambert(nrm)), 1.0);
}

// --- 2e. Sphere impostors: crowns and emitter markers -------------------------

struct BlobOut {
  @builtin(position) pos : vec4f,
  @location(0) uv : vec2f,
  @location(1) @interpolate(flat) center : vec3f,
  @location(2) @interpolate(flat) radius : f32,
  @location(3) @interpolate(flat) color : vec3f,
};

struct Blob { center : vec3f, radius : f32, color : vec3f };

fn blob_for(ii: u32) -> Blob {
  var b : Blob;
  b.radius = 0.0;
  if (ii < N_TREES * CROWN_BLOBS) {
    let ti = ii / CROWN_BLOBS;
    let k = ii % CROWN_BLOBS;
    let td = scene.trees[ti];
    let size = td.a.z * td.a.w;
    let c = crown_center(ti);
    let r_crown = 0.22 * size;
    let off = vec3f(hash01(ii, 10u) - 0.5, hash01(ii, 11u) - 0.35, hash01(ii, 12u) - 0.5) * r_crown * 1.5;
    let lift = (c.y + off.y) / max(c.y, 1.0e-3);
    let sway = tree_sway(ti, true) * lift;
    b.center = c + off + vec3f(sway.x, 0.0, sway.y);
    b.radius = r_crown * (0.55 + 0.3 * hash01(ii, 13u));
    let hero = select(0.0, 0.12, ti == N_TREES - 1u);
    b.color = mix(rgb(0x166534u), rgb(0x15803du), hash01(ii, 14u)) + vec3f(0.0, hero, 0.02);
    return b;
  }
  let ei = ii - N_TREES * CROWN_BLOBS;
  if (ei >= S.emitter_count || ei >= MAX_EMITTERS) { return b; }
  let em = S.emitters[ei];
  b.center = em.xyz;
  b.radius = 0.1;
  b.color = select(vec3f(0.2, 0.6, 1.0), vec3f(0.95, 0.95, 0.95), em.w > 0.5);
  return b;
}

@vertex
fn vs_blobs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> BlobOut {
  let b = blob_for(ii);
  var o : BlobOut;
  o.center = b.center;
  o.radius = b.radius;
  o.color = b.color;
  o.uv = vec2f(0.0);
  if (b.radius <= 0.0) { o.pos = vec4f(2.0, 2.0, 2.0, 1.0); return o; }
  var q = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
                          vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let to_cam = normalize(S.cam_pos - b.center);
  let right = normalize(cross(vec3f(0.0, 1.0, 0.0), to_cam) + vec3f(1.0e-5, 0.0, 0.0));
  let up = cross(to_cam, right);
  // Pull the card toward the camera so the front of the sphere is not clipped.
  let p = b.center + to_cam * b.radius + (right * q[vi].x + up * q[vi].y) * b.radius;
  o.uv = q[vi];
  o.pos = S.view_proj * vec4f(p, 1.0);
  return o;
}

struct BlobFrag {
  @location(0) color : vec4f,
  @builtin(frag_depth) depth : f32,
};

@fragment
fn fs_blobs(in: BlobOut) -> BlobFrag {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard; }
  let to_cam = normalize(S.cam_pos - in.center);
  let right = normalize(cross(vec3f(0.0, 1.0, 0.0), to_cam) + vec3f(1.0e-5, 0.0, 0.0));
  let up = cross(to_cam, right);
  let nrm = right * in.uv.x + up * in.uv.y + to_cam * sqrt(1.0 - r2);
  let p = in.center + nrm * in.radius;
  let clip = S.view_proj * vec4f(p, 1.0);
  var o : BlobFrag;
  o.color = vec4f(in.color * lambert(nrm), 1.0);
  o.depth = clamp(clip.z / clip.w, 0.0, 1.0);
  return o;
}

// --- 2f. Droplets ---------------------------------------------------------------

struct DropOut {
  @builtin(position) pos : vec4f,
  @location(0) uv : vec2f,
};

@vertex
fn vs_drops(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> DropOut {
  var o : DropOut;
  o.pos = vec4f(2.0, 2.0, 2.0, 1.0);
  o.uv = vec2f(0.0);
  if (ii >= S.max_drops || !has(F_DROPS)) { return o; }
  let d = drops[ii];
  if (d.vel_a.w <= 0.0) { return o; } // w: parcel multiplicity, 0 = dead
  var q = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
                          vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let c = S.view_proj * vec4f(d.pos_r.xyz, 1.0);
  // Constant pixel size, slightly larger for large drops.
  let px = 1.2 + clamp(d.pos_r.w / 4.0e-4, 0.0, 1.5);
  let off = q[vi] * px / vec2f(S.side.y, S.side.z) * 2.0 * c.w;
  o.pos = vec4f(c.xy + off, c.z, c.w);
  o.uv = q[vi];
  return o;
}

@fragment
fn fs_drops(in: DropOut) -> @location(0) vec4f {
  let r = length(in.uv);
  if (r > 1.0) { discard; }
  return vec4f(0.62, 0.84, 1.0, 0.55 * (1.0 - r * r));
}

// --- 3a. Temperature raymarch -------------------------------------------------

const MARCH_STEPS : u32 = 72u;

fn ramp_temp(t: f32) -> vec3f {
  // Same ramp as the slice overlay (render.wgsl).
  let cold = vec3f(0.20, 0.50, 0.98);
  let mid  = vec3f(0.055, 0.07, 0.11);
  let warm = vec3f(0.95, 0.42, 0.13);
  if (t < 0.0) { return mix(mid, cold, -t); }
  return mix(mid, warm, t);
}

@fragment
fn fs_volume(in: FullOut) -> @location(0) vec4f {
  if (!has(F_VOLUME)) { discard; }
  let dir = ray_dir(in.ndc);
  let o = S.cam_pos;
  // Slab test against the yard box.
  let inv = 1.0 / select(dir, vec3f(1.0e-6), abs(dir) < vec3f(1.0e-6));
  let ta = (vec3f(0.0) - o) * inv;
  let tb = (S.domain - o) * inv;
  let t0 = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), max(min(ta.z, tb.z), 0.0));
  let t1 = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z));
  if (t1 <= t0) { discard; }
  // Stop at the opaque scene.
  let px = vec2i(in.pos.xy);
  let d = textureLoad(depth_tex, px, 0);
  let wp = S.inv_view_proj * vec4f(in.ndc, d, 1.0);
  let t_scene = length(wp.xyz / wp.w - o);
  let t_end = min(t1, t_scene);
  if (t_end <= t0) { discard; }

  let dt = (t_end - t0) / f32(MARCH_STEPS);
  let jitter = hash01(u32(px.x) * 1973u + u32(px.y) * 9277u, 3u);
  var acc = vec4f(0.0);
  for (var s = 0u; s < MARCH_STEPS; s++) {
    let p = o + dir * (t0 + (f32(s) + jitter) * dt);
    let dev = (sample_t(p) - S.t_amb) / S.contrast;
    // Cold air reads strongly; warm spray plume faintly.
    let a = clamp(select(dev * 0.35, -dev, dev < 0.0), 0.0, 1.0);
    let density = a * a * dt * 1.6;
    let col = ramp_temp(clamp(dev, -1.0, 1.0)) * 1.5;
    let alpha = 1.0 - exp(-density);
    acc += vec4f(col * alpha, alpha) * (1.0 - acc.a);
    if (acc.a > 0.97) { break; }
  }
  if (acc.a <= 0.002) { discard; }
  return acc;
}

// --- 3b. Side view: the z slice, drawn like the slice app ---------------------

fn ramp_speed(t: f32) -> vec3f {
  let c0 = vec3f(0.045, 0.06, 0.10);
  let c1 = vec3f(0.05, 0.42, 0.52);
  let c2 = vec3f(0.95, 0.92, 0.65);
  if (t < 0.5) { return mix(c0, c1, t * 2.0); }
  return mix(c1, c2, (t - 0.5) * 2.0);
}

@fragment
fn fs_side(in: FullOut) -> @location(0) vec4f {
  let r = S.side_rect;
  var uv = (in.pos.xy - r.xy) / r.zw;
  uv.y = 1.0 - uv.y;
  // Letterbox to the domain's x:y aspect.
  let aspect = S.domain.x / S.domain.y;
  let box_aspect = r.z / r.w;
  var u = uv;
  if (box_aspect > aspect) {
    u.x = (uv.x - 0.5) * box_aspect / aspect + 0.5;
  } else {
    u.y = (uv.y - 0.5) * aspect / box_aspect + 0.5;
  }
  if (any(u < vec2f(0.0)) || any(u > vec2f(1.0))) { return vec4f(0.03, 0.04, 0.06, 1.0); }
  let world = vec3f(u.x * S.domain.x, u.y * S.domain.y, S.side.x);
  var col : vec3f;
  if (has(F_SIDE_SPEED)) {
    col = ramp_speed(clamp(length(sample_vel(world)) / 3.0, 0.0, 1.0));
  } else {
    col = ramp_temp(clamp((sample_t(world) - S.t_amb) / 3.0, -1.0, 1.0));
  }
  // Solid cells, cell-exact and hatched, as in the slice app.
  let c = vec3i(floor(world / S.h));
  if (solid_cell(c)) {
    let hatch = step(0.5, fract((world.x + world.y) / (S.h.x * 3.0)));
    col = mix(vec3f(0.16, 0.19, 0.25), vec3f(0.21, 0.25, 0.32), hatch);
  }
  // Thin frame.
  let e = min(min(u.x, 1.0 - u.x) * r.z, min(u.y, 1.0 - u.y) * r.w);
  col = mix(vec3f(0.35, 0.45, 0.55), col, smoothstep(0.0, 1.5, e));
  return vec4f(col, 1.0);
}
