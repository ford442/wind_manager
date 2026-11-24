// common.wgsl — shared constants, structs and helpers.
// Prepended to every other shader at pipeline-creation time.

const PI      : f32 = 3.14159265359;
const G_ACC   : f32 = 9.81;
const L_VAP   : f32 = 2.45e6;
const CP_AIR  : f32 = 1005.0;
const RHO_AIR : f32 = 1.2;
const RHO_W   : f32 = 1000.0;
const MU_AIR  : f32 = 1.8e-5;
const NU_AIR  : f32 = 1.5e-5;
const DV_H2O  : f32 = 2.5e-5;
const SC_AIR  : f32 = 0.6;
const R_KILL  : f32 = 2.0e-6;

const SCALE_Q   : f32 = 1.0e10;
const SCALE_T   : f32 = 1.0e7;
const SCALE_M   : f32 = 1.0e8;
const SCALE_WET : f32 = 1.0e8;

const GROUND_SPLASH : f32 = 0.22;
const GROUND_TEMP   : f32 = 0.12;
const WET_DECAY     : f32 = 0.85;
const QDEP_DECAY    : f32 = 0.06;
const QDEP_INTEG    : f32 = 3.2;

// Solid obstacles (src/sim/solids.ts). The mask holds 0 for fluid and
// `id + 1` inside solid `id`; houses first, then tree trunks. Wall wetness
// lives after the `nx` ground columns of `wet`: slot `nx + id * 2 + face`,
// face 0 = left wall, 1 = right wall.
const N_SOLIDS      : u32 = 8u;
const N_SOLID_FACES : u32 = 2u;
// Share of an impacting drop's along-wall velocity shed into the air (the
// wall counterpart of GROUND_SPLASH), and of wall water that runs down to wet
// the ground column at the foot of the wall.
const WALL_SPLASH : f32 = 0.3;
const WALL_DRAIN  : f32 = 0.35;

struct Params {
  // Global per-substep values. Layout: SIM_PARAMS_LAYOUT in
  // src/sim/simParamsUniform.ts (64 bytes). Per-emitter data lives in
  // `EmitterParams` (droplets.wgsl), not here.
  nx           : u32,
  ny           : u32,
  h            : f32,
  dt           : f32,
  t_amb        : f32,
  q_amb        : f32,
  latent_on    : f32,
  time         : f32,
  max_droplets : u32,
  seed         : u32,
  relax        : f32,
  damp         : f32,
  emitter_count: u32,
  _pad0        : u32,
  _pad1        : u32,
  _pad2        : u32,
};

struct Droplet {
  pos    : vec2f,
  vel    : vec2f,
  radius : f32,
  alive  : u32,
};

struct Tracer {
  pos       : vec2f,
  prev_pos  : vec2f,
  age       : f32,
  alive     : u32,
  tint      : f32,
  _pad      : f32,
};

struct Bilin {
  i0 : i32, j0 : i32,
  i1 : i32, j1 : i32,
  fx : f32, fy : f32,
};

fn cell_index(i: i32, j: i32, nx: i32) -> i32 {
  return j * nx + i;
}

fn solid_face_slot(nx: i32, id: u32, face: u32) -> i32 {
  return nx + i32(id * N_SOLID_FACES + face);
}

// Bilinear corner weights (w00, w10, w01, w11) with solid corners dropped and
// the rest renormalized. `fluid` holds 1.0 per fluid corner, same order. All
// zero when every corner is solid. Callers read the mask from their own binding.
fn fluid_weights(b: Bilin, fluid: vec4f) -> vec4f {
  let w = vec4f(
    (1.0 - b.fx) * (1.0 - b.fy), b.fx * (1.0 - b.fy),
    (1.0 - b.fx) * b.fy, b.fx * b.fy,
  ) * fluid;
  let s = w.x + w.y + w.z + w.w;
  if (s > 1.0e-6) { return w / s; }
  // Only corners with ~zero weight are fluid: split evenly among them.
  let c = fluid.x + fluid.y + fluid.z + fluid.w;
  if (c > 0.0) { return fluid / c; }
  return vec4f(0.0);
}

fn q_sat(t_c: f32) -> f32 {
  let es = 610.94 * exp(17.625 * t_c / (t_c + 243.04));
  let p  = 101325.0;
  return 0.622 * es / (p - 0.378 * es);
}

fn virt_temp(t_c: f32, q: f32) -> f32 {
  return (t_c + 273.15) * (1.0 + 0.61 * q);
}

fn pcg(v: u32) -> u32 {
  var s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

fn rand01(seed: ptr<function, u32>) -> f32 {
  *seed = pcg(*seed);
  return f32(*seed) / 4294967296.0;
}

fn bilin_at(pos: vec2f, h: f32, nx: i32, ny: i32) -> Bilin {
  let gx = clamp(pos.x / h - 0.5, 0.0, f32(nx - 1) - 1.0e-4);
  let gy = clamp(pos.y / h - 0.5, 0.0, f32(ny - 1) - 1.0e-4);
  let i0 = i32(floor(gx));
  let j0 = i32(floor(gy));
  var b : Bilin;
  b.i0 = i0;
  b.j0 = j0;
  b.i1 = min(i0 + 1, nx - 1);
  b.j1 = min(j0 + 1, ny - 1);
  b.fx = gx - f32(i0);
  b.fy = gy - f32(j0);
  return b;
}

// Bilinear blend of four corner samples with a Bilin's weights. Callers load the
// corners from their own `vel` binding: a helper taking `ptr<storage, ...>` needs
// the WGSL `unrestricted_pointer_parameters` extension, which Naga (CI validator,
// Firefox) does not implement.
fn bilin_blend(b: Bilin, v00: vec2f, v10: vec2f, v01: vec2f, v11: vec2f) -> vec2f {
  return mix(mix(v00, v10, b.fx), mix(v01, v11, b.fx), b.fy);
}
