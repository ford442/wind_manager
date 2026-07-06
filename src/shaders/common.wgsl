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

struct Params {
  nx          : u32,
  ny          : u32,
  h           : f32,
  dt          : f32,
  t_amb       : f32,
  q_amb       : f32,
  latent_on   : f32,
  time        : f32,
  emit_x      : f32,
  emit_y      : f32,
  emit_angle  : f32,
  emit_spread : f32,
  emit_speed  : f32,
  emit_count  : u32,
  emit_type   : u32,
  r_min       : f32,
  r_max       : f32,
  max_droplets: u32,
  seed        : u32,
  relax       : f32,
  damp        : f32,
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

fn sample_vel(pos: vec2f, h: f32, nx: i32, ny: i32, vel: ptr<storage, array<vec2f>, read>) -> vec2f {
  let b = bilin_at(pos, h, nx, ny);
  let k00 = cell_index(b.i0, b.j0, nx);
  let k10 = cell_index(b.i1, b.j0, nx);
  let k01 = cell_index(b.i0, b.j1, nx);
  let k11 = cell_index(b.i1, b.j1, nx);
  return mix(
    mix((*vel)[k00], (*vel)[k10], b.fx),
    mix((*vel)[k01], (*vel)[k11], b.fx),
    b.fy,
  );
}
