// common.wgsl — shared constants, structs and helpers.
// This file is prepended to every other shader at pipeline-creation time
// (WGSL has no #include). Keep it free of @group/@binding declarations.

const PI      : f32 = 3.14159265359;
const G_ACC   : f32 = 9.81;       // m/s^2
const L_VAP   : f32 = 2.45e6;     // J/kg, latent heat of vaporization
const CP_AIR  : f32 = 1005.0;     // J/(kg K)
const RHO_AIR : f32 = 1.2;        // kg/m^3
const RHO_W   : f32 = 1000.0;     // kg/m^3
const MU_AIR  : f32 = 1.8e-5;     // Pa s, dynamic viscosity
const NU_AIR  : f32 = 1.5e-5;     // m^2/s, kinematic viscosity
const DV_H2O  : f32 = 2.5e-5;     // m^2/s, vapor diffusivity
const SC_AIR  : f32 = 0.6;        // Schmidt number
const R_KILL  : f32 = 2.0e-6;     // m, droplets below this radius die

// Fixed-point scatter scales. Chosen so a typical per-droplet per-step deposit
// is many integer counts (a 50 um droplet evaporating over ~1 s contributes
// ~4e-9 kg/kg of vapor per substep -> ~40 counts at 1e10), while worst-case
// per-cell sums stay orders of magnitude below the i32 limit of ~2.1e9.
const SCALE_Q : f32 = 1.0e10;     // specific humidity, kg/kg
const SCALE_T : f32 = 1.0e7;      // temperature, K
const SCALE_M : f32 = 1.0e8;      // velocity, m/s

struct Params {
  nx          : u32,
  ny          : u32,
  h           : f32,   // cell size, m
  dt          : f32,   // substep, s
  t_amb       : f32,   // ambient temperature, degC
  q_amb       : f32,   // ambient specific humidity, kg/kg
  latent_on   : f32,   // 1.0 = deposit latent cooling, 0.0 = skip (toggle)
  time        : f32,   // sim time, s
  emit_x      : f32,   // emitter position, m
  emit_y      : f32,
  emit_angle  : f32,   // rad from +y (up)
  emit_spread : f32,   // rad, half-angle of spray cone
  emit_speed  : f32,   // m/s
  emit_count  : u32,   // droplets to spawn this substep
  r_min       : f32,   // m, smallest emitted radius
  r_max       : f32,   // m, largest emitted radius
  max_droplets: u32,
  seed        : u32,
  relax       : f32,   // 1/s, relaxation of T,q toward ambient
  damp        : f32,   // 1/s, velocity damping
};

struct Droplet {
  pos    : vec2f,
  vel    : vec2f,
  radius : f32,
  alive  : u32,
};

struct Bilin {
  i0 : i32, j0 : i32,
  i1 : i32, j1 : i32,
  fx : f32, fy : f32,
};

fn cell_index(i: i32, j: i32, nx: i32) -> i32 {
  return j * nx + i;
}

// Saturation specific humidity (Magnus/Tetens), T in degC.
fn q_sat(t_c: f32) -> f32 {
  let es = 610.94 * exp(17.625 * t_c / (t_c + 243.04)); // Pa
  let p  = 101325.0;                                    // Pa
  return 0.622 * es / (p - 0.378 * es);
}

// Virtual temperature in K: moist air is lighter than dry air at the same T.
fn virt_temp(t_c: f32, q: f32) -> f32 {
  return (t_c + 273.15) * (1.0 + 0.61 * q);
}

// PCG hash for per-thread randomness.
fn pcg(v: u32) -> u32 {
  var s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

fn rand01(seed: ptr<function, u32>) -> f32 {
  *seed = pcg(*seed);
  return f32(*seed) / 4294967296.0;
}

// Cell-centered bilinear weights for a world position, clamped to the grid.
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
