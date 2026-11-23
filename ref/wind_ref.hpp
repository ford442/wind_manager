// wind_ref: CPU reference for the scalar formulas the README documents.
//
// This is NOT a second simulator. The solver lives in src/shaders/*.wgsl and runs
// on the GPU; this library only restates the handful of closed-form scalars so CI
// can pin them to known numbers without a GPU:
//
//   q_sat       saturation mixing ratio          common.wgsl `q_sat`, params.ts `qSat`
//   virt_temp   virtual temperature              common.wgsl `virt_temp`
//   d2_step     one d^2 radius step + dm         droplets.wgsl `update`
//   latent_dT   latent-heat temperature change   droplets.wgsl `update`
//
// Double precision here, f32 on the GPU: agreement is to ~1e-6 relative, which is
// far inside the golden tolerance used by wind_ref_test.cpp.
//
// The constants below are textually mirrored from src/shaders/common.wgsl.
// scripts/check-ref-parity.mjs fails CI if they (or the pinned formulas) drift.
#pragma once

namespace wind_ref {

// --- Mirrored from common.wgsl (keep `constexpr double NAME = value;` on one line) ---
constexpr double PI = 3.14159265359;  // truncated exactly as in WGSL; not M_PI
constexpr double L_VAP = 2.45e6;      // J/kg, latent heat of vaporisation
constexpr double CP_AIR = 1005.0;     // J/(kg K)
constexpr double RHO_AIR = 1.2;       // kg/m^3
constexpr double RHO_W = 1000.0;      // kg/m^3
constexpr double NU_AIR = 1.5e-5;     // m^2/s, kinematic viscosity
constexpr double DV_H2O = 2.5e-5;     // m^2/s, vapour diffusivity
constexpr double SC_AIR = 0.6;        // Schmidt number

// Saturation mixing ratio [kg/kg] at t_c [deg C], 1013.25 hPa (Magnus form).
// Like params.ts (not common.wgsl) returns 0 at/below -243 C, where the formula is singular.
double q_sat(double t_c);

// Virtual temperature [K] from t_c [deg C] and mixing ratio q [kg/kg].
double virt_temp(double t_c, double q);

// Result of one d^2 step for a single droplet.
struct D2Step {
  double re;       // Reynolds number 2 r |u_rel| / nu
  double sh;       // Sherwood number 2 + 0.6 sqrt(Re) Sc^(1/3)
  double deficit;  // max(0, q_sat(T) - q) [kg/kg]
  double dr2;      // d(r^2)/dt = -(2 D sh rho_air / rho_w) * deficit [m^2/s]
  double r_new;    // sqrt(max(0, r^2 + dr2 dt)) [m]
  double dm;       // rho_w 4/3 pi (r^3 - r_new^3), clamped >= 0 [kg]
};

// One explicit d^2 step. `radius` [m] (the sim stores radius, not diameter),
// `rel_speed` = |u_air - u_drop| [m/s], `t_c` and `q` are the air at the drop,
// `dt` [s] is the substep (params.dt).
D2Step d2_step(double radius, double rel_speed, double t_c, double q, double dt);

// Air mass of one grid cell per unit depth [kg]: rho_air * h * h.
double cell_mass(double h);

// Temperature change [K] of a cell that receives evaporated mass dm [kg]:
//   dT = -L_v dm / (m_cell c_p)  when latent heat is on, exactly 0 when off.
double latent_dT(double dm, double m_cell, bool latent_on);

}  // namespace wind_ref
