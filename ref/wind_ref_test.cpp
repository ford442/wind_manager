// Golden-value test for the wind_ref scalars. Exit status is the number of failures
// (capped at 255), so `ctest` / CI go red the moment any number drifts.
//
// Provenance: every `want` below was derived independently of this library, with a
// 50-digit evaluation (Python mpmath) of the formulas in README.md and
// src/shaders/{common,droplets}.wgsl, then rounded to 17 significant digits. To
// change one deliberately, re-derive it the same way; do not paste this library's
// output back in, or the test stops checking anything.
//
// Scenario ("dry hot day"): ambient 30 C at 20 % RH, so
//   q_amb = 0.2 * q_sat(30 C) = 5.2850035053674944e-3 kg/kg.
// Drop: radius 100 um (the sim stores radius; emitters are 50-800 um).
// Grid: 32 m / 512 cells => h = 0.0625 m, m_cell = 1.2 * h^2 = 4.6875e-3 kg.
// Substep: dt = 1/120 s (params.dt; substeps repeat it, they do not divide it).
//
// Tolerance: 1e-9 relative. Double-precision rounding is ~1e-13 here (worst case
// is the r^3 - r_new^3 cancellation), while the smallest plausible constant typo
// (a transposed digit, e.g. 2.45e6 -> 2.54e6, or 0.622 -> 0.662) moves results by
// >= 1e-2. Do not loosen this to make a failure go away: find the drift.

#include <cmath>
#include <cstdio>

#include "wind_ref.hpp"

namespace {

using namespace wind_ref;

constexpr double REL_TOL = 1e-9;

int g_failures = 0;
int g_checks = 0;

void close(const char* name, double got, double want) {
  ++g_checks;
  const double err = std::fabs(got - want);
  const double rel = want != 0.0 ? err / std::fabs(want) : err;
  const bool ok = rel <= REL_TOL;
  if (!ok) ++g_failures;
  std::printf("%s %-34s got %.17g want %.17g (rel err %.2e)\n", ok ? "ok  " : "FAIL", name, got,
              want, rel);
}

void exactly(const char* name, double got, double want) {
  ++g_checks;
  const bool ok = got == want;
  if (!ok) ++g_failures;
  std::printf("%s %-34s got %.17g want %.17g (exact)\n", ok ? "ok  " : "FAIL", name, got, want);
}

}  // namespace

int main() {
  // --- q_sat -------------------------------------------------------------
  close("q_sat(-10 C)", q_sat(-10.0), 1.7622878245302041e-3);
  close("q_sat(0 C)", q_sat(0.0), 3.7589217646244116e-3);
  close("q_sat(20 C)", q_sat(20.0), 1.4449993066664177e-2);
  close("q_sat(30 C)", q_sat(30.0), 2.6425017526837472e-2);
  exactly("q_sat(-243 C) guard", q_sat(-243.0), 0.0);

  const double q_amb = 0.2 * q_sat(30.0);
  close("q_amb = 0.2 q_sat(30 C)", q_amb, 5.2850035053674944e-3);

  // --- virtual temperature -------------------------------------------------
  close("virt_temp(30 C, q=0)", virt_temp(30.0, 0.0), 303.15);
  close("virt_temp(30 C, q_amb)", virt_temp(30.0, q_amb), 304.12731077571782);
  close("virt_temp(30 C, q_sat)", virt_temp(30.0, q_sat(30.0)), 308.03655387858908);

  // --- cell mass -------------------------------------------------------------
  const double h = 32.0 / 512.0;
  const double m_cell = cell_mass(h);
  close("cell_mass(h=0.0625)", m_cell, 4.6875e-3);

  const double dt = 1.0 / 120.0;

  // --- d^2 step: 100 um drop in still air (Re = 0, Sh = 2) --------------------
  const D2Step still = d2_step(100e-6, 0.0, 30.0, q_amb, dt);
  exactly("still: Re", still.re, 0.0);
  exactly("still: Sh", still.sh, 2.0);
  close("still: deficit", still.deficit, 2.1140014021469978e-2);
  close("still: dr2/dt", still.dr2, -2.5368016825763973e-9);
  close("still: r_new", still.r_new, 9.9894244008243688e-5);
  close("still: dm", still.dm, 1.3275640182208212e-11);
  close("still: dT latent ON", latent_dT(still.dm, m_cell, true), -6.9042135342296769e-6);
  exactly("still: dT latent OFF", latent_dT(still.dm, m_cell, false), 0.0);

  // --- d^2 step: 1 m/s relative air speed (Sherwood correction active) --------
  const D2Step moving = d2_step(100e-6, 1.0, 30.0, q_amb, dt);
  close("1 m/s: Re", moving.re, 13.333333333333333);
  close("1 m/s: Sh", moving.sh, 3.8478683860898915);
  close("1 m/s: dr2/dt", moving.dr2, -4.8806394980826816e-9);
  close("1 m/s: r_new", moving.r_new, 9.9796432822935331e-5);
  close("1 m/s: dm", moving.dm, 2.5528966723073722e-11);
  close("1 m/s: dT latent ON", latent_dT(moving.dm, m_cell, true), -1.3276756159794227e-5);
  exactly("1 m/s: dT latent OFF", latent_dT(moving.dm, m_cell, false), 0.0);

  // --- d^2 step: dt long enough to clamp r_new to 0 (whole drop evaporates) ---
  const D2Step gone = d2_step(100e-6, 0.0, 30.0, q_amb, 10.0);
  exactly("clamp: r_new", gone.r_new, 0.0);
  close("clamp: dm = whole drop", gone.dm, 4.1887902047866667e-9);
  close("clamp: dT latent ON", latent_dT(gone.dm, m_cell, true), -2.1784487698525683e-3);
  exactly("clamp: dT latent OFF", latent_dT(gone.dm, m_cell, false), 0.0);

  // --- no evaporation without a humidity deficit ------------------------------
  const D2Step sat = d2_step(100e-6, 0.0, 30.0, q_sat(30.0), dt);
  exactly("saturated: deficit", sat.deficit, 0.0);
  exactly("saturated: r_new == r", sat.r_new, 100e-6);
  exactly("saturated: dm", sat.dm, 0.0);
  const D2Step super = d2_step(100e-6, 0.0, 30.0, 2.0 * q_sat(30.0), dt);
  exactly("supersaturated: dm (no growth)", super.dm, 0.0);

  std::printf("\n%d/%d checks passed\n", g_checks - g_failures, g_checks);
  return g_failures > 255 ? 255 : g_failures;
}
