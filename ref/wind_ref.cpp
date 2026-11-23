#include "wind_ref.hpp"

#include <algorithm>
#include <cmath>

namespace wind_ref {

double q_sat(double t_c) {
  if (t_c <= -243.0) return 0.0;
  const double es = 610.94 * std::exp(17.625 * t_c / (t_c + 243.04));
  const double p = 101325.0;
  return 0.622 * es / (p - 0.378 * es);
}

double virt_temp(double t_c, double q) {
  return (t_c + 273.15) * (1.0 + 0.61 * q);
}

D2Step d2_step(double radius, double rel_speed, double t_c, double q, double dt) {
  D2Step s{};
  s.re = 2.0 * radius * rel_speed / NU_AIR;
  s.sh = 2.0 + 0.6 * std::sqrt(s.re) * std::pow(SC_AIR, 1.0 / 3.0);
  s.deficit = std::max(0.0, q_sat(t_c) - q);
  s.dr2 = -(2.0 * DV_H2O * s.sh * RHO_AIR / RHO_W) * s.deficit;
  s.r_new = std::sqrt(std::max(0.0, radius * radius + s.dr2 * dt));
  s.dm = RHO_W * (4.0 / 3.0) * PI *
         std::max(0.0, radius * radius * radius - s.r_new * s.r_new * s.r_new);
  return s;
}

double cell_mass(double h) {
  return RHO_AIR * h * h;
}

double latent_dT(double dm, double m_cell, bool latent_on) {
  if (!latent_on) return 0.0;
  return -(L_VAP * dm) / (m_cell * CP_AIR);
}

}  // namespace wind_ref
