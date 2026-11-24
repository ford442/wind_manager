@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read_write> vel : array<vec2f>;
@group(0) @binding(2) var<storage, read_write> tf : array<f32>;
@group(0) @binding(3) var<storage, read_write> qf : array<f32>;
@group(0) @binding(4) var<storage, read> solid : array<u32>;

fn is_solid(i: i32, j: i32, nx: i32, ny: i32) -> bool {
  if (i < 0 || j < 0 || i >= nx || j >= ny) { return false; }
  return solid[cell_index(i, j, nx)] != 0u;
}

@compute @workgroup_size(8, 8)
fn buoyancy(@builtin(global_invocation_id) gid: vec3u) {
  let nx = i32(P.nx);
  let ny = i32(P.ny);
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (i >= nx || j >= ny) { return; }
  let k = cell_index(i, j, nx);
  if (solid[k] != 0u) { return; }

  let tv = virt_temp(tf[k], qf[k]);
  let tv_amb = virt_temp(P.t_amb, P.q_amb);
  var v = vel[k];
  v.y += P.dt * G_ACC * (tv - tv_amb) / tv_amb;

  v *= 1.0 - P.damp * P.dt;
  vel[k] = v;
  tf[k] += (P.t_amb - tf[k]) * P.relax * P.dt;
  qf[k] += (P.q_amb - qf[k]) * P.relax * P.dt;
}

// Domain edges (open sides, open top, floor) and solid walls. Runs last in the
// substep. Edge cells copy from the next row in; solid cells copy T/q from
// their fluid neighbours. sim/solids.ts keeps every solid at least two cells
// from the side and top edges, so no cell reads a neighbour that another
// thread in this dispatch rewrites (interior wall-side cells change only
// their own velocity, which nobody else reads here).
@compute @workgroup_size(8, 8)
fn boundaries(@builtin(global_invocation_id) gid: vec3u) {
  let nx = i32(P.nx);
  let ny = i32(P.ny);
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (i >= nx || j >= ny) { return; }
  let k = cell_index(i, j, nx);

  if (solid[k] != 0u) {
    // Inside a solid: no velocity. T/q are the mean of the fluid 4-neighbours
    // (ambient if buried), so bilinear samplers next to a wall read air values.
    var t_sum = 0.0;
    var q_sum = 0.0;
    var cnt = 0.0;
    let nb = array<vec2i, 4>(vec2i(1, 0), vec2i(-1, 0), vec2i(0, 1), vec2i(0, -1));
    for (var n = 0; n < 4; n++) {
      let ii = i + nb[n].x;
      let jj = j + nb[n].y;
      if (ii < 0 || jj < 0 || ii >= nx || jj >= ny || is_solid(ii, jj, nx, ny)) { continue; }
      let kk = cell_index(ii, jj, nx);
      t_sum += tf[kk];
      q_sum += qf[kk];
      cnt += 1.0;
    }
    vel[k] = vec2f(0.0);
    if (cnt > 0.0) {
      tf[k] = t_sum / cnt;
      qf[k] = q_sum / cnt;
    } else {
      tf[k] = P.t_amb;
      qf[k] = P.q_amb;
    }
    return;
  }

  let wall_e = is_solid(i + 1, j, nx, ny);
  let wall_w = is_solid(i - 1, j, nx, ny);
  let wall_n = is_solid(i, j + 1, nx, ny);
  let wall_s = is_solid(i, j - 1, nx, ny);
  let edge = (i == 0) || (i == nx - 1) || (j == 0) || (j == ny - 1);
  if (!edge && !(wall_e || wall_w || wall_n || wall_s)) { return; }

  var v = vel[k];
  var t = tf[k];
  var q = qf[k];

  if (i == 0) {
    let kk = cell_index(1, j, nx);
    v = vel[kk];
    t = tf[kk];
    q = qf[kk];
    v.x *= 0.92;
  } else if (i == nx - 1) {
    let kk = cell_index(nx - 2, j, nx);
    v = vel[kk];
    t = tf[kk];
    q = qf[kk];
    v.x *= 0.92;
  }
  if (j == ny - 1) {
    let kk = cell_index(i, ny - 2, nx);
    v = vel[kk];
    t = mix(tf[kk], P.t_amb, 0.5);
    q = mix(qf[kk], P.q_amb, 0.5);
  }
  if (j == 0) {
    v.y = 0.0;
  }

  // Solid faces are walls: drop the velocity component that points into one.
  if (wall_e) { v.x = min(v.x, 0.0); }
  if (wall_w) { v.x = max(v.x, 0.0); }
  if (wall_n) { v.y = min(v.y, 0.0); }
  if (wall_s) { v.y = max(v.y, 0.0); }

  vel[k] = v;
  tf[k] = t;
  qf[k] = q;
}
