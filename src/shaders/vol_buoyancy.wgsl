// vol_buoyancy.wgsl — virtual-temperature buoyancy and the edge / wall pass
// (buoyancy.wgsl in 3D).

@group(0) @binding(0) var<uniform> P : VParams;
@group(0) @binding(1) var<storage, read_write> vel : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> tf : array<f32>;
@group(0) @binding(3) var<storage, read_write> qf : array<f32>;
@group(0) @binding(4) var<storage, read> solid : array<u32>;

fn dims() -> vec3i { return vec3i(i32(P.nx), i32(P.ny), i32(P.nz)); }

fn is_solid(c: vec3i) -> bool {
  let n = dims();
  if (!vinside(c.x, c.y, c.z, n)) { return false; }
  return solid[vcell(c.x, c.y, c.z, n)] != 0u;
}

fn tv_anom(i: i32, j: i32, k: i32) -> f32 {
  let n = dims();
  let jj = clamp(j, 0, n.y - 1);
  let kk = vcell(i, jj, k, n);
  let tv_amb = virt_temp(P.t_amb, P.q_amb);
  return (virt_temp(tf[kk], qf[kk]) - tv_amb) / tv_amb;
}

// Buoyancy on the collocated grid, weighted [1/4, 1/2, 1/4] in y. The centred
// gradient makes p(j-1) - p(j+1) the integral of the buoyancy over
// [y(j-1), y(j+1)]: half of each neighbour and all of cell j. Without the
// weights the floor row's buoyancy (where spray cools the air most) never
// reaches the pressure, and a cold pool on the ground cannot push outward.
// Reads T/q only (relaxation is in vol_advect), so no thread races another.
@compute @workgroup_size(4, 4, 4)
fn vol_buoyancy(@builtin(global_invocation_id) gid: vec3u) {
  let n = dims();
  let c = vec3i(gid);
  if (!vinside(c.x, c.y, c.z, n)) { return; }
  let k = vcell(c.x, c.y, c.z, n);
  if (solid[k] != 0u) { return; }
  let b = 0.25 * tv_anom(c.x, c.y - 1, c.z) + 0.5 * tv_anom(c.x, c.y, c.z) + 0.25 * tv_anom(c.x, c.y + 1, c.z);
  var v = vel[k].xyz;
  v.y += P.dt * G_ACC * b;
  v *= 1.0 - P.damp * P.dt;
  vel[k] = vec4f(v, 0.0);
}

// Domain edges (open x and z sides, open top, floor) and solid walls, before
// the projection. An edge cell copies the nearest cell that is not on any
// edge, so no thread reads a cell another thread rewrites: src/volume/solids.ts
// keeps every solid three cells clear of the sides and top, so those interior
// source cells are never wall-adjacent either.
@compute @workgroup_size(4, 4, 4)
fn vol_boundaries(@builtin(global_invocation_id) gid: vec3u) {
  let n = dims();
  let c = vec3i(gid);
  if (!vinside(c.x, c.y, c.z, n)) { return; }
  let k = vcell(c.x, c.y, c.z, n);

  if (solid[k] != 0u) {
    // Inside a solid: no velocity; T/q are the mean of the fluid 6-neighbours
    // (ambient if buried) so trilinear samplers next to a wall read air.
    var t_sum = 0.0;
    var q_sum = 0.0;
    var cnt = 0.0;
    let nb = array<vec3i, 6>(
      vec3i(1, 0, 0), vec3i(-1, 0, 0), vec3i(0, 1, 0),
      vec3i(0, -1, 0), vec3i(0, 0, 1), vec3i(0, 0, -1),
    );
    for (var d = 0; d < 6; d++) {
      let o = c + nb[d];
      if (!vinside(o.x, o.y, o.z, n) || is_solid(o)) { continue; }
      let kk = vcell(o.x, o.y, o.z, n);
      t_sum += tf[kk];
      q_sum += qf[kk];
      cnt += 1.0;
    }
    vel[k] = vec4f(0.0);
    if (cnt > 0.0) {
      tf[k] = t_sum / cnt;
      qf[k] = q_sum / cnt;
    } else {
      tf[k] = P.t_amb;
      qf[k] = P.q_amb;
    }
    return;
  }

  let wall_xp = is_solid(c + vec3i(1, 0, 0));
  let wall_xm = is_solid(c - vec3i(1, 0, 0));
  let wall_yp = is_solid(c + vec3i(0, 1, 0));
  let wall_ym = is_solid(c - vec3i(0, 1, 0));
  let wall_zp = is_solid(c + vec3i(0, 0, 1));
  let wall_zm = is_solid(c - vec3i(0, 0, 1));
  let side_x = c.x == 0 || c.x == n.x - 1;
  let side_z = c.z == 0 || c.z == n.z - 1;
  let top = c.y == n.y - 1;
  let floor_cell = c.y == 0;
  let walls = wall_xp || wall_xm || wall_yp || wall_ym || wall_zp || wall_zm;
  if (!(side_x || side_z || top || floor_cell || walls)) { return; }

  var v = vel[k].xyz;
  var t = tf[k];
  var q = qf[k];

  if (side_x || side_z || top) {
    // Open edge: zero-gradient from the nearest non-edge cell.
    let src = vec3i(
      clamp(c.x, 1, n.x - 2),
      select(c.y, n.y - 2, top),
      clamp(c.z, 1, n.z - 2),
    );
    let kk = vcell(src.x, src.y, src.z, n);
    v = vel[kk].xyz;
    t = tf[kk];
    q = qf[kk];
    if (side_x) { v.x *= 0.92; }
    if (side_z) { v.z *= 0.92; }
    if (top) {
      t = mix(t, P.t_amb, 0.5);
      q = mix(q, P.q_amb, 0.5);
    }
  }
  if (floor_cell) { v.y = 0.0; }

  // Solid faces are walls: drop the velocity component pointing into one.
  if (wall_xp) { v.x = min(v.x, 0.0); }
  if (wall_xm) { v.x = max(v.x, 0.0); }
  if (wall_yp) { v.y = min(v.y, 0.0); }
  if (wall_ym) { v.y = max(v.y, 0.0); }
  if (wall_zp) { v.z = min(v.z, 0.0); }
  if (wall_zm) { v.z = max(v.z, 0.0); }

  vel[k] = vec4f(v, 0.0);
  tf[k] = t;
  qf[k] = q;
}
