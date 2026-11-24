@group(0) @binding(0) var<uniform> P : Params;
@group(0) @binding(1) var<storage, read_write> vel : array<vec2f>;
@group(0) @binding(2) var<storage, read_write> div : array<f32>;
@group(0) @binding(3) var<storage, read> p_in : array<f32>;
@group(0) @binding(4) var<storage, read_write> p_out : array<f32>;
@group(0) @binding(5) var<storage, read> solid : array<u32>;

fn is_solid(i: i32, j: i32, nx: i32, ny: i32) -> bool {
  if (i < 0 || j < 0 || i >= nx || j >= ny) { return false; }
  return solid[cell_index(i, j, nx)] != 0u;
}

// Neighbour velocity for the divergence stencil. Solid faces are walls with
// zero velocity; the domain edge copies the edge cell (open sides).
fn vel_at(i: i32, j: i32, nx: i32, ny: i32) -> vec2f {
  if (is_solid(i, j, nx, ny)) { return vec2f(0.0); }
  return vel[cell_index(clamp(i, 0, nx - 1), clamp(j, 0, ny - 1), nx)];
}

// Neighbour pressure. Open top: p = 0 (Dirichlet). Sides, floor and solid
// faces: Neumann, the neighbour takes the centre value, so no pressure force
// acts across the face and cells inside a solid never enter the stencil.
fn p_at(i: i32, j: i32, k_center: i32, nx: i32, ny: i32) -> f32 {
  if (j >= ny) { return 0.0; }
  if (i < 0 || i >= nx || j < 0) { return p_in[k_center]; }
  let k = cell_index(i, j, nx);
  if (solid[k] != 0u) { return p_in[k_center]; }
  return p_in[k];
}

@compute @workgroup_size(8, 8)
fn divergence(@builtin(global_invocation_id) gid: vec3u) {
  let nx = i32(P.nx);
  let ny = i32(P.ny);
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (i >= nx || j >= ny) { return; }
  let k = cell_index(i, j, nx);
  p_out[k] = 0.0;
  if (solid[k] != 0u) {
    div[k] = 0.0;
    return;
  }

  div[k] = -0.5 * P.h * ((vel_at(i + 1, j, nx, ny).x - vel_at(i - 1, j, nx, ny).x)
                       + (vel_at(i, j + 1, nx, ny).y - vel_at(i, j - 1, nx, ny).y));
}

@compute @workgroup_size(8, 8)
fn jacobi(@builtin(global_invocation_id) gid: vec3u) {
  let nx = i32(P.nx);
  let ny = i32(P.ny);
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (i >= nx || j >= ny) { return; }
  let k = cell_index(i, j, nx);
  // Cells inside a solid are not unknowns.
  if (solid[k] != 0u) {
    p_out[k] = 0.0;
    return;
  }

  p_out[k] = (div[k]
            + p_at(i + 1, j, k, nx, ny) + p_at(i - 1, j, k, nx, ny)
            + p_at(i, j + 1, k, nx, ny) + p_at(i, j - 1, k, nx, ny)) * 0.25;
}

@compute @workgroup_size(8, 8)
fn project(@builtin(global_invocation_id) gid: vec3u) {
  let nx = i32(P.nx);
  let ny = i32(P.ny);
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (i >= nx || j >= ny) { return; }
  let k = cell_index(i, j, nx);
  if (solid[k] != 0u) { return; } // no velocity is projected inside a solid

  var v = vel[k];
  v.x -= 0.5 * (p_at(i + 1, j, k, nx, ny) - p_at(i - 1, j, k, nx, ny)) / P.h;
  v.y -= 0.5 * (p_at(i, j + 1, k, nx, ny) - p_at(i, j - 1, k, nx, ny)) / P.h;
  vel[k] = v;
}
