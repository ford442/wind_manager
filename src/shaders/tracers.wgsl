struct TracerParams {
  max_tracers : u32,
  spawn_count : u32,
  lifetime    : f32,
  dt          : f32,
  nx          : u32,
  ny          : u32,
  h           : f32,
  domain_w    : f32,
  domain_h    : f32,
  seed        : u32,
  margin      : f32,
};

@group(0) @binding(0) var<uniform> T : TracerParams;
@group(0) @binding(1) var<storage, read_write> tracers : array<Tracer>;
@group(0) @binding(2) var<storage, read_write> slot_counter : atomic<u32>;
@group(0) @binding(3) var<storage, read> vel : array<vec2f>;
@group(0) @binding(4) var<storage, read> solid : array<u32>;

fn in_solid(pos: vec2f, h: f32, nx: i32, ny: i32) -> bool {
  let i = i32(floor(pos.x / h));
  let j = i32(floor(pos.y / h));
  if (i < 0 || j < 0 || i >= nx || j >= ny) { return false; }
  return solid[cell_index(i, j, nx)] != 0u;
}

// Bilinear velocity lookup on this module's `vel` binding.
fn sample_vel(pos: vec2f, h: f32, nx: i32, ny: i32) -> vec2f {
  let b = bilin_at(pos, h, nx, ny);
  return bilin_blend(
    b,
    vel[cell_index(b.i0, b.j0, nx)], vel[cell_index(b.i1, b.j0, nx)],
    vel[cell_index(b.i0, b.j1, nx)], vel[cell_index(b.i1, b.j1, nx)],
  );
}

@compute @workgroup_size(64)
fn seed_tracers(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= T.spawn_count) { return; }
  var rng = pcg(T.seed ^ (gid.x * 0x85EBCA6Bu + 1u));

  let slot = atomicAdd(&slot_counter, 1u) % T.max_tracers;
  let m = T.margin;
  let pos = vec2f(
    m + rand01(&rng) * (T.domain_w - 2.0 * m),
    m + rand01(&rng) * (T.domain_h - 2.0 * m),
  );

  var tr : Tracer;
  tr.pos = pos;
  tr.prev_pos = pos;
  tr.age = 0.0;
  // A seed that lands inside a house or trunk is simply not born.
  tr.alive = select(1u, 0u, in_solid(pos, T.h, i32(T.nx), i32(T.ny)));
  tr.tint = rand01(&rng);
  tr._pad = 0.0;
  tracers[slot] = tr;
}

@compute @workgroup_size(128)
fn update_tracers(@builtin(global_invocation_id) gid: vec3u) {
  let ti = gid.x;
  if (ti >= T.max_tracers) { return; }

  var tr = tracers[ti];
  if (tr.alive == 0u) { return; }

  let nx = i32(T.nx);
  let ny = i32(T.ny);

  let v0 = sample_vel(tr.pos, T.h, nx, ny);
  let mid = tr.pos + v0 * (T.dt * 0.5);
  let v1 = sample_vel(mid, T.h, nx, ny);

  // Never step into a solid: slide along the wall on whichever axis stays in
  // fluid, otherwise hold position (streaks wrap around obstacles).
  let step = v1 * T.dt;
  var next = tr.pos + step;
  if (in_solid(next, T.h, nx, ny)) {
    let slide_x = tr.pos + vec2f(step.x, 0.0);
    let slide_y = tr.pos + vec2f(0.0, step.y);
    if (!in_solid(slide_x, T.h, nx, ny) && abs(step.x) >= abs(step.y)) {
      next = slide_x;
    } else if (!in_solid(slide_y, T.h, nx, ny)) {
      next = slide_y;
    } else if (!in_solid(slide_x, T.h, nx, ny)) {
      next = slide_x;
    } else {
      next = tr.pos;
    }
  }
  tr.prev_pos = tr.pos;
  tr.pos = next;
  tr.age += T.dt;

  let dom_w = f32(nx) * T.h;
  let dom_h = f32(ny) * T.h;
  let m = T.margin * 0.5;
  let oob = tr.pos.x < m || tr.pos.x > dom_w - m
         || tr.pos.y < m || tr.pos.y > dom_h - m;

  if (tr.age >= T.lifetime || oob) {
    tr.alive = 0u;
  }

  tracers[ti] = tr;
}
