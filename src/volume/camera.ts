import { VOLUME_DOMAIN } from './grid';
import type { CameraMode } from './params';

/** Column-major 4×4 matrices (WGSL `mat4x4f` layout). */
export type Mat4 = Float32Array;

export function mul(a: Mat4, b: Mat4): Mat4 {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
  }
  return o;
}

/** Perspective with WebGPU clip depth in [0, 1]. */
export function perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovY / 2);
  const o = new Float32Array(16);
  o[0] = f / aspect;
  o[5] = f;
  o[10] = far / (near - far);
  o[11] = -1;
  o[14] = (far * near) / (near - far);
  return o;
}

type V3 = [number, number, number];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a: V3): V3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

export function lookAt(eye: V3, target: V3, up: V3 = [0, 1, 0]): Mat4 {
  const z = norm(sub(eye, target));
  const x = norm(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ]);
}

export function invert(m: Mat4): Mat4 {
  const inv = new Float32Array(16);
  const a = m;
  inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
  inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
  inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
  inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
  inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
  inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
  inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
  inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
  inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
  inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
  inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
  inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
  inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
  inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
  inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
  inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5];
  const det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
  const s = det !== 0 ? 1 / det : 0;
  for (let i = 0; i < 16; i++) inv[i] *= s;
  return inv;
}

export const EYE_HEIGHT = 1.7;

/**
 * Orbit (around a target in the yard) or walk (eye height, look around).
 * Angles in radians; yaw 0 looks along +x, π/2 along +z.
 */
export class YardCamera {
  mode: CameraMode = 'orbit';
  // From the front of the yard (+z), +x to the right like the side view,
  // framing the default hose and the hero tree.
  target: V3 = [VOLUME_DOMAIN.w * 0.75, 1.0, VOLUME_DOMAIN.d * 0.5];
  yaw = 1.25;
  pitch = 0.36;
  dist = 17;
  walkPos: V3 = [VOLUME_DOMAIN.w * 0.45, EYE_HEIGHT, VOLUME_DOMAIN.d * 0.72];
  walkYaw = -0.25;
  walkPitch = -0.05;
  fovY = (55 * Math.PI) / 180;

  eye(): V3 {
    if (this.mode === 'walk') return [...this.walkPos];
    const cp = Math.cos(this.pitch);
    return [
      this.target[0] + this.dist * cp * Math.cos(this.yaw),
      this.target[1] + this.dist * Math.sin(this.pitch),
      this.target[2] + this.dist * cp * Math.sin(this.yaw),
    ];
  }

  forward(): V3 {
    if (this.mode === 'walk') {
      const cp = Math.cos(this.walkPitch);
      return [cp * Math.cos(this.walkYaw), Math.sin(this.walkPitch), cp * Math.sin(this.walkYaw)];
    }
    return norm(sub(this.target, this.eye()));
  }

  viewProj(aspect: number): { viewProj: Mat4; invViewProj: Mat4; eye: V3 } {
    const eye = this.eye();
    const f = this.forward();
    const view = lookAt(eye, [eye[0] + f[0], eye[1] + f[1], eye[2] + f[2]]);
    const proj = perspective(this.fovY, aspect, 0.05, 200);
    const viewProj = mul(proj, view);
    return { viewProj, invViewProj: invert(viewProj), eye };
  }

  /** Pointer drag in pixels. */
  rotate(dx: number, dy: number): void {
    if (this.mode === 'walk') {
      this.walkYaw += dx * 0.004;
      this.walkPitch = Math.max(-1.3, Math.min(1.3, this.walkPitch - dy * 0.004));
    } else {
      this.yaw += dx * 0.005;
      this.pitch = Math.max(0.03, Math.min(1.45, this.pitch + dy * 0.005));
    }
  }

  pan(dx: number, dy: number): void {
    if (this.mode === 'walk') return;
    const s = this.dist * 0.0016;
    const right: V3 = [-Math.sin(this.yaw), 0, Math.cos(this.yaw)];
    const fwd: V3 = [Math.cos(this.yaw), 0, Math.sin(this.yaw)];
    this.target[0] = clampX(this.target[0] - (right[0] * dx - fwd[0] * dy) * s);
    this.target[2] = clampZ(this.target[2] - (right[2] * dx - fwd[2] * dy) * s);
  }

  zoom(delta: number): void {
    if (this.mode === 'walk') {
      this.move(-delta * 0.01, 0);
      return;
    }
    this.dist = Math.max(3, Math.min(60, this.dist * Math.exp(delta * 0.001)));
  }

  /** Walk: forward/back and strafe in metres, staying at eye height inside the yard. */
  move(forward: number, strafe: number): void {
    const fx = Math.cos(this.walkYaw);
    const fz = Math.sin(this.walkYaw);
    this.walkPos[0] = clampX(this.walkPos[0] + fx * forward - fz * strafe);
    this.walkPos[2] = clampZ(this.walkPos[2] + fz * forward + fx * strafe);
    this.walkPos[1] = EYE_HEIGHT;
  }

  /** World ray through a viewport point (`ndc` in [-1, 1]). */
  ray(ndcX: number, ndcY: number, aspect: number): { o: V3; d: V3 } {
    const { invViewProj, eye } = this.viewProj(aspect);
    const v = [ndcX, ndcY, 1, 1];
    const w = [0, 0, 0, 0];
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) w[r] += invViewProj[c * 4 + r] * v[c];
    const far: V3 = [w[0] / w[3], w[1] / w[3], w[2] / w[3]];
    return { o: eye, d: norm(sub(far, eye)) };
  }
}

const clampX = (x: number) => Math.max(-5, Math.min(VOLUME_DOMAIN.w + 5, x));
const clampZ = (z: number) => Math.max(-5, Math.min(VOLUME_DOMAIN.d + 5, z));
