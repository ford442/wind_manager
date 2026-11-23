import type { WgslSource } from '../shaders/wgsl';

/**
 * Shared WebGPU helpers used by the sim (`step.ts`), renderer (`overlays.ts`),
 * environment (`environment.ts`), and the HUD readback (`fieldDiagnostics.ts`).
 *
 * Pipelines use `layout: 'auto'`; bind-group layouts are inferred from the first
 * `createBindGroup` call after pipeline creation.
 */

/** Usage flags for GPU→CPU readback staging buffers (`COPY_DST` + `MAP_READ`). */
export const MAP_READ_BUFFER_USAGE =
  GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ;

/** Allocate a buffer suitable for `copyBufferToBuffer` followed by `mapAsync(READ)`. */
export function createReadbackBuffer(device: GPUDevice, byteLength: number): GPUBuffer {
  return device.createBuffer({ size: byteLength, usage: MAP_READ_BUFFER_USAGE });
}

/** Returns the WebGPU canvas context or throws if unavailable. */
export function getWebGPUContext(canvas: HTMLCanvasElement): GPUCanvasContext {
  const context = canvas.getContext('webgpu');
  if (!context) {
    throw new Error('WebGPU canvas context unavailable');
  }
  return context;
}

/**
 * Configure the swap chain explicitly so output does not depend on browser
 * defaults: preferred format, opaque alpha, sRGB, and `standard` tone mapping
 * (values clamp to [0, 1]; no HDR extended range). Call again after a resize.
 */
export function configureWebGPUCanvas(
  context: GPUCanvasContext,
  device: GPUDevice,
  format: GPUTextureFormat,
): void {
  context.configure({
    device,
    format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
    alphaMode: 'opaque',
    colorSpace: 'srgb',
    toneMapping: { mode: 'standard' },
  });
}

/**
 * Map a buffer for read, run `read`, then always unmap.
 * Prefer this over manual map/unmap when the mapped range is consumed synchronously.
 */
export async function readMappedBuffer<T>(
  buffer: GPUBuffer,
  read: (mapped: ArrayBuffer) => T,
): Promise<T> {
  await buffer.mapAsync(GPUMapMode.READ);
  try {
    return read(buffer.getMappedRange());
  } finally {
    buffer.unmap();
  }
}

/** Standard alpha blending for particles, arrows, and emitter overlays. */
export function alphaBlendState(): GPUBlendState {
  return {
    color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
  };
}

export function opaqueColorTarget(format: GPUTextureFormat): GPUColorTargetState {
  return { format };
}

export function blendedColorTarget(format: GPUTextureFormat): GPUColorTargetState {
  return { format, blend: alphaBlendState() };
}

/** One named piece of a concatenated WGSL module (for mapping error lines back). */
export interface WgslPart {
  name: string;
  code: WgslSource | string;
}

/** Thrown when a shader module fails to compile; `message` lists `file:line:col` errors. */
export class ShaderCompileError extends Error {
  constructor(
    readonly label: string,
    readonly lines: string[],
  ) {
    super(`Shader "${label}" failed to compile:\n${lines.join('\n')}`);
    this.name = 'ShaderCompileError';
  }
}

/** Thrown when a validation error scope around pipeline creation catches an error. */
export class PipelineValidationError extends Error {
  constructor(label: string, detail: string) {
    super(`Pipeline "${label}" failed validation: ${detail}`);
    this.name = 'PipelineValidationError';
  }
}

function lineCount(code: string): number {
  let n = 1;
  for (let i = 0; i < code.length; i++) if (code.charCodeAt(i) === 10) n++;
  return n;
}

/** Map a 1-based line in the joined module to `part:line`. Parts are joined with `\n`. */
function locate(parts: readonly WgslPart[], line: number): { name: string; line: number } {
  let start = 1;
  for (const part of parts) {
    const n = lineCount(part.code);
    if (line < start + n) return { name: part.name, line: line - start + 1 };
    start += n;
  }
  return { name: parts[parts.length - 1]?.name ?? '?', line };
}

/**
 * Create a shader module from named WGSL parts (joined with `\n`), await its
 * compilation info, and throw {@link ShaderCompileError} with per-file line
 * numbers if it has errors. Warnings are logged. Call before pipeline creation
 * so a WGSL mistake fails at init instead of as an uncaptured error mid-frame.
 */
export async function compileShaderModule(
  device: GPUDevice,
  label: string,
  parts: readonly WgslPart[],
): Promise<GPUShaderModule> {
  const code = parts.map((p) => p.code).join('\n');
  // Scope the creation so a WGSL error is reported once, below, with mapped
  // line numbers instead of also reaching the device's `uncapturederror`.
  // Push/create/pop back to back (no await between) so concurrent compiles
  // never share a scope; the stack is per-device, not per-call.
  device.pushErrorScope('validation');
  const module = device.createShaderModule({ label, code });
  const scopePromise = device.popErrorScope();
  const [info, scoped] = await Promise.all([module.getCompilationInfo(), scopePromise]);
  const errors: string[] = [];
  for (const msg of info.messages) {
    const where = msg.lineNum > 0 ? locate(parts, msg.lineNum) : null;
    const loc = where ? `${where.name}:${where.line}:${msg.linePos}` : label;
    const text = `${loc}: ${msg.type}: ${msg.message}`;
    if (msg.type === 'error') {
      errors.push(text);
    } else {
      console.warn(`[wind_manager] ${text}`);
    }
  }
  if (errors.length === 0 && scoped) errors.push(`${label}: ${scoped.message}`);
  if (errors.length > 0) {
    for (const e of errors) console.error(`[wind_manager] ${e}`);
    throw new ShaderCompileError(label, errors);
  }
  return module;
}

/**
 * Run synchronous pipeline creation inside a `validation` error scope and throw
 * {@link PipelineValidationError} if anything inside failed.
 */
export async function withValidationScope<T>(
  device: GPUDevice,
  label: string,
  create: () => T,
): Promise<T> {
  device.pushErrorScope('validation');
  let result: T;
  try {
    result = create();
  } catch (thrown) {
    await device.popErrorScope();
    throw thrown;
  }
  const error = await device.popErrorScope();
  if (error) throw new PipelineValidationError(label, error.message);
  return result;
}

/** Create a compute pipeline with an auto-generated bind-group layout. */
export function computePipeline(
  device: GPUDevice,
  descriptor: GPUComputePipelineDescriptor,
): GPUComputePipeline {
  return device.createComputePipeline(descriptor);
}

/** Create a render pipeline with an auto-generated bind-group layout. */
export function renderPipeline(
  device: GPUDevice,
  descriptor: GPURenderPipelineDescriptor,
): GPURenderPipeline {
  return device.createRenderPipeline(descriptor);
}

export type Workgroup1D = number;
export type Workgroup2D = [workgroupCountX: number, workgroupCountY: number];

/** 1D dispatch size: `ceil(count / workgroupSize)`. */
export function workgroups1d(count: number, size: number): Workgroup1D {
  return Math.ceil(count / size);
}

/** 2D dispatch size for `8×8` (or custom) compute tiles over an `nx×ny` grid. */
export function workgroups2d(nx: number, ny: number, sx: number, sy: number): Workgroup2D {
  return [Math.ceil(nx / sx), Math.ceil(ny / sy)];
}
