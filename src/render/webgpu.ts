import type { WgslSource } from '../shaders/wgsl';

/**
 * Shared WebGPU helpers used by the sim (`step.ts`), renderer (`overlays.ts`),
 * and CPU readback paths (`fieldDiagnostics.ts`, `velSampler.ts`).
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

export function configureWebGPUCanvas(
  context: GPUCanvasContext,
  device: GPUDevice,
  format: GPUTextureFormat,
): void {
  context.configure({ device, format, alphaMode: 'opaque' });
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

export function shaderModule(device: GPUDevice, label: string, code: WgslSource): GPUShaderModule {
  return device.createShaderModule({ label, code });
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
