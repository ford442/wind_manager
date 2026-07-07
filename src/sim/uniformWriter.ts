/**
 * Small typed writer for uniform `ArrayBuffer`s (little-endian, 4-byte aligned fields).
 * Field offsets are declared explicitly in layout modules (see `simParamsUniform.ts`).
 */
export class UniformWriter {
  readonly buffer: ArrayBuffer;
  private readonly view: DataView;

  constructor(byteLength: number) {
    this.buffer = new ArrayBuffer(byteLength);
    this.view = new DataView(this.buffer);
  }

  u32(byteOffset: number, value: number): void {
    this.view.setUint32(byteOffset, value >>> 0, true);
  }

  f32(byteOffset: number, value: number): void {
    this.view.setFloat32(byteOffset, value, true);
  }
}
