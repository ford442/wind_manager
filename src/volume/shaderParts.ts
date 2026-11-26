import { MAX_EMITTERS } from '../sim/emitters';
import type { WgslPart } from '../render/webgpu';

import commonSrc from '../shaders/common.wgsl?raw';
import volCommonSrc from '../shaders/vol_common.wgsl?raw';

/**
 * How every `vol_*.wgsl` module is assembled: the generated MAX_EMITTERS
 * prelude, `common.wgsl` (constants, q_sat, virt_temp, RNG), `vol_common.wgsl`
 * (VParams, VDroplet, trilinear helpers), then the module itself.
 * `scripts/validate-wgsl.mjs` rebuilds exactly this text.
 */
export function volumeShaderParts(name: string, src: string): WgslPart[] {
  return [
    { name: 'generated', code: `const MAX_EMITTERS : u32 = ${MAX_EMITTERS}u;` },
    { name: 'common.wgsl', code: commonSrc },
    { name: 'vol_common.wgsl', code: volCommonSrc },
    { name: `${name}.wgsl`, code: src },
  ];
}
