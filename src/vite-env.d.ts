/// <reference types="vite/client" />

import type { WgslSource } from './shaders/wgsl';

declare module '*.wgsl?raw' {
  const content: WgslSource;
  export default content;
}

declare module '*?raw' {
  const content: string;
  export default content;
}
