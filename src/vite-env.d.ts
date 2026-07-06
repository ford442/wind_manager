/// <reference types="vite/client" />

// Allow importing WGSL shaders as raw strings via Vite's ?raw suffix
declare module '*.wgsl?raw' {
  const content: string;
  export default content;
}

// Allow importing other text assets if needed
declare module '*?raw' {
  const content: string;
  export default content;
}