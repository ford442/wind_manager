/** Branded WGSL source text from `?raw` imports — not arbitrary strings. */
export type WgslSource = string & { readonly __wgslBrand: unique symbol };

export function asWgsl(source: string): WgslSource {
  return source as WgslSource;
}

export function concatWgsl(...parts: readonly (WgslSource | string)[]): WgslSource {
  return parts.join('\n') as WgslSource;
}
