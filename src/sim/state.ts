import {
  MAX_EMITTERS,
  clampEmitterPosition,
  createEmitter,
  type Emitter,
  type EmitterType,
} from './emitters';
import { sanitizeAdvancedParams } from './advanced';
import { defaultParams, type Params } from './params';

export const STATE_VERSION = 1;
const HASH_PREFIX = '#s=';

export interface SavedEmitter {
  type: EmitterType;
  x: number;
  y: number;
  angleDeg: number;
  spreadDeg: number;
  speed: number;
  rate: number;
  rMinUm: number;
  rMaxUm: number;
}

export interface SavedState {
  v: typeof STATE_VERSION;
  nx: number;
  ny: number;
  domainW: number;
  domainH: number;
  dt: number;
  substeps: number;
  jacobiIters: number;
  tAmb: number;
  rhAmb: number;
  latentOn: boolean;
  relax: number;
  damp: number;
  overlay: number;
  showArrows: boolean;
  showDroplets: boolean;
  showTracers: boolean;
  showTracerStreaks: boolean;
  tracerDensity: number;
  tracerLifetime: number;
  showGrass: boolean;
  grassDensity: number;
  grassStiffness: number;
  showTrees: boolean;
  showHouses: boolean;
  showClouds: boolean;
  showGroundMist: boolean;
  showWetGround: boolean;
  showGroundMoisture: boolean;
  stylizedView: boolean;
  emitters: SavedEmitter[];
  selectedEmitter: number;
}

export interface ApplyStateResult {
  needsGridRebuild: boolean;
  warnings: string[];
  adjusted: string[];
}

function num(v: unknown, fallback: number): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

function bool(v: unknown, fallback: boolean): boolean {
  return typeof v === 'boolean' ? v : fallback;
}

function toSavedEmitter(em: Emitter): SavedEmitter {
  return {
    type: em.type,
    x: em.x,
    y: em.y,
    angleDeg: em.angleDeg,
    spreadDeg: em.spreadDeg,
    speed: em.speed,
    rate: em.rate,
    rMinUm: em.rMinUm,
    rMaxUm: em.rMaxUm,
  };
}

export function serializeState(p: Params): SavedState {
  const selected = p.selectedEmitterId
    ? Math.max(0, p.emitters.findIndex((e) => e.id === p.selectedEmitterId))
    : 0;

  return {
    v: STATE_VERSION,
    nx: p.nx,
    ny: p.ny,
    domainW: p.domainW,
    domainH: p.domainH,
    dt: p.dt,
    substeps: p.substeps,
    jacobiIters: p.jacobiIters,
    tAmb: p.tAmb,
    rhAmb: p.rhAmb,
    latentOn: p.latentOn,
    relax: p.relax,
    damp: p.damp,
    overlay: p.overlay,
    showArrows: p.showArrows,
    showDroplets: p.showDroplets,
    showTracers: p.showTracers,
    showTracerStreaks: p.showTracerStreaks,
    tracerDensity: p.tracerDensity,
    tracerLifetime: p.tracerLifetime,
    showGrass: p.showGrass,
    grassDensity: p.grassDensity,
    grassStiffness: p.grassStiffness,
    showTrees: p.showTrees,
    showHouses: p.showHouses,
    showClouds: p.showClouds,
    showGroundMist: p.showGroundMist,
    showWetGround: p.showWetGround,
    showGroundMoisture: p.showGroundMoisture,
    stylizedView: p.stylizedView,
    emitters: p.emitters.map(toSavedEmitter),
    selectedEmitter: selected < 0 ? 0 : selected,
  };
}

function parseEmitter(raw: unknown): SavedEmitter | null {
  if (!raw || typeof raw !== 'object') return null;
  const o = raw as Record<string, unknown>;
  const type: EmitterType | null =
    o.type === 'water' ? 'water' : o.type === 'air' ? 'air' : null;
  if (!type) return null;
  return {
    type,
    x: num(o.x, 0),
    y: num(o.y, 0),
    angleDeg: num(o.angleDeg, 0),
    spreadDeg: num(o.spreadDeg, type === 'water' ? 15 : 22),
    speed: num(o.speed, type === 'water' ? 10 : 9),
    rate: num(o.rate, type === 'water' ? 6000 : 5000),
    rMinUm: num(o.rMinUm, 50),
    rMaxUm: num(o.rMaxUm, 800),
  };
}

export function parseSavedState(raw: unknown): { state: SavedState } | { error: string } {
  if (!raw || typeof raw !== 'object') {
    return { error: 'Invalid JSON — expected an object' };
  }
  const o = raw as Record<string, unknown>;
  const version = o.v;
  if (version !== STATE_VERSION) {
    return { error: `Unsupported state version ${String(version)} (expected ${STATE_VERSION})` };
  }

  const defs = defaultParams();
  const emitters: SavedEmitter[] = [];
  if (Array.isArray(o.emitters)) {
    for (const item of o.emitters) {
      const em = parseEmitter(item);
      if (em) emitters.push(em);
    }
  }

  const state: SavedState = {
    v: STATE_VERSION,
    nx: num(o.nx, defs.nx),
    ny: num(o.ny, defs.ny),
    domainW: num(o.domainW, defs.domainW),
    domainH: num(o.domainH, defs.domainH),
    dt: num(o.dt, defs.dt),
    substeps: num(o.substeps, defs.substeps),
    jacobiIters: num(o.jacobiIters, defs.jacobiIters),
    tAmb: num(o.tAmb, defs.tAmb),
    rhAmb: num(o.rhAmb, defs.rhAmb),
    latentOn: bool(o.latentOn, defs.latentOn),
    relax: num(o.relax, defs.relax),
    damp: num(o.damp, defs.damp),
    overlay: num(o.overlay, defs.overlay),
    showArrows: bool(o.showArrows, defs.showArrows),
    showDroplets: bool(o.showDroplets, defs.showDroplets),
    showTracers: bool(o.showTracers, defs.showTracers),
    showTracerStreaks: bool(o.showTracerStreaks, defs.showTracerStreaks),
    tracerDensity: num(o.tracerDensity, defs.tracerDensity),
    tracerLifetime: num(o.tracerLifetime, defs.tracerLifetime),
    showGrass: bool(o.showGrass, defs.showGrass),
    grassDensity: num(o.grassDensity, defs.grassDensity),
    grassStiffness: num(o.grassStiffness, defs.grassStiffness),
    showTrees: bool(o.showTrees, defs.showTrees),
    showHouses: bool(o.showHouses, defs.showHouses),
    showClouds: bool(o.showClouds, defs.showClouds),
    showGroundMist: bool(o.showGroundMist, defs.showGroundMist),
    showWetGround: bool(o.showWetGround, defs.showWetGround),
    showGroundMoisture: bool(o.showGroundMoisture, defs.showGroundMoisture),
    stylizedView: bool(o.stylizedView, defs.stylizedView),
    emitters,
    selectedEmitter: num(o.selectedEmitter, 0),
  };

  return { state };
}

export function applySavedState(target: Params, saved: SavedState): ApplyStateResult {
  const prev = { nx: target.nx, ny: target.ny };
  const warnings: string[] = [];

  target.nx = saved.nx;
  target.ny = saved.ny;
  target.domainW = saved.domainW;
  target.domainH = saved.domainH;
  target.dt = saved.dt;
  target.substeps = saved.substeps;
  target.jacobiIters = saved.jacobiIters;
  target.tAmb = saved.tAmb;
  target.rhAmb = saved.rhAmb;
  target.latentOn = saved.latentOn;
  target.relax = saved.relax;
  target.damp = saved.damp;
  target.overlay = saved.overlay;
  target.showArrows = saved.showArrows;
  target.showDroplets = saved.showDroplets;
  target.showTracers = saved.showTracers;
  target.showTracerStreaks = saved.showTracerStreaks;
  target.tracerDensity = saved.tracerDensity;
  target.tracerLifetime = saved.tracerLifetime;
  target.showGrass = saved.showGrass;
  target.grassDensity = saved.grassDensity;
  target.grassStiffness = saved.grassStiffness;
  target.showTrees = saved.showTrees;
  target.showHouses = saved.showHouses;
  target.showClouds = saved.showClouds;
  target.showGroundMist = saved.showGroundMist;
  target.showWetGround = saved.showWetGround;
  target.showGroundMoisture = saved.showGroundMoisture;
  target.stylizedView = saved.stylizedView;

  const defs = saved.emitters.slice(0, MAX_EMITTERS);
  if (saved.emitters.length > MAX_EMITTERS) {
    warnings.push(`Truncated ${saved.emitters.length - MAX_EMITTERS} emitter(s) (max ${MAX_EMITTERS})`);
  }

  target.emitters = defs.map((def) => {
    const em = createEmitter(def.type, def.x, def.y, def);
    clampEmitterPosition(em, target);
    return em;
  });

  const sel = Math.max(0, Math.min(defs.length - 1, Math.round(saved.selectedEmitter)));
  target.selectedEmitterId = target.emitters[sel]?.id ?? target.emitters[0]?.id ?? null;

  const result = sanitizeAdvancedParams(target, prev);
  return {
    needsGridRebuild: result.needsGridRebuild,
    warnings: [...warnings, ...result.warnings],
    adjusted: result.adjusted,
  };
}

export function exportStateJson(p: Params, pretty = true): string {
  const state = serializeState(p);
  return pretty ? JSON.stringify(state, null, 2) : JSON.stringify(state);
}

export function importStateJson(json: string): { state: SavedState } | { error: string } {
  let raw: unknown;
  try {
    raw = JSON.parse(json);
  } catch {
    return { error: 'Invalid JSON syntax' };
  }
  return parseSavedState(raw);
}

function toBase64Url(text: string): string {
  const bytes = new TextEncoder().encode(text);
  let bin = '';
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
}

function fromBase64Url(encoded: string): string {
  const pad = encoded.length % 4 === 0 ? '' : '='.repeat(4 - (encoded.length % 4));
  const bin = atob(encoded.replace(/-/g, '+').replace(/_/g, '/') + pad);
  const bytes = Uint8Array.from(bin, (c) => c.charCodeAt(0));
  return new TextDecoder().decode(bytes);
}

export function encodeStateHash(p: Params): string {
  return HASH_PREFIX + toBase64Url(exportStateJson(p, false));
}

export function decodeStateHash(hash: string): { state: SavedState } | { error: string } {
  if (!hash || !hash.startsWith(HASH_PREFIX)) {
    return { error: 'No share hash in URL' };
  }
  try {
    const json = fromBase64Url(hash.slice(HASH_PREFIX.length));
    return importStateJson(json);
  } catch {
    return { error: 'Could not decode URL hash' };
  }
}

export function loadStateFromHash(): { state: SavedState } | { error: string } | null {
  const hash = location.hash;
  if (!hash || !hash.startsWith(HASH_PREFIX)) return null;
  return decodeStateHash(hash);
}

export function pushStateToUrl(p: Params, replace = true): void {
  const hash = encodeStateHash(p);
  const url = `${location.pathname}${location.search}${hash}`;
  if (replace) {
    history.replaceState(null, '', url);
  } else {
    history.pushState(null, '', url);
  }
}

export function shareableUrl(p: Params): string {
  const hash = encodeStateHash(p);
  return `${location.origin}${location.pathname}${location.search}${hash}`;
}
