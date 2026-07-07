import {
  previewAdvancedWarnings,
  resetAdvancedToDefaults,
  sanitizeAdvancedParams,
  type AdvancedKey,
} from '../sim/advanced';
import type { Params } from '../sim/params';

type SliderSpec = {
  id: string;
  key: AdvancedKey;
  min: number;
  max: number;
  step: number;
  fmt: (v: number, p: Params) => string;
};

const SLIDERS: SliderSpec[] = [
  {
    id: 'adv-dt',
    key: 'dt',
    min: 1 / 480,
    max: 1 / 40,
    step: 1 / 480,
    fmt: (v) => `${(v * 1000).toFixed(2)} ms`,
  },
  {
    id: 'adv-substeps',
    key: 'substeps',
    min: 1,
    max: 8,
    step: 1,
    fmt: (v) => `${v}`,
  },
  {
    id: 'adv-jacobi',
    key: 'jacobiIters',
    min: 5,
    max: 80,
    step: 1,
    fmt: (v) => `${v}`,
  },
  {
    id: 'adv-damp',
    key: 'damp',
    min: 0,
    max: 0.12,
    step: 0.005,
    fmt: (v) => v.toFixed(3),
  },
  {
    id: 'adv-relax',
    key: 'relax',
    min: 0.002,
    max: 0.1,
    step: 0.002,
    fmt: (v) => `${v.toFixed(3)} /s`,
  },
  {
    id: 'adv-domain-w',
    key: 'domainW',
    min: 8,
    max: 40,
    step: 0.5,
    fmt: (v) => `${v.toFixed(1)} m`,
  },
  {
    id: 'adv-domain-h',
    key: 'domainH',
    min: 4,
    max: 12,
    step: 0.25,
    fmt: (v) => `${v.toFixed(2)} m`,
  },
  {
    id: 'adv-nx',
    key: 'nx',
    min: 128,
    max: 512,
    step: 8,
    fmt: (v) => `${v}`,
  },
  {
    id: 'adv-ny',
    key: 'ny',
    min: 64,
    max: 256,
    step: 8,
    fmt: (v) => `${v}`,
  },
];

function setAdvancedParam(params: Params, key: AdvancedKey, value: number): void {
  params[key] = value;
}

function cellLabel(p: Params): string {
  const h = (p.domainW / p.nx) * 1000;
  return `cell ≈ ${h.toFixed(1)} mm · ${p.nx}×${p.ny}`;
}

function showMessages(warnings: string[], adjusted: string[], extra?: string): void {
  const el = document.getElementById('advanced-warnings')!;
  const parts: string[] = [];
  if (extra) parts.push(extra);
  for (const a of adjusted) parts.push(`↳ ${a}`);
  for (const w of warnings) parts.push(`⚠ ${w}`);
  el.textContent = parts.join('\n');
  el.classList.toggle('has-warn', warnings.length > 0 || adjusted.length > 0 || !!extra);
}

export function syncAdvancedUI(params: Params): void {
  for (const spec of SLIDERS) {
    const el = document.getElementById(spec.id) as HTMLInputElement;
    const label = document.getElementById(spec.id + '-val')!;
    const val = params[spec.key];
    el.value = String(val);
    label.textContent = spec.fmt(val, params);
  }
  const cellEl = document.getElementById('adv-cell-val');
  if (cellEl) cellEl.textContent = cellLabel(params);
  showMessages(previewAdvancedWarnings(params), []);
}

export function setupAdvanced(
  params: Params,
  {
    onApply,
    onPause,
  }: {
    onApply: (result: ReturnType<typeof sanitizeAdvancedParams>) => void;
    onPause: () => void;
  },
): { notifyInstability: (message: string) => void } {
  for (const spec of SLIDERS) {
    const el = document.getElementById(spec.id) as HTMLInputElement;
    const label = document.getElementById(spec.id + '-val')!;
    el.addEventListener('input', () => {
      const raw = parseFloat(el.value);
      const value =
        spec.key === 'substeps' || spec.key === 'jacobiIters' || spec.key === 'nx' || spec.key === 'ny'
          ? Math.round(raw)
          : raw;
      setAdvancedParam(params, spec.key, value);
      label.textContent = spec.fmt(params[spec.key], params);
      const cellEl = document.getElementById('adv-cell-val');
      if (cellEl) cellEl.textContent = cellLabel(params);
      showMessages(previewAdvancedWarnings(params), []);
    });
  }

  document.getElementById('advanced-reset')!.addEventListener('click', () => {
    resetAdvancedToDefaults(params);
    syncAdvancedUI(params);
    showMessages([], [], 'Advanced values reset — click Apply to run.');
  });

  document.getElementById('advanced-apply')!.addEventListener('click', () => {
    const prev = { nx: params.nx, ny: params.ny };
    const result = sanitizeAdvancedParams(params, prev);
    syncAdvancedUI(params);
    showMessages(result.warnings, result.adjusted, 'Applied — simulation reset.');
    onApply(result);
  });

  syncAdvancedUI(params);

  return {
    notifyInstability(message: string) {
      onPause();
      showMessages(
        previewAdvancedWarnings(params),
        [],
        `⛔ ${message} Simulation paused — try smaller dt, more substeps, or higher Jacobi iterations.`,
      );
    },
  };
}

export type AdvancedControls = ReturnType<typeof setupAdvanced>;
