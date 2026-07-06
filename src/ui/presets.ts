import type { Params } from '../sim/params';
import { syncAdvancedUI } from './advanced';
import { applyPreset, presetHint, SIM_PRESETS } from '../sim/presets';
import { syncControlsUI } from './controls';
import { syncEmitterPanelUI } from './emitterPanel';

export function setupPresets(
  params: Params,
  {
    onApply,
  }: {
    onApply: () => void;
  },
): void {
  const select = document.getElementById('preset-select') as HTMLSelectElement;
  const loadBtn = document.getElementById('preset-load') as HTMLButtonElement;
  const hint = document.getElementById('preset-hint')!;

  select.innerHTML = '';
  for (const preset of SIM_PRESETS) {
    const opt = document.createElement('option');
    opt.value = preset.id;
    opt.textContent = preset.label;
    select.appendChild(opt);
  }

  function showHint(id: string): void {
    const preset = SIM_PRESETS.find((p) => p.id === id);
    hint.textContent = preset ? presetHint(preset, params) : '';
  }

  function loadSelected(): void {
    const preset = SIM_PRESETS.find((p) => p.id === select.value);
    if (!preset) return;
    applyPreset(params, preset);
    syncControlsUI(params);
    syncAdvancedUI(params);
    syncEmitterPanelUI(params);
    onApply();
    showHint(preset.id);
  }

  select.addEventListener('change', () => showHint(select.value));
  loadBtn.addEventListener('click', loadSelected);

  select.value = 'default';
  showHint('default');
}
