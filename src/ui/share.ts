import type { ApplyStateResult, SavedState } from '../sim/state';
import {
  applySavedState,
  exportStateJson,
  importStateJson,
  pushStateToUrl,
  shareableUrl,
} from '../sim/state';
import type { Params } from '../sim/params';
import { syncAdvancedUI } from './advanced';
import { syncControlsUI } from './controls';
import { syncEmitterPanelUI } from './emitterPanel';

function showStatus(msg: string, isError = false): void {
  const el = document.getElementById('share-status')!;
  el.textContent = msg;
  el.style.color = isError ? 'var(--bad)' : 'var(--muted)';
}

function formatApplyResult(result: ApplyStateResult): string {
  const parts: string[] = ['Configuration applied.'];
  for (const a of result.adjusted) parts.push(`↳ ${a}`);
  for (const w of result.warnings) parts.push(`⚠ ${w}`);
  return parts.join(' ');
}

async function copyText(text: string, btn: HTMLButtonElement, okLabel: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    const prev = btn.textContent;
    btn.textContent = okLabel;
    setTimeout(() => {
      btn.textContent = prev;
    }, 1200);
    return true;
  } catch {
    return false;
  }
}

function downloadJson(json: string): void {
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'wind_manager-config.json';
  a.click();
  URL.revokeObjectURL(url);
}

export function setupShare(
  params: Params,
  {
    onApply,
    loadedFromHash = false,
    hashError = null,
  }: {
    onApply: (result: ApplyStateResult) => void;
    loadedFromHash?: boolean;
    hashError?: string | null;
  },
): { notifyChange: () => void } {
  const textarea = document.getElementById('share-json') as HTMLTextAreaElement;
  const exportBtn = document.getElementById('share-export') as HTMLButtonElement;
  const copyBtn = document.getElementById('share-copy') as HTMLButtonElement;
  const linkBtn = document.getElementById('share-link') as HTMLButtonElement;
  const importBtn = document.getElementById('share-import-btn') as HTMLButtonElement;
  const fileBtn = document.getElementById('share-file-btn') as HTMLButtonElement;
  const fileInput = document.getElementById('share-file') as HTMLInputElement;
  const panel = document.querySelector('.panel')!;

  function syncAll(): void {
    syncControlsUI(params);
    syncAdvancedUI(params);
    syncEmitterPanelUI(params);
  }

  function applyFromParsed(parsed: { state: SavedState }): void {
    const result = applySavedState(params, parsed.state);
    syncAll();
    onApply(result);
    pushStateToUrl(params);
    showStatus(formatApplyResult(result));
  }

  let hashTimer = 0;
  function notifyChange(): void {
    clearTimeout(hashTimer);
    hashTimer = window.setTimeout(() => pushStateToUrl(params), 400);
  }

  exportBtn.addEventListener('click', () => {
    const json = exportStateJson(params);
    textarea.value = json;
    downloadJson(json);
    showStatus('Exported JSON — also shown in the text area.');
  });

  copyBtn.addEventListener('click', async () => {
    const json = exportStateJson(params);
    textarea.value = json;
    const ok = await copyText(json, copyBtn, 'Copied!');
    showStatus(ok ? 'JSON copied to clipboard.' : 'Copy failed — use Export or select the text area.', !ok);
  });

  linkBtn.addEventListener('click', async () => {
    const url = shareableUrl(params);
    const ok = await copyText(url, linkBtn, 'Link copied!');
    showStatus(
      ok
        ? 'Shareable link copied — bookmark or send to reproduce this experiment.'
        : 'Copy failed — copy the URL from the address bar after adjusting controls.',
      !ok,
    );
    pushStateToUrl(params);
  });

  importBtn.addEventListener('click', () => {
    const parsed = importStateJson(textarea.value.trim());
    if ('error' in parsed) {
      showStatus(parsed.error, true);
      return;
    }
    applyFromParsed(parsed);
  });

  fileBtn.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result ?? '');
      textarea.value = text;
      const parsed = importStateJson(text.trim());
      if ('error' in parsed) {
        showStatus(parsed.error, true);
        return;
      }
      applyFromParsed(parsed);
    };
    reader.readAsText(file);
    fileInput.value = '';
  });

  panel.addEventListener('input', notifyChange);
  panel.addEventListener('change', notifyChange);

  pushStateToUrl(params);
  if (hashError) showStatus(hashError, true);
  else if (loadedFromHash) showStatus('Loaded experiment from URL hash.');

  return { notifyChange };
}
