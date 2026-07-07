export function setupHelp(): void {
  const panel = document.getElementById('help-panel') as HTMLDetailsElement | null;
  if (!panel) return;

  if (new URLSearchParams(location.search).has('help')) {
    panel.open = true;
  }

  panel.querySelectorAll('[data-preset]').forEach((el) => {
    el.addEventListener('click', (e) => {
      const id = (e.currentTarget as HTMLElement).dataset.preset;
      if (!id) return;
      const select = document.getElementById('preset-select') as HTMLSelectElement | null;
      const loadBtn = document.getElementById('preset-load') as HTMLButtonElement | null;
      if (select && loadBtn) {
        select.value = id;
        select.dispatchEvent(new Event('change'));
        loadBtn.click();
        panel.open = false;
      }
    });
  });
}
