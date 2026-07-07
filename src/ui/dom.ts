/** Typed `getElementById` — assumes the element exists (same as legacy UI code). */
export function $<T extends HTMLElement = HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

export function $input(id: string): HTMLInputElement {
  return $<HTMLInputElement>(id);
}

export function $button(id: string): HTMLButtonElement {
  return $<HTMLButtonElement>(id);
}

/** Set text on a known stats/label element; no-op if missing (diagnostics path). */
export function setText(id: string, text: string): void {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

export function setTextClass(id: string, className: string, add: boolean): void {
  const el = document.getElementById(id);
  if (el) el.classList.toggle(className, add);
}
