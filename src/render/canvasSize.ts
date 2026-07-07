export interface CanvasSize {
  cssWidth: number;
  cssHeight: number;
  pixelWidth: number;
  pixelHeight: number;
  dpr: number;
}

/** Cap DPR for performance on 3× mobile / Retina displays. */
export function getDisplayDpr(max = 2.5): number {
  return Math.min(window.devicePixelRatio || 1, max);
}

export function measureElement(el: HTMLElement): { width: number; height: number } {
  const rect = el.getBoundingClientRect();
  return {
    width: Math.max(1, rect.width),
    height: Math.max(1, rect.height),
  };
}

export function applyCanvasBackingStore(
  canvas: HTMLCanvasElement,
  cssWidth: number,
  cssHeight: number,
  dpr: number,
  pixelWidth?: number,
  pixelHeight?: number,
): { pixelWidth: number; pixelHeight: number; changed: boolean } {
  const pw = pixelWidth ?? Math.max(1, Math.round(cssWidth * dpr));
  const ph = pixelHeight ?? Math.max(1, Math.round(cssHeight * dpr));
  const changed = canvas.width !== pw || canvas.height !== ph;
  if (changed) {
    canvas.width = pw;
    canvas.height = ph;
  }
  return { pixelWidth: pw, pixelHeight: ph, changed };
}

function readPixelBox(
  entry: ResizeObserverEntry | undefined,
  cssWidth: number,
  cssHeight: number,
  dpr: number,
): { pixelWidth: number; pixelHeight: number; usedDeviceBox: boolean } {
  const box = entry?.devicePixelContentBoxSize?.[0];
  if (box && box.inlineSize > 0 && box.blockSize > 0) {
    return {
      pixelWidth: box.inlineSize,
      pixelHeight: box.blockSize,
      usedDeviceBox: true,
    };
  }
  return {
    pixelWidth: Math.max(1, Math.round(cssWidth * dpr)),
    pixelHeight: Math.max(1, Math.round(cssHeight * dpr)),
    usedDeviceBox: false,
  };
}

export function setupCanvasResize(options: {
  container: HTMLElement;
  canvases: HTMLCanvasElement[];
  onResize: (size: CanvasSize) => void;
  maxDpr?: number;
}): () => void {
  const { container, canvases, onResize, maxDpr = 2.5 } = options;
  let raf = 0;
  let lastEntry: ResizeObserverEntry | undefined;
  let dprMql: MediaQueryList | null = null;

  const update = (): void => {
    raf = 0;
    const { width: cssWidth, height: cssHeight } = measureElement(container);
    const dpr = getDisplayDpr(maxDpr);
    const { pixelWidth, pixelHeight } = readPixelBox(lastEntry, cssWidth, cssHeight, dpr);

    for (const canvas of canvases) {
      applyCanvasBackingStore(canvas, cssWidth, cssHeight, dpr, pixelWidth, pixelHeight);
    }
    onResize({ cssWidth, cssHeight, pixelWidth, pixelHeight, dpr });
  };

  const schedule = (entries?: ResizeObserverEntry[]): void => {
    if (entries?.[0]) lastEntry = entries[0];
    if (!raf) raf = requestAnimationFrame(update);
  };

  const bindDprListener = (): void => {
    dprMql?.removeEventListener('change', onDprChange);
    const dpr = window.devicePixelRatio || 1;
    dprMql = window.matchMedia(`(resolution: ${dpr}dppx)`);
    dprMql.addEventListener('change', onDprChange);
  };

  const onDprChange = (): void => {
    lastEntry = undefined;
    bindDprListener();
    schedule();
  };

  let ro: ResizeObserver;
  try {
    ro = new ResizeObserver((entries) => schedule(entries));
    ro.observe(container, { box: 'device-pixel-content-box' } as ResizeObserverOptions);
  } catch {
    ro = new ResizeObserver((entries) => schedule(entries));
    ro.observe(container);
  }

  const onViewportChange = (): void => schedule();
  window.addEventListener('orientationchange', onViewportChange);
  window.addEventListener('resize', onViewportChange);
  window.visualViewport?.addEventListener('resize', onViewportChange);
  window.visualViewport?.addEventListener('scroll', onViewportChange);
  bindDprListener();

  update();

  return () => {
    ro.disconnect();
    dprMql?.removeEventListener('change', onDprChange);
    window.removeEventListener('orientationchange', onViewportChange);
    window.removeEventListener('resize', onViewportChange);
    window.visualViewport?.removeEventListener('resize', onViewportChange);
    window.visualViewport?.removeEventListener('scroll', onViewportChange);
    if (raf) cancelAnimationFrame(raf);
  };
}
