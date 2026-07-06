// controls.js — wires the DOM controls to the params object.

export function setupControls(params, { onReset }) {
  const $ = (id) => document.getElementById(id);

  // Latent-heat toggle: the proof-of-mechanism switch.
  const latent = $('latent-toggle');
  latent.checked = params.latentOn;
  latent.addEventListener('change', () => {
    params.latentOn = latent.checked;
    $('latent-state').textContent = params.latentOn
      ? 'ON — evaporative cooling → downdraft'
      : 'OFF — momentum only → updraft';
  });

  // Overlay selector.
  const modes = [
    ['ov-none', 0],
    ['ov-vel', 1],
    ['ov-temp', 2],
    ['ov-hum', 3],
  ];
  function setOverlay(mode) {
    params.overlay = mode;
    for (const [id, m] of modes) {
      $(id).classList.toggle('active', m === mode);
    }
  }
  for (const [id, m] of modes) {
    $(id).addEventListener('click', () => setOverlay(m));
  }
  setOverlay(params.overlay);

  const arrows = $('show-arrows');
  arrows.checked = params.showArrows;
  arrows.addEventListener('change', () => (params.showArrows = arrows.checked));

  const dropsChk = $('show-droplets');
  dropsChk.checked = params.showDroplets;
  dropsChk.addEventListener('change', () => (params.showDroplets = dropsChk.checked));

  // Sliders: [id, param key, format].
  const sliders = [
    ['emit-speed', 'emitSpeed', (v) => `${v.toFixed(1)} m/s`],
    ['emit-spread', 'emitSpreadDeg', (v) => `±${v.toFixed(0)}°`],
    ['emit-angle', 'emitAngleDeg', (v) => `${v.toFixed(0)}°`],
    ['emit-rate', 'emitRate', (v) => `${(v / 1000).toFixed(1)}k/s`],
    ['r-min', 'rMinUm', (v) => `${v.toFixed(0)} µm`],
    ['r-max', 'rMaxUm', (v) => `${(v / 1000).toFixed(2)} mm`],
    ['t-amb', 'tAmb', (v) => `${v.toFixed(0)} °C`],
    ['rh-amb', 'rhAmb', (v) => `${v.toFixed(0)} %`],
  ];
  for (const [id, key, fmt] of sliders) {
    const el = $(id);
    const label = $(id + '-val');
    el.value = params[key];
    label.textContent = fmt(params[key]);
    el.addEventListener('input', () => {
      params[key] = parseFloat(el.value);
      // keep the size range ordered (and reflect it in the other slider)
      if (params.rMinUm > params.rMaxUm) {
        const [otherId, otherKey] = key === 'rMinUm'
          ? ['r-max', 'rMaxUm'] : ['r-min', 'rMinUm'];
        params[otherKey] = params[key];
        const other = $(otherId);
        other.value = params[otherKey];
        other.dispatchEvent(new Event('input'));
      }
      label.textContent = fmt(params[key]);
    });
  }

  const pauseBtn = $('pause');
  pauseBtn.addEventListener('click', () => {
    params.paused = !params.paused;
    pauseBtn.textContent = params.paused ? 'Resume' : 'Pause';
  });

  $('reset').addEventListener('click', onReset);
}
