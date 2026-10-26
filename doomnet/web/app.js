// Generic controls/timing/transport/presentation. No map, state fields, or game drawing.
const canvas = document.querySelector('#frame');
const context = canvas.getContext('2d', {alpha: false});
const toggle = document.querySelector('#toggle');
const overlay = document.querySelector('#overlay');
const status = document.querySelector('#status');
const metrics = document.querySelector('#metrics');
const error = document.querySelector('#error');
const mouse = document.querySelector('#mouse');
const pressed = new Set();
let active = false, busy = false, accumulator = 0, previous = performance.now();
let weapon = -1, mouseTurn = 0, mouseFire = false, spec, image;
let lastPaint = performance.now(), paintCount = 0, measuredFps = 0;

function setActive(value) {
  active = value;
  accumulator = 0;
  previous = performance.now();
  pressed.clear();
  mouseFire = false;
  mouseTurn = 0;
  toggle.textContent = active ? 'Pause' : 'Start';
  overlay.hidden = active;
  if (active) canvas.focus();
  else if (document.pointerLockElement === canvas) document.exitPointerLock();
}

function controls() {
  const down = code => Number(pressed.has(code));
  const result = [down('KeyW') - down('KeyS'), down('KeyD') - down('KeyA'),
    Math.max(-1, Math.min(1, down('ArrowLeft') - down('ArrowRight') + mouseTurn)),
    Math.max(down('Space'), Number(mouseFire)), down('KeyE'), down('KeyR'), weapon,
    Math.max(down('ShiftLeft'), down('ShiftRight'))];
  weapon = -1;
  mouseTurn = 0;
  return result;
}

async function step(input, dt, steps = 1) {
  const began = performance.now();
  const response = await fetch('/step', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({controls: input, dt, steps})});
  if (!response.ok) throw new Error((await response.json()).error || `Step failed: ${response.status}`);
  const rgb = new Uint8Array(await response.arrayBuffer());
  if (rgb.length !== spec.width * spec.height * 3) throw new Error('RGB output has the wrong byte length');
  // Canvas accepts RGBA; expand the already completed RGB output without drawing a scene.
  for (let source = 0, target = 0; source < rgb.length; source += 3, target += 4) {
    image.data[target] = rgb[source];
    image.data[target + 1] = rgb[source + 1];
    image.data[target + 2] = rgb[source + 2];
    image.data[target + 3] = 255;
  }
  context.putImageData(image, 0, 0);
  paintCount++;
  const elapsed = performance.now() - lastPaint;
  if (elapsed >= 1000) { measuredFps = paintCount * 1000 / elapsed; paintCount = 0; lastPaint = performance.now(); }
  metrics.textContent = `${measuredFps.toFixed(1)} displayed fps · ${(performance.now() - began).toFixed(0)} ms / ${steps} step${steps === 1 ? '' : 's'} · ${spec.width}×${spec.height}`;
}

function fail(problem) {
  setActive(false);
  error.textContent = problem.message;
  error.hidden = false;
  status.textContent = 'Execution stopped';
  toggle.disabled = true;
}

async function loop(now) {
  const elapsed = (now - previous) / 1000;
  previous = now;
  if (active) accumulator = Math.min(0.25, accumulator + elapsed);
  if (active && !busy && accumulator >= spec.max_dt) {
    const steps = Math.min(8, Math.floor(accumulator / spec.max_dt));
    accumulator -= steps * spec.max_dt;
    busy = true;
    try { await step(controls(), spec.max_dt, steps); }
    catch (problem) { fail(problem); }
    finally { busy = false; }
  }
  requestAnimationFrame(loop);
}

toggle.addEventListener('click', () => setActive(!active));
const validKeys = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowLeft', 'ArrowRight', 'Space', 'KeyE', 'KeyR', 'ShiftLeft', 'ShiftRight']);
window.addEventListener('keydown', event => {
  if (event.code === 'KeyP' || event.code === 'Escape') {
    event.preventDefault(); if (!event.repeat && !toggle.disabled) setActive(event.code === 'Escape' ? false : !active); return;
  }
  if (!active) return;
  if (validKeys.has(event.code) || ['Digit1', 'Digit2', 'Digit3'].includes(event.code)) event.preventDefault();
  if (validKeys.has(event.code)) pressed.add(event.code);
  if (['Digit1', 'Digit2', 'Digit3'].includes(event.code)) weapon = Number(event.code.slice(-1)) - 1;
});
window.addEventListener('keyup', event => pressed.delete(event.code));
window.addEventListener('blur', () => setActive(false));
document.addEventListener('visibilitychange', () => { if (document.hidden) setActive(false); });
canvas.addEventListener('mousedown', event => {
  if (!active || event.button !== 0) return;
  mouseFire = true;
  if (mouse.checked && document.pointerLockElement !== canvas) canvas.requestPointerLock();
});
window.addEventListener('mouseup', () => { mouseFire = false; });
document.addEventListener('mousemove', event => {
  if (active && document.pointerLockElement === canvas) mouseTurn -= event.movementX * 0.03;
});
mouse.addEventListener('change', () => { if (!mouse.checked && document.pointerLockElement === canvas) document.exitPointerLock(); });
for (const button of document.querySelectorAll('[data-key]')) {
  button.addEventListener('pointerdown', event => {
    event.preventDefault(); if (!active) return;
    button.setPointerCapture(event.pointerId); pressed.add(button.dataset.key); button.classList.add('held');
  });
  const release = () => { pressed.delete(button.dataset.key); button.classList.remove('held'); };
  button.addEventListener('pointerup', release); button.addEventListener('pointercancel', release);
}
for (const button of document.querySelectorAll('[data-weapon]')) button.addEventListener('click', () => { if (active) weapon = Number(button.dataset.weapon); });

try {
  const response = await fetch('/status');
  if (!response.ok) throw new Error(`Status failed: ${response.status}`);
  spec = await response.json();
  canvas.width = spec.width; canvas.height = spec.height;
  canvas.parentElement.style.aspectRatio = `${spec.width} / ${spec.height}`;
  image = context.createImageData(spec.width, spec.height);
  status.textContent = `${spec.execution.device.toUpperCase()} · ${spec.execution.cpu_threads} thread${spec.execution.cpu_threads === 1 ? '' : 's'}${spec.recording ? ' · recording' : ''}`;
  await step([0, 0, 0, 0, 0, 0, -1, 0], 0);
  toggle.disabled = false;
  requestAnimationFrame(loop);
} catch (problem) { fail(problem); }
