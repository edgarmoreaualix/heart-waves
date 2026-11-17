// 09 · Heart waves — play with an Aliev-Panfilov sheet of heart cells (same model as sim.py in this repo).
import { THREE, makeStage, COLORS, slider, select, check, buttons, Plot } from './shared/kit.js';
import * as C from './core.js';

const N = 160, S = 8, H = 0.9;                         // grid, sheet size (world), voltage → height
const P = { ...C.DEFAULTS, speed: 25, running: true, pace: true, period: 80, tool: 'spark', ghost: true, tipsOn: true, trails: true,
  pacing: 'off', strength: 0.66, untimedPeriod: 30 };
const SHOCK_I = 1.0, SHOCK_TU = 2.0;                  // defibrillator: a global pulse ≈ 5.5× threshold
let pacer = new C.FeedbackPacer(), shockUntil = -1, energy = 0, peak = 0;
const T = C.createTissue(N);
const E = C.makeElectrode(T, 0.85, 0.5, 4);
let probe = [80, 80], lastPace = -1e9, events = [];

const stage = makeStage(document.getElementById('stage'), { cam: [0, 7.5, 8.5], target: [0, 0, 0.4], grid: false });
const { scene } = stage;

// ---- the tissue: one vertex per cell ----
const geo = new THREE.PlaneGeometry(S, S, N - 1, N - 1).rotateX(-Math.PI / 2); // vertex i = cell (x=i%N, y=⌊i/N⌋)
const col = new Float32Array(N * N * 3);
geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, H / 2, 0), S);
const sheet = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, metalness: 0.05, side: THREE.DoubleSide }));
scene.add(sheet);
// recovery ("battery debt") hangs underneath as a violet ghost
const ggeo = new THREE.PlaneGeometry(S, S, N - 1, N - 1).rotateX(-Math.PI / 2);
ggeo.boundingSphere = geo.boundingSphere.clone();
const ghost = new THREE.Mesh(ggeo, new THREE.MeshBasicMaterial({ color: COLORS.violet, transparent: true, opacity: 0.28, side: THREE.DoubleSide, depthWrite: false }));
scene.add(ghost);
const frame = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(S, 0.001, S)), new THREE.LineBasicMaterial({ color: 0x2a323d }));
scene.add(frame);

const W = (gx, gy) => [(gx / (N - 1) - 0.5) * S, (gy / (N - 1) - 0.5) * S]; // grid → world x,z
const G = (p) => [Math.round((p.x / S + 0.5) * (N - 1)), Math.round((p.z / S + 0.5) * (N - 1))];

// probe (cyan pin) + ECG electrode (gold ball hovering above)
const pin = new THREE.Mesh(new THREE.SphereGeometry(0.09, 16, 12), new THREE.MeshStandardMaterial({ color: COLORS.cyan, emissive: COLORS.cyan, emissiveIntensity: 0.3 }));
const pinPole = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 1, 6), new THREE.MeshBasicMaterial({ color: COLORS.cyan }));
scene.add(pin, pinPole);
const elec = new THREE.Mesh(new THREE.SphereGeometry(0.14, 20, 14), new THREE.MeshStandardMaterial({ color: COLORS.gold, emissive: COLORS.gold, emissiveIntensity: 0.4 }));
{ const [x, z] = W(E.fx * N, E.fy * N); elec.position.set(x, 4 * S / N + 1.2, z); }
scene.add(elec);
// pacemaker strip marker
const sa = new THREE.Mesh(new THREE.BoxGeometry(4 * S / N, 0.04, S), new THREE.MeshBasicMaterial({ color: COLORS.green, transparent: true, opacity: 0.5 }));
sa.position.set(-S / 2 + 2 * S / N, 0.02, 0); scene.add(sa);
// spiral tips: rings coloured by sense of rotation (red ↻ clockwise = sign −1, blue ↺ counter-clockwise = sign +1)
const tipMeshes = Array.from({ length: 16 }, () => {
  const m = new THREE.Mesh(new THREE.TorusGeometry(0.16, 0.035, 8, 24).rotateX(Math.PI / 2),
    new THREE.MeshBasicMaterial({ color: COLORS.red, depthTest: false, transparent: true }));
  m.renderOrder = 3; m.visible = false; scene.add(m); return m;
});
// tip trails: fading dots drawn on top of everything (like the white tip trajectories in the paper)
const TRAIL_MAX = 1500;
const trailGeo = new THREE.BufferGeometry();
const trailPos = new Float32Array(TRAIL_MAX * 3), trailCol = new Float32Array(TRAIL_MAX * 3);
trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
trailGeo.setAttribute('color', new THREE.BufferAttribute(trailCol, 3));
const trailPts = new THREE.Points(trailGeo, new THREE.PointsMaterial({ size: 0.045, vertexColors: true, depthTest: false, transparent: true }));
trailPts.renderOrder = 2; trailPts.frustumCulled = false; scene.add(trailPts);
let trail = []; // {x, y, sign, t}
const RED = new THREE.Color(COLORS.red), BLUE = new THREE.Color(COLORS.blue), BG = new THREE.Color(COLORS.bg);
function drawTrails() {
  const k = Math.min(trail.length, TRAIL_MAX), now = T.t, c = new THREE.Color();
  for (let j = 0; j < k; j++) {
    const p = trail[trail.length - k + j], [x, z] = W(p.x, p.y);
    trailPos[3 * j] = x; trailPos[3 * j + 1] = 0.05; trailPos[3 * j + 2] = z;
    const age = Math.min(1, (now - p.t) / 1500);           // fade over ~19 s of heart time
    c.copy(p.sign < 0 ? RED : BLUE).lerp(BG, age * 0.85);
    trailCol[3 * j] = c.r; trailCol[3 * j + 1] = c.g; trailCol[3 * j + 2] = c.b;
  }
  trailGeo.setDrawRange(0, k);
  trailGeo.attributes.position.needsUpdate = true; trailGeo.attributes.color.needsUpdate = true;
}

// ---- colours: rest slate → fired orange/gold/white; refractory tinted violet; scar grey ----
const REST = [0.02, 0.035, 0.06], VIO = [0.22, 0.08, 0.42], SCAR = [0.16, 0.18, 0.21];
function colour(u, v, out, o) {
  if (u > 0.08) {
    const s = Math.min(1, u);
    out[o] = 0.25 + 0.75 * Math.min(1, s * 1.6); out[o + 1] = 0.12 + 0.75 * s * s; out[o + 2] = 0.08 + 0.55 * Math.max(0, s - 0.7) * 3;
    return;
  }
  const r = Math.min(1, v / 1.6);
  for (let k = 0; k < 3; k++) out[o + k] = REST[k] + (VIO[k] - REST[k]) * r * 0.85;
}

function redraw() {
  const pos = geo.attributes.position.array, gpos = ggeo.attributes.position.array;
  for (let i = 0; i < N * N; i++) {
    if (!T.tissue[i]) { pos[3 * i + 1] = -0.03; col[3 * i] = SCAR[0]; col[3 * i + 1] = SCAR[1]; col[3 * i + 2] = SCAR[2]; gpos[3 * i + 1] = -0.03; continue; }
    const u = T.u[i], v = T.v[i];
    pos[3 * i + 1] = Math.max(0, u) * H;
    gpos[3 * i + 1] = -0.04 - Math.min(v, 2.5) * 0.28;
    colour(u, v, col, 3 * i);
  }
  geo.attributes.position.needsUpdate = true; geo.attributes.color.needsUpdate = true; geo.computeVertexNormals();
  ggeo.attributes.position.needsUpdate = true;
  const pi = probe[1] * N + probe[0], [px, pz] = W(...probe), hy = Math.max(0, T.u[pi]) * H;
  pin.position.set(px, hy + 0.45, pz); pinPole.position.set(px, (hy + 0.45) / 2, pz); pinPole.scale.y = hy + 0.45;
}

// ---- knobs ----
const knobs = document.getElementById('knobs');
buttons(knobs, [
  ['♥ Heartbeat', () => preset('beat'), 'primary'],
  ['Spiral ↻', () => preset('cw')],
  ['Spiral ↺', () => preset('ccw')],
  ['Figure-8', () => preset('fig8')],
  ['4 spirals', () => preset('quad')],
  ['Scar + beat', () => preset('scar')],
  ['Scar loop', () => preset('scarloop')],
  ['Fibrosis', () => preset('fib')],
]);
buttons(knobs, [
  ['⚡ Defibrillate', () => { shockUntil = T.t + SHOCK_TU; }, 'primary'],
  ['Calm', () => { C.reset(T); events = []; plotE.clear(); plotC.clear(); }],
  ['Heal scar', () => C.clearScar(T)],
  ['Pause', (b) => { P.running = !P.running; b.textContent = P.running ? 'Pause' : 'Play'; }],
]);
const toolSel = select(knobs, { label: 'Your mouse is…', value: P.tool, onChange: (v) => (P.tool = v), options: {
  spark: 'an electrode: drag to fire cells', scar: 'a scar: drag to kill tissue', heal: 'a healer: drag to restore tissue',
  probe: 'a probe / feedback sensor: click a cell to listen to it', view: 'a camera: drag to orbit' } });
const paceChk = check(knobs, { label: 'Pacemaker (SA node, green strip) on', value: P.pace, onChange: (v) => (P.pace = v) });
const bpmOf = (per) => 60000 / (per * C.TU_MS);
slider(knobs, { label: 'Pacemaker period', min: 30, max: 200, value: P.period, fmt: (v) => `${v.toFixed(0)} t.u. = ${bpmOf(v).toFixed(0)} bpm`, onInput: (v) => (P.period = v) });
slider(knobs, { label: 'Firing threshold a (how easily a cell is triggered)', min: 0.05, max: 0.3, value: P.a, onInput: (v) => (P.a = v) });
slider(knobs, { label: 'Battery recharge speed ε₀ (shorter rest)', min: 0.0005, max: 0.02, value: P.eps0, log: true, onInput: (v) => (P.eps0 = v) });
slider(knobs, { label: 'Cell-to-cell coupling D (gap junctions)', min: 0.15, max: 1.5, value: P.D, onInput: (v) => (P.D = v) });
slider(knobs, { label: 'Simulation speed', min: 2, max: 60, value: P.speed, fmt: (v) => `${(v * C.TU_MS / 1000).toFixed(2)}× real time`, onInput: (v) => (P.speed = v) });
check(knobs, { label: 'Show recovery (battery debt) underneath', value: P.ghost, onChange: (v) => (ghost.visible = P.ghost = v) });
check(knobs, { label: 'Mark spiral tips (red ↻ clockwise, blue ↺ counter-clockwise)', value: P.tipsOn, onChange: (v) => (P.tipsOn = v) });
check(knobs, { label: 'Draw tip trails', value: P.trails, onChange: (v) => { P.trails = v; trailPts.visible = v; } });

const pk = document.createElement('div');
pk.innerHTML = '<h2 style="margin-top:14px">Gentle feedback pacing <span class="muted" style="text-transform:none;letter-spacing:0">(Roshan &amp; Majumder 2025)</span></h2>';
knobs.appendChild(pk);
const pacingSel = select(pk, { label: 'Whole-sheet weak pulses', value: P.pacing, onChange: (v) => { P.pacing = v; pacer = new C.FeedbackPacer(); }, options: {
  off: 'off', feedback: 'timed by feedback: pulse each time a wave passes the cyan sensor (rF-ATP)',
  untimed: 'untimed: same pulses on a fixed clock (control)' } });
slider(pk, { label: 'Pulse strength', min: 0.1, max: 1.5, value: P.strength, fmt: (v) => `${(v * 100).toFixed(0)} % of firing threshold${v >= 1 ? ' (a mini-shock!)' : ''}`, onInput: (v) => (P.strength = v) });
slider(pk, { label: 'Untimed clock period', min: 15, max: 80, value: P.untimedPeriod, fmt: (v) => `${(v * C.TU_MS).toFixed(0)} ms`, onInput: (v) => (P.untimedPeriod = v) });
buttons(pk, [['Clear trails', () => { trail = []; drawTrails(); }], ['Reset energy meter', () => { energy = 0; peak = 0; pacer.pulses = 0; }]]);

const plotEl = document.getElementById('plot');
const plotE = new Plot(plotEl, { title: 'pseudo-ECG at the gold electrode', series: [{ name: 'ECG', color: COLORS.gold }] });
const plotC = new Plot(plotEl, { title: 'one cell (cyan probe)', series: [{ name: 'voltage u', color: COLORS.cyan }, { name: 'recovery v', color: COLORS.violet }, { name: 'global pulse ×4', color: COLORS.red }] });

function preset(name) {
  C.reset(T); C.clearScar(T); events = []; lastPace = -1e9; plotE.clear(); plotC.clear();
  trail = []; pacer = new C.FeedbackPacer(); shockUntil = -1; energy = 0; peak = 0;
  if (name === 'beat') { paceChk.set(true); }
  if (C.RECIPES[name]) { paceChk.set(false); C.paceLeft(T); events.push({ t: C.S2_AT, fn: () => C.stimulate(T, C.RECIPES[name](T)) }); }
  if (name === 'scar') { C.setScar(T, C.disc(T, N * 0.5, N * 0.5, N * 0.14), true); paceChk.set(true); }
  if (name === 'scarloop') {
    C.setScar(T, C.disc(T, N * 0.5, N * 0.5, N * 0.14), true); paceChk.set(false); C.paceLeft(T);
    events.push({ t: 59, fn: () => C.stimulate(T, C.rect(T, N >> 1, 102, (N >> 1) + 4, N)) });
  }
  if (name === 'fib') { C.fibrosis(T, 0.3, (Math.random() * 1e9) | 0); paceChk.set(true); }
  const s2 = 'S1 from the left, then S2 at 45 t.u. (580 ms) inside the vulnerable window';
  flash({ cw: s2 + ' → one clockwise spiral', ccw: s2 + ' (mirrored) → one counter-clockwise spiral',
    fig8: s2 + ': a short line away from the edges → two counter-rotating spirals',
    quad: s2 + ': two short lines → four spirals (two figure-8s side by side)',
    scarloop: 'S1, then a line below the scar at 59 t.u. (760 ms): only one direction is open → the wave circles the scar' }[name] ?? '');
}

// ---- mouse tools: drag on the sheet paints; elsewhere (or tool=camera) orbits ----
const stageEl = document.getElementById('stage');
let painting = false;
function applyTool(ev) {
  const hit = stage.pick(ev, [sheet]);
  if (!hit) return false;
  const [gx, gy] = G(hit.point);
  if (P.tool === 'spark') C.stimulate(T, C.disc(T, gx, gy, 3), 1.0);
  if (P.tool === 'scar') C.setScar(T, C.disc(T, gx, gy, 4), true);
  if (P.tool === 'heal') C.setScar(T, C.disc(T, gx, gy, 5), false);
  if (P.tool === 'probe') { probe = [Math.min(N - 1, Math.max(0, gx)), Math.min(N - 1, Math.max(0, gy))]; plotC.clear(); meter = new C.RateMeter(); pacer.prev = T.u[probe[1] * N + probe[0]]; }
  return true;
}
stageEl.addEventListener('pointerdown', (ev) => {
  if (P.tool === 'view' || ev.button !== 0) return;
  if (applyTool(ev)) { painting = P.tool !== 'probe'; stage.controls.enabled = false; }
}, { capture: true });
stageEl.addEventListener('pointermove', (ev) => { if (painting) applyTool(ev); });
addEventListener('pointerup', () => { painting = false; stage.controls.enabled = true; });

// ---- loop ----
const hud = document.getElementById('hud');
let meter = new C.RateMeter(), tipList = [], frameNo = 0, flashMsg = '', flashUntil = 0;
function flash(m) { flashMsg = m; flashUntil = performance.now() + 5000; }
function status(active, nt) {
  if (active === 0) return 'quiet: every cell resting';
  if (nt >= 5) return 'CHAOS: many wavelets (fibrillation-like)';
  if (nt === 0 && !P.pace && T.t > 300) return 're-entry: a wave circling the scar (no free tip)';
  if (nt >= 1) return P.pace ? 're-entry: a spiral is fighting the pacemaker' : 're-entry: a spiral drives the heart by itself';
  return P.pace ? 'normal beat: waves from the pacemaker' : 'a wave is crossing';
}

stage.onFrame((dt) => {
  if (P.running) {
    const steps = Math.max(1, Math.round((P.speed * dt) / P.dt));  // speed may exceed the slider via #speed=
    for (let s = 0; s < steps; s++) {
      if (P.pace && T.t - lastPace >= P.period) { C.paceLeft(T); lastPace = T.t; }
      for (const e of events) if (!e.done && T.t >= e.t) { e.fn(); e.done = true; }
      const Ip = pacer.update(T, P.pacing, T.u[probe[1] * N + probe[0]], P.strength * C.GLOBAL_THRESHOLD, P.untimedPeriod);
      T.I = Ip + (T.t < shockUntil ? SHOCK_I : 0);
      if (T.I) { energy += T.I * T.I * P.dt; peak = Math.max(peak, T.I); }
      C.step(T, P, 1);
      if (s % 4 === 0) meter.push(T.t, T.u[probe[1] * N + probe[0]]);
    }
    const pi = probe[1] * N + probe[0];
    plotE.push([C.ecg(T, E)]); plotC.push([T.u[pi], T.v[pi], 4 * T.I]);
    plotE.draw(); plotC.draw();
  }
  if (!P.pace) lastPace = -1e9;
  if (frameNo++ % 4 === 0) {
    tipList = C.tips(T);
    if (P.running) for (const p of tipList) trail.push({ x: p[0], y: p[1], sign: p[2], t: T.t });
    if (trail.length > TRAIL_MAX) trail.splice(0, trail.length - TRAIL_MAX);
    if (P.trails) drawTrails();
  }
  tipMeshes.forEach((m, i) => {
    const p = P.tipsOn ? tipList[i] : null; m.visible = !!p;
    if (p) { const [x, z] = W(p[0], p[1]); m.position.set(x, 0.12, z); m.material.color.setHex(p[2] < 0 ? COLORS.red : COLORS.blue); }
  });
  frame.material.color.setHex(T.I ? COLORS.red : 0x2a323d);
  redraw();
  const active = C.excitedFraction(T), bpm = meter.bpm(T.t), pi = probe[1] * N + probe[0];
  hud.textContent =
    `t        ${(T.t * C.TU_MS / 1000).toFixed(2)} s\n` +
    `state    ${status(active, tipList.length)}\n` +
    `rate     ${bpm ? bpm.toFixed(0) + ' bpm (at probe)' : '—'}\n` +
    `fired    ${(active * 100).toFixed(1)} % of cells\n` +
    `tips     ${tipList.length}  (${tipList.filter((p) => p[2] < 0).length} ↻ red, ${tipList.filter((p) => p[2] > 0).length} ↺ blue)\n` +
    `probe    ${(100 * T.u[pi] - 80).toFixed(0)} mV\n` +
    `pacing   ${P.pacing}${P.pacing !== 'off' ? `, ${pacer.pulses} pulses` : ''}${T.I ? '  ● PULSE' : ''}\n` +
    `delivered energy ${energy.toFixed(3)}  (peak ${(peak / C.GLOBAL_THRESHOLD * 100).toFixed(0)} % of threshold)` +
    (performance.now() < flashUntil ? `\n\n${flashMsg}` : '');
});

// URL hash shortcuts, e.g. #fig8,feedback,pace@400  (preset, pacing mode, and when to switch pacing on, in t.u.)
{
  const h = decodeURIComponent(location.hash.slice(1)).split(',').filter(Boolean);
  preset(h.find((x) => C.RECIPES[x] || ['beat', 'scar', 'scarloop', 'fib'].includes(x)) ?? 'beat');
  const mode = h.find((x) => x === 'feedback' || x === 'untimed');
  const at = +(h.find((x) => x.startsWith('pace@'))?.slice(5) ?? 0);
  const sp = h.find((x) => x.startsWith('speed='));
  if (sp) P.speed = +sp.slice(6);
  if (mode) events.push({ t: at, fn: () => { pacingSel.set(mode); } });
}
