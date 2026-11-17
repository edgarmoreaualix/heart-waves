// kit.js — the tiny toolbox every study shares.
//   makeStage(el)   → three.js scene + orbit camera + render loop
//   knobs: slider / select / check / button / readout
//   Plot           → live line chart (loss curves etc.) on a <canvas>
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
export { THREE };

export const COLORS = {
  gold: 0xf2b84b, cyan: 0x4fc3d9, red: 0xff6b6b, blue: 0x6b9bff,
  green: 0x7bd88f, violet: 0xc792ea, grey: 0x5c6773, bg: 0x0e1116,
};
export const hex = (c) => '#' + c.toString(16).padStart(6, '0');

export function makeStage(el, { cam = [6, 5, 7], target = [0, 0, 0], grid = true, fov = 45 } = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  el.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(COLORS.bg);
  const camera = new THREE.PerspectiveCamera(fov, 1, 0.05, 200);
  camera.position.set(...cam);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(...target);
  controls.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x20242a, 1.1));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(5, 10, 4);
  scene.add(sun);
  if (grid) {
    const g = new THREE.GridHelper(12, 24, 0x2a323d, 0x1a2029);
    g.position.y = -0.001;
    scene.add(g);
  }
  const resize = () => {
    const w = el.clientWidth, h = el.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(el);
  resize();

  const frameFns = [];
  const clock = new THREE.Clock();
  renderer.setAnimationLoop(() => {
    const dt = Math.min(clock.getDelta(), 0.05);
    for (const f of frameFns) f(dt);
    controls.update();
    renderer.render(scene, camera);
  });

  // pick(event, objects) → first ray hit or null (for click-to-place interactions)
  const ray = new THREE.Raycaster();
  const pick = (ev, objects) => {
    const r = renderer.domElement.getBoundingClientRect();
    const p = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(p, camera);
    return ray.intersectObjects(objects, false)[0] ?? null;
  };
  // fire only on a click, not at the end of an orbit drag
  const onClick = (fn) => {
    let down = null;
    renderer.domElement.addEventListener('pointerdown', (e) => (down = [e.clientX, e.clientY]));
    renderer.domElement.addEventListener('pointerup', (e) => {
      if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 5) fn(e);
      down = null;
    });
  };
  return { scene, camera, renderer, controls, onFrame: (f) => frameFns.push(f), pick, onClick };
}

// ---------- knobs ----------
const fmtDefault = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(2));

/** slider({label, min, max, value, step, log, fmt, onInput}) → {get, set}. log=true makes the slider exponential. */
export function slider(parent, { label, min, max, value, step, log = false, fmt = fmtDefault, onInput = () => {} }) {
  const box = document.createElement('div');
  box.className = 'knob';
  box.innerHTML = `<label>${label}</label><span class="val"></span><input type="range">`;
  const [, val, input] = box.children;
  const toPos = (v) => (log ? Math.log10(v) : v);
  const fromPos = (p) => (log ? 10 ** p : p);
  input.min = toPos(min); input.max = toPos(max);
  input.step = log ? (toPos(max) - toPos(min)) / 200 : step ?? (max - min) / 200;
  let v = value;
  const show = () => { input.value = toPos(v); val.textContent = fmt(v); };
  input.addEventListener('input', () => { v = fromPos(+input.value); val.textContent = fmt(v); onInput(v); });
  show();
  parent.appendChild(box);
  return { get: () => v, set: (x) => { v = x; show(); onInput(v); } };
}

export function select(parent, { label, options, value, onChange = () => {} }) {
  const box = document.createElement('div');
  box.className = 'knob';
  box.innerHTML = `<label>${label}</label><span></span><select></select>`;
  const sel = box.querySelector('select');
  for (const [k, name] of Object.entries(options)) sel.add(new Option(name, k));
  sel.value = value;
  sel.addEventListener('change', () => onChange(sel.value));
  parent.appendChild(box);
  return { get: () => sel.value, set: (x) => { sel.value = x; onChange(x); } };
}

export function check(parent, { label, value = false, onChange = () => {} }) {
  const l = document.createElement('label');
  l.className = 'check';
  l.innerHTML = `<input type="checkbox"> <span>${label}</span>`;
  const input = l.querySelector('input');
  input.checked = value;
  input.addEventListener('change', () => onChange(input.checked));
  parent.appendChild(l);
  return { get: () => input.checked, set: (x) => { input.checked = x; onChange(x); } };
}

export function buttons(parent, specs) {
  const row = document.createElement('div');
  row.className = 'row';
  const out = {};
  for (const [label, fn, cls] of specs) {
    const b = document.createElement('button');
    b.textContent = label;
    if (cls) b.className = cls;
    b.addEventListener('click', () => fn(b));
    row.appendChild(b);
    out[label] = b;
  }
  parent.appendChild(row);
  return out;
}

// ---------- live plot ----------
export class Plot {
  /** series: [{name, color}], logY: plot log10(value) */
  constructor(parent, { series, logY = false, maxPoints = 600, title = '' }) {
    this.series = series.map((s) => ({ ...s, data: [] }));
    this.logY = logY; this.maxPoints = maxPoints; this.title = title;
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'plot';
    parent.appendChild(this.canvas);
    this.legend = document.createElement('div');
    this.legend.className = 'legend';
    parent.appendChild(this.legend);
    this.renderLegend();
  }
  renderLegend() {
    this.legend.innerHTML = this.series.map((s) => `<span><i style="background:${hex(s.color)}"></i>${s.name}</span>`).join('');
  }
  setSeries(series) { this.series = series.map((s) => ({ ...s, data: [] })); this.renderLegend(); this.draw(); }
  clear() { for (const s of this.series) s.data.length = 0; this.draw(); }
  push(values) {
    values.forEach((v, i) => {
      const d = this.series[i]?.data;
      if (!d || v == null || !Number.isFinite(v)) return;
      d.push(this.logY ? Math.log10(Math.max(v, 1e-12)) : v);
      if (d.length > this.maxPoints) d.shift();
    });
  }
  draw() {
    const c = this.canvas, dpr = devicePixelRatio;
    const W = (c.width = c.clientWidth * dpr), H = (c.height = c.clientHeight * dpr);
    const g = c.getContext('2d');
    g.clearRect(0, 0, W, H);
    let lo = Infinity, hi = -Infinity, n = 0;
    for (const s of this.series) for (const v of s.data) { lo = Math.min(lo, v); hi = Math.max(hi, v); n = Math.max(n, s.data.length); }
    g.font = `${10 * dpr}px monospace`;
    g.fillStyle = '#8b96a3';
    if (this.title) g.fillText(this.title, 6 * dpr, 12 * dpr);
    if (!Number.isFinite(lo)) return;
    if (hi - lo < 1e-9) { hi += 0.5; lo -= 0.5; }
    const pad = 16 * dpr;
    const y = (v) => H - pad - ((v - lo) / (hi - lo)) * (H - 2 * pad);
    const lab = (v) => (this.logY ? `1e${v.toFixed(1)}` : v.toPrecision(3));
    g.fillText(lab(hi), W - 60 * dpr, pad - 3 * dpr);
    g.fillText(lab(lo), W - 60 * dpr, H - 4 * dpr);
    for (const s of this.series) {
      if (s.data.length < 2) continue;
      g.strokeStyle = hex(s.color); g.lineWidth = 1.5 * dpr; g.beginPath();
      s.data.forEach((v, i) => {
        const x = (i / Math.max(n - 1, 1)) * W;
        i ? g.lineTo(x, y(v)) : g.moveTo(x, y(v));
      });
      g.stroke();
    }
  }
}

// gaussian noise (Box–Muller)
export function randn() {
  let u = 0; while (!u) u = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}

// a line whose points you can update cheaply (trails, rollouts)
export function dynLine(scene, maxPts, color, opacity = 1) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(maxPts * 3), 3));
  geo.setDrawRange(0, 0);
  const mat = new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity });
  const line = new THREE.Line(geo, mat);
  line.frustumCulled = false;
  scene.add(line);
  return {
    line,
    set(points) { // points: array of [x,y,z]
      const a = geo.attributes.position.array, k = Math.min(points.length, maxPts);
      for (let i = 0; i < k; i++) { a[3 * i] = points[i][0]; a[3 * i + 1] = points[i][1]; a[3 * i + 2] = points[i][2]; }
      geo.attributes.position.needsUpdate = true;
      geo.setDrawRange(0, k);
    },
  };
}

// text label that always faces the camera
export function textSprite(text, { color = '#dfe5ec', size = 0.3, font = '600 48px system-ui' } = {}) {
  const c = document.createElement('canvas'), g = c.getContext('2d');
  g.font = font;
  const w = Math.ceil(g.measureText(text).width) + 16;
  c.width = w; c.height = 64;
  g.font = font; g.fillStyle = color; g.textBaseline = 'middle';
  g.fillText(text, 8, 34);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthWrite: false, transparent: true }));
  s.scale.set((size * w) / 64, size, 1);
  return s;
}

// a cylinder "limb" you can stretch between two points each frame
export function limb(scene, color, radius = 0.05, opacity = 1) {
  const m = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, 1, 12),
    new THREE.MeshStandardMaterial({ color, roughness: 0.5, transparent: opacity < 1, opacity }),
  );
  scene.add(m);
  const up = new THREE.Vector3(0, 1, 0), d = new THREE.Vector3();
  return {
    mesh: m,
    set(a, b) { // a, b: THREE.Vector3
      d.subVectors(b, a);
      const L = d.length();
      m.position.copy(a).addScaledVector(d, 0.5);
      m.scale.set(1, Math.max(L, 1e-4), 1);
      if (L > 1e-6) m.quaternion.setFromUnitVectors(up, d.divideScalar(L));
    },
  };
}
