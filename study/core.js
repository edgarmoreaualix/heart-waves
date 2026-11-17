// core.js — Aliev-Panfilov excitable tissue, the same model as ../sim.py.
// Pure logic (no DOM) so node can test it.
//   du/dt = D·lap(u) − k·u(u−a)(u−1) − u·v      u = voltage (0 rest … 1 fired)
//   dv/dt = (eps0 + mu1·v/(u+mu2))·(−v − k·u(u−a−1))   v = recovery ("battery recharging")
// Scar cells are holes: no current enters or leaves them (proper no-flux, not just a sink).

export const TU_MS = 12.9; // 1 time unit ≈ 12.9 ms; V_mV = 100·u − 80

export const DEFAULTS = { D: 1.0, k: 8.0, a: 0.15, eps0: 0.002, mu1: 0.2, mu2: 0.3, dt: 0.04, dx: 0.5 };

export function createTissue(n = 160) {
  const N = n * n;
  return {
    n, t: 0,
    u: new Float32Array(N), v: new Float32Array(N), du: new Float32Array(N),
    tissue: new Uint8Array(N).fill(1),
    stims: [], // {cells: Int32Array, until}
    I: 0,      // global current injected into every living cell (feedback pacing / shock)
  };
}

export function reset(T) { T.u.fill(0); T.v.fill(0); T.stims.length = 0; T.t = 0; }

const idx = (T, x, y) => y * T.n + x;

/** cells inside a disc (grid coords) */
export function disc(T, cx, cy, r) {
  const out = [];
  const r2 = r * r;
  for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(T.n - 1, Math.ceil(cy + r)); y++)
    for (let x = Math.max(0, Math.floor(cx - r)); x <= Math.min(T.n - 1, Math.ceil(cx + r)); x++)
      if ((x - cx) ** 2 + (y - cy) ** 2 <= r2) out.push(idx(T, x, y));
  return out;
}
export function rect(T, x0, y0, x1, y1) {
  const out = [];
  for (let y = Math.max(0, y0); y < Math.min(T.n, y1); y++) for (let x = Math.max(0, x0); x < Math.min(T.n, x1); x++) out.push(idx(T, x, y));
  return out;
}

/** hold u = 1 on these cells for `dur` time units (an electrode pulse) */
export function stimulate(T, cells, dur = 1.0) { T.stims.push({ cells: Int32Array.from(cells), until: T.t + dur }); }
export const paceLeft = (T) => stimulate(T, rect(T, 0, 0, 4, T.n));            // SA-node strip on the left edge
export const s2Quadrant = (T) => stimulate(T, rect(T, 0, T.n >> 1, T.n >> 1, T.n)); // lower-left quadrant (y down)
/** defibrillator: every living cell fires at once, so the whole sheet goes refractory together */
export function shock(T) { const c = []; for (let i = 0; i < T.u.length; i++) if (T.tissue[i]) c.push(i); stimulate(T, c, 2.0); }

export function setScar(T, cells, scar = true) {
  for (const i of cells) { T.tissue[i] = scar ? 0 : 1; if (scar) { T.u[i] = 0; T.v[i] = 0; } }
}
export function fibrosis(T, frac, seed = 1) {
  let s = seed >>> 0;
  const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  for (let i = 0; i < T.tissue.length; i++) if (rnd() < frac) { T.tissue[i] = 0; T.u[i] = 0; T.v[i] = 0; }
}
export function clearScar(T) { T.tissue.fill(1); }

/** advance `steps` explicit-Euler steps */
export function step(T, P, steps = 1) {
  const { n, u, v, du, tissue } = T;
  const { D, k, a, eps0, mu1, mu2, dt } = P;
  const c = D / (P.dx * P.dx);
  for (let s = 0; s < steps; s++) {
    for (const st of T.stims) for (const i of st.cells) if (tissue[i]) u[i] = 1;
    for (let y = 0; y < n; y++) {
      const row = y * n;
      for (let x = 0; x < n; x++) {
        const i = row + x;
        if (!tissue[i]) { du[i] = 0; continue; }
        const ui = u[i];
        let lap = 0;
        if (x > 0 && tissue[i - 1]) lap += u[i - 1] - ui;
        if (x < n - 1 && tissue[i + 1]) lap += u[i + 1] - ui;
        if (y > 0 && tissue[i - n]) lap += u[i - n] - ui;
        if (y < n - 1 && tissue[i + n]) lap += u[i + n] - ui;
        const vi = v[i];
        du[i] = c * lap - k * ui * (ui - a) * (ui - 1) - ui * vi + T.I;
        v[i] = vi + dt * (eps0 + (mu1 * vi) / (ui + mu2)) * (-vi - k * ui * (ui - a - 1));
      }
    }
    for (let i = 0; i < u.length; i++) u[i] += dt * du[i];
    T.t += dt;
    if (T.stims.length) T.stims = T.stims.filter((st) => st.until > T.t);
  }
}

/** pseudo-ECG: far-field potential at an electrode hovering above the sheet, φ = Σ ∇u·∇(1/r) */
export function makeElectrode(T, fx = 0.85, fy = 0.5, h = 4) {
  const n = T.n, ex = fx * n, ey = fy * n;
  const gx = new Float32Array(n * n), gy = new Float32Array(n * n);
  const inv = (x, y) => 1 / Math.sqrt((x - ex) ** 2 + (y - ey) ** 2 + h * h);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    gx[y * n + x] = (inv(x + 1, y) - inv(x - 1, y)) / 2;
    gy[y * n + x] = (inv(x, y + 1) - inv(x, y - 1)) / 2;
  }
  return { fx, fy, h, gx, gy };
}
export function ecg(T, E) {
  const { n, u } = T;
  let s = 0;
  for (let y = 1; y < n - 1; y++) for (let x = 1; x < n - 1; x++) {
    const i = y * n + x;
    s += ((u[i + 1] - u[i - 1]) / 2) * E.gx[i] + ((u[i + n] - u[i - n]) / 2) * E.gy[i];
  }
  return s;
}

export function excitedFraction(T) {
  let e = 0, m = 0;
  for (let i = 0; i < T.u.length; i++) if (T.tissue[i]) { m++; if (T.u[i] > 0.5) e++; }
  return m ? e / m : 0;
}

/** spiral tips = phase singularities. phase θ = atan2(v − vc, u − uc); a tip is a 2×2 plaquette
 *  whose phase winds by ±2π. Returns [[x, y, sign], …] in grid coords. */
export function tips(T, uc = 0.5, vc = 0.5) {
  const { n, u, v, tissue } = T;
  const th = new Float32Array(n * n);
  for (let i = 0; i < th.length; i++) th[i] = Math.atan2(v[i] - vc, u[i] - uc);
  const wrap = (d) => (d > Math.PI ? d - 2 * Math.PI : d < -Math.PI ? d + 2 * Math.PI : d);
  const out = [];
  for (let y = 1; y < n - 2; y++) for (let x = 1; x < n - 2; x++) {
    const a = y * n + x, b = a + 1, c = a + n + 1, d = a + n;
    if (!(tissue[a] && tissue[b] && tissue[c] && tissue[d])) continue;
    const w = wrap(th[b] - th[a]) + wrap(th[c] - th[b]) + wrap(th[d] - th[c]) + wrap(th[a] - th[d]);
    if (Math.abs(w) > 5) out.push([x + 0.5, y + 0.5, Math.sign(w)]);
  }
  // merge neighbours (one tip often lights 2–3 plaquettes)
  const merged = [];
  for (const p of out) if (!merged.some((q) => Math.hypot(q[0] - p[0], q[1] - p[1]) < 4)) merged.push(p);
  return merged;
}

/** heart rate from the times a cell fires (u crossing 0.5 upward) */
export class RateMeter {
  constructor() { this.prev = 0; this.times = []; }
  push(t, val) {
    if (this.prev < 0.5 && val >= 0.5) { this.times.push(t); if (this.times.length > 6) this.times.shift(); }
    this.prev = val;
  }
  bpm(now) {
    const ts = this.times;
    if (ts.length < 2 || now - ts[ts.length - 1] > 200) return null;
    const dts = ts.slice(1).map((t, i) => t - ts[i]);
    const per = dts.slice(-3).reduce((s, d) => s + d, 0) / Math.min(3, dts.length);
    return 60000 / (per * TU_MS);
  }
}

// ---------- spiral recipes (S1–S2 cross-field, as in Roshan & Majumder 2025, Fig. 1) ----------
// all start with an S1 plane wave from the left at t=0; the S2 fires at t=45 behind its tail.
export const S2_AT = 45;
export const RECIPES = {
  cw:   (T) => rect(T, 0, T.n >> 1, T.n >> 1, T.n),               // lower-left quadrant
  ccw:  (T) => rect(T, 0, 0, T.n >> 1, T.n >> 1),                  // upper-left quadrant
  fig8: (T) => rect(T, 45, 50, 49, 110),                           // short line, away from edges → two tips
  quad: (T) => [...rect(T, 45, 14, 49, 66), ...rect(T, 45, 94, 49, 146)], // two short lines → four tips
};

/** whole-sheet pulse strength that just fires resting tissue (pulse length PULSE_TU) */
export const PULSE_TU = 1.0;
export const GLOBAL_THRESHOLD = 0.18;

/** rF-ATP: each upward crossing of u=0.5 at the sensor → one subthreshold global pulse */
export class FeedbackPacer {
  constructor() { this.prev = 0; this.until = -1; this.lastUntimed = -1e9; this.energy = 0; this.pulses = 0; }
  update(T, mode, sensorU, strength, period) {
    let fire = false;
    if (mode === 'feedback' && this.prev < 0.5 && sensorU >= 0.5) fire = true;
    if (mode === 'untimed' && T.t - this.lastUntimed >= period) { fire = true; this.lastUntimed = T.t; }
    this.prev = sensorU;
    if (fire && T.t >= this.until) { this.until = T.t + PULSE_TU; this.pulses++; }
    return T.t < this.until ? strength : 0;
  }
}
