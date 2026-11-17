import * as C from './core.js';
const P = { ...C.DEFAULTS };
// 1) planar wave speed
let T = C.createTissue(160); C.paceLeft(T);
let tHalf = null, tEnd = null;
while (T.t < 120) { C.step(T, P, 25); const n = T.n, y = 80;
  if (tHalf == null && T.u[y*n+80] > 0.5) tHalf = T.t; if (tEnd == null && T.u[y*n+150] > 0.5) tEnd = T.t; }
console.log('CV grid/t.u.', (70/(tEnd-tHalf)).toFixed(3), 'space units/t.u.', (70*0.5/(tEnd-tHalf)).toFixed(3));
// 2) S2 vulnerable window scan
for (const d of [30, 35, 40, 45, 50, 55, 60, 65, 70]) {
  T = C.createTissue(160); C.paceLeft(T); let fired = false;
  const meter = new C.RateMeter(); let maxTips = 0;
  while (T.t < 600) { C.step(T, P, 25); if (!fired && T.t >= d) { C.s2Quadrant(T); fired = true; }
    meter.push(T.t, T.u[40*160+120]); if (T.t > 300) maxTips = Math.max(maxTips, C.tips(T).length); }
  console.log('S2 at', d, 'active frac', C.excitedFraction(T).toFixed(3), 'bpm', meter.bpm(T.t)?.toFixed(0), 'tips', maxTips);
}
// 3) defib
T = C.createTissue(160); C.paceLeft(T); let f=false;
while (T.t < 400) { C.step(T, P, 25); if (!f && T.t>=50){C.s2Quadrant(T);f=true;} }
C.shock(T); while (T.t < 700) C.step(T, P, 25);
console.log('after shock active', C.excitedFraction(T).toFixed(4));
const t0 = performance.now(); C.step(T, P, 500); console.log('ms/step', ((performance.now()-t0)/500).toFixed(3));
