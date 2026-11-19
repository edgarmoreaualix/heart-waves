---
license: cc-by-4.0
task_categories:
- time-series-forecasting
tags:
- physics
- cardiac-electrophysiology
- arrhythmia
- reaction-diffusion
- the-well
- pde
pretty_name: Cardiac Electrophysiology 2D (spiral waves & re-entry)
size_categories:
- 1K<n<10K
---

# Cardiac Electrophysiology 2D

▶ Play with the physics in your browser: [edgarmoreau/heart-waves](https://huggingface.co/spaces/edgarmoreau/heart-waves) ·
🧠 Walrus fine-tuned on it: [edgarmoreau/walrus-heart](https://huggingface.co/edgarmoreau/walrus-heart) ·
💻 Code: [github.com/edgarmoreaualix/heart-waves](https://github.com/edgarmoreaualix/heart-waves)

2,000 simulations of electrical waves in a 2D sheet of heart tissue, including **normal paced beats**, **premature beats**,
and the **spiral-wave re-entry** that drives ventricular tachycardia and fibrillation, on healthy, fibrotic, and post-infarct (scarred) tissue.
Every trajectory comes with three virtual ECG leads and an arrhythmia label.

Stored in [The Well](https://github.com/PolymathicAI/the_well) HDF5 format, so it plugs directly into physics foundation models such as
[polymathic-ai/walrus](https://huggingface.co/polymathic-ai/walrus) for fine-tuning.

## Physics
Aliev–Panfilov monodomain model (Aliev & Panfilov, 1996), a standard phenomenological model of ventricular myocytes:

$$\partial_t u = \nabla\cdot(\sigma\nabla u) - k\,u(u-a)(u-1) - u v,\qquad
\partial_t v = \Big(\varepsilon_0 + \frac{\mu_1 v}{u+\mu_2}\Big)\big(-v - k\,u(u-a-1)\big)$$

- `u`: transmembrane voltage, dimensionless (V ≈ 100·u − 80 mV); `v`: recovery variable
- `k = 8, ε0 = 0.002, μ1 = 0.2, μ2 = 0.3`; excitability `a ~ U(0.10, 0.16)` per trajectory
- 128 × 128 grid, Δx = 0.78 (100 × 100 dimensionless units), explicit Euler Δt = 0.05, no-flux walls
- Conductance-weighted Laplacian: no current flows into scar (σ = 0)
- Time unit ≈ 12.9 ms; 161 frames every 3 t.u. (≈ 39 ms), ≈ 6.2 s of heart time

## Scenarios (sampled per trajectory)
| Factor | Distribution |
|---|---|
| Tissue | healthy 40 % · diffuse fibrosis 35 % (5–35 % non-conducting, patch size 1–6 px) · infarct scar 25 % (elliptic core + patchy border zone) |
| Pacing site | left / right / top / bottom edge, or a point source |
| Pacing | 1–3 beats, period 60–110 t.u. (≈ 42–78 bpm) |
| Premature S2 | 80 % of runs: rectangle covering 9–36 % of tissue, 30–100 t.u. after the last beat (spans the vulnerable window) |

## Files
```
data/{train,valid,test}/cardiac_ep_XXXXX.hdf5   # The Well format, 1 trajectory per file (80/10/10 split)
  t0_fields/voltage       (1, 161, 128, 128) float32
  t0_fields/recovery      (1, 161, 128, 128) float32
  t0_fields/conductivity  (1, 128, 128)      float32   time-constant (0 = scar)
meta/trajectories.parquet  # all parameters, labels, 3 pseudo-ECG leads (every 0.5 t.u.)
meta/summary.json
stats.yaml                 # normalisation stats (train split)
```
Labels: `sustained_reentry` (activity persists ≥150 t.u. after the last stimulus), `reentry_rate_bpm` (dominant ECG frequency).

## Limitations
2D isotropic sheet, phenomenological cell model, no fibre orientation, no torso. ECGs are far-field pseudo-ECGs over the tissue,
not 12-lead body-surface ECGs. A research and teaching dataset — not for clinical use.

## Generation
Fully reproducible with [`gen.py`](https://github.com/edgarmoreaualix/heart-waves/blob/main/gen.py): `uv run gen.py --n 2000 --seed 2026` (PyTorch, batched on one GPU, ~15 min on an A10G).
