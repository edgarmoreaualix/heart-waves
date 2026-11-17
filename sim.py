"""2D cardiac tissue, Aliev-Panfilov monodomain model (finite differences, explicit Euler).

u = transmembrane voltage (0 rest .. 1 excited, dimensionless), v = recovery variable.
  du/dt = D lap(u) - k u (u - a)(u - 1) - u v
  dv/dt = (eps0 + mu1 v / (u + mu2)) (-v - k u (u - a - 1))
Time unit: 1 t.u. ~ 12.9 ms; voltage: V_mV = 100 u - 80  (Aliev & Panfilov 1996).
No-flux boundaries. Pseudo-ECG at a virtual electrode above the tissue:
  phi(t) = sum grad(u) . grad(1/r) dA   (standard far-field approximation).
"""
from dataclasses import dataclass, field
import numpy as np

TU_MS = 12.9


@dataclass
class Params:
    n: int = 200            # grid points per side
    dx: float = 0.5         # space step (dimensionless)
    dt: float = 0.02        # time step (stable for dx^2/(4D) = 0.0625)
    D: float = 1.0
    k: float = 8.0
    a: float = 0.15
    eps0: float = 0.002
    mu1: float = 0.2
    mu2: float = 0.3
    t_end: float = 900.0    # t.u.
    save_every: float = 2.0 # t.u. between saved frames
    pace_period: float = 0  # S1 train from the left edge every pace_period t.u. (0 = single S1 at t=0)
    n_pace: int = 1
    s2_time: float | None = None   # cross-field S2 in the lower-left quadrant -> spiral
    electrode: tuple = (0.85, 0.5, 4.0)  # (x frac, y frac, height above tissue in grid units)
    fibrosis: float = 0.0   # fraction of randomly non-conducting cells (scar), 0 = healthy
    seed: int = 0


def lap(u, dx):
    p = np.pad(u, 1, mode="edge")  # no-flux
    return (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * u) / dx**2


def run(P: Params):
    rng = np.random.default_rng(P.seed)
    n = P.n
    u = np.zeros((n, n)); v = np.zeros((n, n))
    mask = np.ones((n, n))
    if P.fibrosis > 0:
        mask = (rng.random((n, n)) > P.fibrosis).astype(float)
    # electrode geometry for pseudo-ECG
    ys, xs = np.mgrid[0:n, 0:n].astype(float)
    ex, ey, ez = P.electrode[0] * n, P.electrode[1] * n, P.electrode[2]
    r = np.sqrt((xs - ex) ** 2 + (ys - ey) ** 2 + ez**2)
    g1y, g1x = np.gradient(1.0 / r)
    stims = [i * P.pace_period for i in range(P.n_pace)] if P.pace_period else [0.0]
    stim_dur = 1.0
    steps = int(P.t_end / P.dt); save_k = int(P.save_every / P.dt)
    frames, vframes, ts, ecg, ecg_t = [], [], [], [], []
    for s in range(steps + 1):
        t = s * P.dt
        if any(t0 <= t < t0 + stim_dur for t0 in stims):
            u[:, :5] = 1.0
        if P.s2_time is not None and P.s2_time <= t < P.s2_time + stim_dur:
            u[n // 2:, : n // 2] = 1.0  # lower-left quadrant (rows = y)
        if s % 10 == 0:
            gy, gx = np.gradient(u)
            ecg.append(float((gx * g1x + gy * g1y).sum())); ecg_t.append(t)
        if s % save_k == 0:
            frames.append(u.astype(np.float16)); vframes.append(v.astype(np.float16)); ts.append(t)
        du = P.D * lap(u, P.dx) * mask - P.k * u * (u - P.a) * (u - 1) - u * v
        dv = (P.eps0 + P.mu1 * v / (u + P.mu2)) * (-v - P.k * u * (u - P.a - 1))
        u = u + P.dt * du; v = v + P.dt * dv
    return dict(u=np.array(frames), v=np.array(vframes), t=np.array(ts), ecg=np.array(ecg), ecg_t=np.array(ecg_t),
                mask=mask.astype(np.uint8))


if __name__ == "__main__":
    import sys, time
    P = Params(s2_time=float(sys.argv[1]) if len(sys.argv) > 1 else None, t_end=float(sys.argv[2]) if len(sys.argv) > 2 else 300)
    t0 = time.time(); out = run(P); print(f"{time.time()-t0:.1f}s", out["u"].shape)
    np.savez_compressed("/tmp/claude-1001/heart_test.npz", **out)
