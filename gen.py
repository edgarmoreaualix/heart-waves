# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "scipy", "h5py", "pyarrow", "pyyaml", "huggingface_hub"]
# ///
"""Generate a cardiac electrophysiology dataset (2D Aliev-Panfilov monodomain) in The Well's HDF5 format.

Each trajectory: 128x128 tissue, no-flux walls, heterogeneous conductivity (scar = 0), S1 pacing train from a random
site, optional premature S2 cross-field stimulus. Batched on GPU with torch (one sample per batch row).
Fields (t0): voltage (u, 0 rest .. 1 firing; V_mV = 100u - 80), recovery (v), conductivity (time-constant map).
Sidecar: meta/trajectories.parquet with parameters, labels (sustained re-entry, dominant rate) and 3 pseudo-ECG leads.

usage: uv run gen.py --n 2000 --out /tmp/heart [--repo Ethgar/cardiac-ep-2d] [--batch 250]
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import h5py

TU_MS = 12.9
N, DX, DT = 128, 0.78125, 0.05          # 100 x 100 dimensionless units
T_END, SAVE_EVERY = 480.0, 3.0           # -> 161 frames (~6.2 s of heart time)
K, EPS0, MU1, MU2 = 8.0, 0.002, 0.2, 0.3
ELECTRODES = [(0.92, 0.5), (0.5, 0.08), (0.08, 0.92)]  # (x, y) fractions, 4 px above tissue
STIM_DUR = 1.0


# ---------------------------------------------------------------- scenario sampling (numpy, per trajectory)
def smooth_noise(rng, corr):
    from scipy.ndimage import gaussian_filter
    z = gaussian_filter(rng.standard_normal((N, N)), corr, mode="reflect")
    return (z - z.mean()) / (z.std() + 1e-9)


def sample(rng, i):
    p = dict(idx=i, seed=int(rng.integers(2**31)))
    p["a"] = float(rng.uniform(0.10, 0.16))
    cond = np.ones((N, N), np.float32)
    kind = rng.choice(["healthy", "diffuse_fibrosis", "infarct_scar"], p=[0.4, 0.35, 0.25])
    p["tissue"] = str(kind)
    if kind == "diffuse_fibrosis":
        frac = float(rng.uniform(0.05, 0.35)); corr = float(rng.uniform(1.0, 6.0))
        z = smooth_noise(rng, corr); cond[z > np.quantile(z, 1 - frac)] = 0
        p.update(fibrosis_frac=frac, fibrosis_corr_px=corr)
    elif kind == "infarct_scar":
        cx, cy = rng.uniform(0.25, 0.75, 2) * N; rx, ry = rng.uniform(0.08, 0.22, 2) * N; th = rng.uniform(0, np.pi)
        yy, xx = np.mgrid[0:N, 0:N].astype(float)
        X = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th); Y = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
        r = (X / rx) ** 2 + (Y / ry) ** 2
        z = smooth_noise(rng, 2.0)
        cond[r < 1] = 0                                  # dense core
        border = (r >= 1) & (r < 1.8) & (z > 0.3)        # patchy border zone (isthmuses)
        cond[border] = 0
        p.update(scar_cx=cx, scar_cy=cy, scar_rx=rx, scar_ry=ry, scar_theta=th)
    p["scar_frac"] = float(1 - cond.mean())
    # S1 pacing
    site = str(rng.choice(["left", "right", "top", "bottom", "point"]))
    s1 = np.zeros((N, N), bool)
    if site == "left": s1[:, :4] = True
    elif site == "right": s1[:, -4:] = True
    elif site == "top": s1[:4, :] = True
    elif site == "bottom": s1[-4:, :] = True
    else:
        px, py = rng.uniform(0.1, 0.9, 2) * N; yy, xx = np.mgrid[0:N, 0:N]
        s1 = (xx - px) ** 2 + (yy - py) ** 2 < 16; p.update(pace_x=px, pace_y=py)
    s1 &= cond > 0
    period = float(rng.uniform(60, 110)); n_beats = int(min(rng.integers(1, 4), max(1, 220 // period)))
    p.update(pace_site=site, pace_period=period, n_beats=n_beats)
    # premature S2 (80 %): a random half-plane-ish rectangle, timed around the vulnerable window
    s2 = np.zeros((N, N), bool); p["s2_time"] = -1.0
    if rng.random() < 0.8:
        w, h = rng.uniform(0.3, 0.6, 2) * N; x0 = rng.uniform(0, N - w); y0 = rng.uniform(0, N - h)
        s2[int(y0):int(y0 + h), int(x0):int(x0 + w)] = True; s2 &= cond > 0
        p["s2_time"] = float((n_beats - 1) * period + rng.uniform(30, 100))
        p.update(s2_x0=x0, s2_y0=y0, s2_w=w, s2_h=h)
    return p, cond, s1, s2


# ---------------------------------------------------------------- batched solver (torch)
def lap_c(u, cx, cy):
    """Conductance-weighted 5-point Laplacian, no flux at walls and into scar (c=0)."""
    du_x = (u[:, :, 1:] - u[:, :, :-1]) * cx      # flux between columns j and j+1
    du_y = (u[:, 1:, :] - u[:, :-1, :]) * cy
    out = torch.zeros_like(u)
    out[:, :, :-1] += du_x; out[:, :, 1:] -= du_x
    out[:, :-1, :] += du_y; out[:, 1:, :] -= du_y
    return out / DX**2


def simulate(params, conds, s1s, s2s, dev):
    B = len(params)
    c = torch.tensor(np.stack(conds), device=dev)
    cx = torch.minimum(c[:, :, 1:], c[:, :, :-1]); cy = torch.minimum(c[:, 1:, :], c[:, :-1, :])
    a = torch.tensor([p["a"] for p in params], device=dev).view(B, 1, 1)
    S1 = torch.tensor(np.stack(s1s), device=dev); S2 = torch.tensor(np.stack(s2s), device=dev)
    s1_t = torch.full((B, 3), 1e9, device=dev)
    for b, p in enumerate(params):
        for k in range(p["n_beats"]):
            s1_t[b, k] = k * p["pace_period"]
    s2_t = torch.tensor([p["s2_time"] if p["s2_time"] >= 0 else 1e9 for p in params], device=dev)
    alive = (c > 0).float()
    u = torch.zeros(B, N, N, device=dev); v = torch.zeros_like(u)
    yy, xx = torch.meshgrid(torch.arange(N, device=dev).float(), torch.arange(N, device=dev).float(), indexing="ij")
    gr = []
    for ex, ey in ELECTRODES:
        r = torch.sqrt((xx - ex * N) ** 2 + (yy - ey * N) ** 2 + 16.0)
        gy, gx = torch.gradient(1.0 / r); gr.append((gx, gy))
    steps = int(round(T_END / DT)); save_k = int(round(SAVE_EVERY / DT)); ecg_k = int(round(0.5 / DT))
    U, V, ECG = [], [], []
    for s in range(steps + 1):
        t = s * DT
        on1 = ((t >= s1_t) & (t < s1_t + STIM_DUR)).any(1).view(B, 1, 1)
        on2 = ((t >= s2_t) & (t < s2_t + STIM_DUR)).view(B, 1, 1)
        u = torch.where((S1 & on1) | (S2 & on2), torch.ones_like(u), u)
        if s % ecg_k == 0:
            gy, gx = torch.gradient(u, dim=(1, 2))
            ECG.append(torch.stack([(gx * g[0] + gy * g[1]).sum((1, 2)) for g in gr], 1).cpu())
        if s % save_k == 0:
            U.append(u.cpu().half()); V.append(v.cpu().half())
        du = lap_c(u, cx, cy) - K * u * (u - a) * (u - 1) - u * v
        dv = (EPS0 + MU1 * v / (u + MU2)) * (-v - K * u * (u - a - 1))
        u = (u + DT * du) * alive; v = (v + DT * dv) * alive
    return (torch.stack(U, 1).numpy(), torch.stack(V, 1).numpy(), torch.stack(ECG, 1).numpy())  # fp16 frames: 4x less RAM


# ---------------------------------------------------------------- labels + The Well HDF5 writer
def label(p, U, ecg):
    t = np.arange(U.shape[0]) * SAVE_EVERY
    last_stim = max((p["n_beats"] - 1) * p["pace_period"], p["s2_time"])
    act = (U > 0.5).mean((1, 2))
    late = act[t > last_stim + 150]
    sustained = bool(len(late) and late.min() > 0.002)
    rate_bpm = float("nan")
    e = ecg[int((last_stim + 100) / 0.5):, 0]
    if sustained and len(e) > 64:
        e = e - e.mean(); f = np.fft.rfftfreq(len(e), 0.5); pw = np.abs(np.fft.rfft(e)); fr = f[1:][pw[1:].argmax()]
        rate_bpm = float(60000 / (TU_MS / fr))
    return dict(sustained_reentry=sustained, reentry_rate_bpm=rate_bpm, final_active_frac=float(act[-1]))


SIM_PARAMS = ["a", "pace_period", "n_beats", "s2_time", "scar_frac"]


def write_well(path, p, U, V, cond):
    with h5py.File(path, "w") as h:
        h.attrs.update(dataset_name="cardiac_electrophysiology_2D", grid_type="cartesian", n_spatial_dims=2,
                       n_trajectories=1)
        h.attrs["simulation_parameters"] = SIM_PARAMS
        for k in SIM_PARAMS:
            h.attrs[k] = float(p[k])
        d = h.create_group("dimensions"); d.attrs["spatial_dims"] = ["x", "y"]
        tt = d.create_dataset("time", data=(np.arange(U.shape[0]) * SAVE_EVERY).astype(np.float32))
        tt.attrs.update(sample_varying=False, time_varying=True)
        for nm in ("x", "y"):
            ds = d.create_dataset(nm, data=((np.arange(N) + 0.5) * DX).astype(np.float32))
            ds.attrs.update(sample_varying=False, time_varying=False)
        bc = h.create_group("boundary_conditions")
        for nm in ("x", "y"):
            g = bc.create_group(f"{nm}_wall")
            g.attrs.update(associated_dims=[nm], associated_fields=[], bc_type="WALL", sample_varying=False, time_varying=False)
            m = np.zeros(N, bool); m[[0, -1]] = True; g.create_dataset("mask", data=m)
        sc = h.create_group("scalars"); sc.attrs["field_names"] = SIM_PARAMS
        for k in SIM_PARAMS:
            ds = sc.create_dataset(k, data=np.float32(p[k])); ds.attrs.update(sample_varying=False, time_varying=False)
        f0 = h.create_group("t0_fields"); f0.attrs["field_names"] = ["voltage", "recovery", "conductivity"]
        kw = dict(compression="gzip", compression_opts=1, chunks=(1, 1, N, N))
        # arrays are (traj, time, x, y): transpose our (time, y, x) images
        for nm, arr in (("voltage", U), ("recovery", V)):
            ds = f0.create_dataset(nm, data=arr.transpose(0, 2, 1)[None].astype(np.float32), **kw)
            ds.attrs.update(dim_varying=[True, True], sample_varying=True, time_varying=True)
        ds = f0.create_dataset("conductivity", data=cond.T[None].astype(np.float32), compression="gzip")
        ds.attrs.update(dim_varying=[True, True], sample_varying=True, time_varying=False)
        for g in ("t1_fields", "t2_fields"):
            h.create_group(g).attrs["field_names"] = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--out", default="/tmp/heart"); ap.add_argument("--repo", default=None)
    ap.add_argument("--seed", type=int, default=2026)
    A = ap.parse_args()
    api = None
    if A.repo:
        from huggingface_hub import HfApi
        api = HfApi(); api.create_repo(A.repo, repo_type="dataset", private=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(A.out); rng = np.random.default_rng(A.seed)
    rows, stats = [], {k: [0.0, 0.0, 0] for k in ("voltage", "recovery", "conductivity")}
    t0 = time.time()
    for b0 in range(0, A.n, A.batch):
        batch = [sample(rng, i) for i in range(b0, min(A.n, b0 + A.batch))]
        P = [x[0] for x in batch]
        U, V, E = simulate(P, [x[1] for x in batch], [x[2] for x in batch], [x[3] for x in batch], dev)
        for j, (p, cond, _, _) in enumerate(batch):
            i = p["idx"]; split = "train" if i % 10 < 8 else ("valid" if i % 10 == 8 else "test")
            (out / "data" / split).mkdir(parents=True, exist_ok=True)
            p.update(label(p, U[j], E[j]), split=split, file=f"data/{split}/cardiac_ep_{i:05d}.hdf5")
            write_well(out / p["file"], p, U[j].astype(np.float32), V[j].astype(np.float32), cond)
            if split == "train":
                for nm, arr in (("voltage", U[j]), ("recovery", V[j]), ("conductivity", cond)):
                    stats[nm][0] += float(arr.astype(np.float64).sum()); stats[nm][1] += float((arr.astype(np.float64) ** 2).sum()); stats[nm][2] += arr.size
            rows.append({**p, **{f"ecg_lead{k}": E[j][:, k].astype(np.float32).tolist() for k in range(3)}})
        del U, V
        if api:  # upload this batch, then free the (RAM-backed) disk
            import shutil
            for _ in range(5):
                try:
                    api.upload_folder(repo_id=A.repo, repo_type="dataset", folder_path=str(out / "data"), path_in_repo="data",
                                      commit_message=f"trajectories {b0}-{b0 + len(batch) - 1}"); break
                except Exception as e:
                    print("upload retry", repr(e)[:200], flush=True); time.sleep(30)
            shutil.rmtree(out / "data")
        print(f"[{len(rows)}/{A.n}] {time.time()-t0:.0f}s  sustained so far: {sum(r['sustained_reentry'] for r in rows)}", flush=True)
    import pyarrow as pa, pyarrow.parquet as pq, yaml
    (out / "meta").mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out / "meta" / "trajectories.parquet")
    st = {"mean": {}, "std": {}, "rms": {}}
    for nm, (s, s2, n) in stats.items():
        m = s / n; st["mean"][nm] = m; st["rms"][nm] = math.sqrt(s2 / n); st["std"][nm] = math.sqrt(max(s2 / n - m * m, 0))
    yaml.safe_dump(st, open(out / "stats.yaml", "w"))
    summ = dict(n=len(rows), sustained=int(sum(r["sustained_reentry"] for r in rows)), wall_s=time.time() - t0, device=dev,
                by_tissue={k: int(sum(r["tissue"] == k for r in rows)) for k in ("healthy", "diffuse_fibrosis", "infarct_scar")})
    json.dump(summ, open(out / "meta" / "summary.json", "w"), indent=1); print(summ)
    if api:
        api.upload_folder(repo_id=A.repo, repo_type="dataset", folder_path=str(out), commit_message="meta + stats")
        print("uploaded", A.repo, flush=True)


if __name__ == "__main__":
    main()
