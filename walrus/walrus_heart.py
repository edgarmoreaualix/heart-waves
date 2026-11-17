# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "walrus @ git+https://github.com/PolymathicAI/walrus@73f2cf6b4c5871c8f5434802725b55666c66b5ab",
#   "huggingface_hub>=0.34", "hf_xet", "safetensors", "pyarrow==17.0.0", "pyyaml", "h5py",
# ]
# ///
"""Walrus-Heart: fine-tune Walrus (1.3B, polymathic-ai/walrus) vs the same architecture from scratch on the
cardiac EP dataset Ethgar/cardiac-ep-2d (2D Aliev-Panfilov, The Well format).

    hf jobs uv run --flavor a100-large --secrets HF_TOKEN --timeout 3h -e MODE=train -e INIT=pretrained ... walrus_heart.py

Bypasses walrus' Hydra trainer (slurm/wandb-centric) but reuses its exact model, checkpoint alignment, formatter
and reversible normalisation (Walrus notebook 1 "Part 2: non-Well data"), replicating Trainer.rollout_model/train_one_epoch:
  * fields: voltage, recovery (state, predicted) + conductivity (constant, appended every step); 3 new field embeddings
    (indices 67,68,69) added via walrus.utils.experiment_utils.align_checkpoint_with_field_to_index_map
  * 2D data padded to 3D with a singleton z (BCs: x,y WALL, z PERIODIC — as InflatedWellDataset does)
  * RMS sample-wise rev-norm, delta prediction, causal-in-time: T_in=6 history gives 6 predictions, first dropped
    (minimum_context=1), loss = 100 x MAE on delta-normalised targets, AdamW, grad clip 10, learnable RoPE (finetuning_mods=all)
  * windows whose target frames contain an S1/S2 stimulus are excluded (stimuli are exogenous, unpredictable)
env: MODE=smoke|train|local  INIT=pretrained|scratch  NTRAIN  STEPS  LR  BATCH  RUN
"""
import os, sys, time, json, math, copy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import h5py

t_start = time.time()
log = lambda *a: print(f"[{time.time()-t_start:7.1f}s]", *a, flush=True)
E = os.environ.get
MODE = E("MODE", "train"); INIT = E("INIT", "pretrained"); NTRAIN = int(E("NTRAIN", "1600"))
STEPS = int(E("STEPS", "1500")); LR = float(E("LR", "1e-4")); BATCH = int(E("BATCH", "4")); TIN = int(E("TIN", "6"))
WARM = int(E("WARMUP", "100")); VAL_EVERY = int(E("VAL_EVERY", "250")); GC = int(E("GC_FREQ", "2"))
N_ROLL = int(E("N_ROLL", "48")); ROLL_B = int(E("ROLL_B", "16")); MAX_MIN = float(E("MAX_TRAIN_MIN", "150"))
RUN = E("RUN", f"{INIT}_n{NTRAIN}")
VIDEO_IDS = [int(x) for x in E("VIDEO_IDS", "19,679,1209,29,449,1129,279,1949,1599,509,139,1309").split(",")]
DATA, MODEL_REPO, RUNS_REPO = "Ethgar/cardiac-ep-2d", "Ethgar/walrus-heart", "Ethgar/frontier-runs"
LOCAL = MODE == "local"
EPS = 1e-5; STIM_DUR = 1.0; DT_FRAME = 3.0; NT = 161
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
AMP = dev.type == "cuda"
OUT = Path(E("OUT", "/tmp/walrus_heart_out")) / RUN; OUT.mkdir(parents=True, exist_ok=True)
log("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", "RUN", RUN, "MODE", MODE)

from einops import rearrange
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate
from the_well.data.datasets import WellMetadata
from walrus.utils.experiment_utils import align_checkpoint_with_field_to_index_map
from walrus.trainer.normalization_strat import RMSSamplewiseRevNormalization
from huggingface_hub import hf_hub_download, snapshot_download, HfApi


# ------------------------------------------------------------------ data
def load_one(path):
    with h5py.File(path) as h:
        u = h["t0_fields/voltage"][0]; v = h["t0_fields/recovery"][0]; c = h["t0_fields/conductivity"][0]
    return np.stack([u, v], -1).astype(np.float16), c.astype(np.float16)   # (161,128,128,2) in file (x,y) order


def stim_times(r):
    ts = [k * r["pace_period"] for k in range(int(r["n_beats"]))]
    if r["s2_time"] >= 0: ts.append(r["s2_time"])
    return ts


def bad_frames(r):
    """bad[j]: the transition j-1 -> j contains (part of) a stimulus [ts, ts+1)."""
    j = np.arange(NT); bad = np.zeros(NT, bool)
    for ts in stim_times(r):
        bad |= (ts <= DT_FRAME * j) & (ts + STIM_DUR > DT_FRAME * (j - 1))
    return bad


def rollout_start(r):
    """Index of the last history frame: first frame fully after the last stimulus."""
    last = max(stim_times(r))
    j = int(math.ceil((last + STIM_DUR) / DT_FRAME))
    return max(j, TIN - 1)


class Split:
    def __init__(self, rows, root):
        self.rows = rows.reset_index(drop=True); n = len(rows)
        self.X = np.empty((n, NT, 128, 128, 2), np.float16); self.C = np.empty((n, 128, 128), np.float16)
        if root is None:   # local synthetic
            rng = np.random.default_rng(0); self.X[:] = rng.random(self.X.shape) * 0.1; self.C[:] = 1
        else:
            paths = [f"{root}/{f}" for f in self.rows.file]
            with ProcessPoolExecutor(min(16, os.cpu_count() or 4)) as ex:
                for i, (x, c) in enumerate(ex.map(load_one, paths, chunksize=4)):
                    self.X[i] = x; self.C[i] = c
        self.bad = np.stack([bad_frames(r) for _, r in self.rows.iterrows()])
        # window start s: inputs s..s+TIN-1, causal targets s+1..s+TIN; valid if none of those transitions is stimulated
        self.windows = [(i, s) for i in range(n) for s in range(NT - TIN) if not self.bad[i, s + 1:s + TIN + 1].any()]

    def batch(self, idx):
        xs = torch.from_numpy(np.stack([self.X[i, s:s + TIN + 1] for i, s in idx])).to(dev).float()   # B T+1 H W 2
        cs = torch.from_numpy(np.stack([self.C[i] for i, _ in idx])).to(dev).float()
        return xs, cs


# ------------------------------------------------------------------ model
cfg = OmegaConf.load(hf_hub_download("polymathic-ai/walrus", "extended_config.yaml")) if not LOCAL else None
if LOCAL:
    cfg = OmegaConf.load(E("LOCAL_CFG"))
fmap = {k: int(v) for k, v in cfg.data.field_index_map_override.items()}
nmap = dict(fmap)
for f in ("voltage", "recovery", "conductivity"):
    nmap[f] = max(nmap.values()) + 1
with open_dict(cfg):
    cfg.model.drop_path = 0.0; cfg.model.gradient_checkpointing_freq = GC; cfg.model.input_field_drop = 0.0
    if LOCAL:
        cfg.model.hidden_dim = 256; cfg.model.processor_blocks = 2
        cfg.model.processor.space_mixing.num_heads = 4; cfg.model.processor.time_mixing.num_heads = 4
model = instantiate(cfg.model, n_states=max(nmap.values()) + 1)
if INIT == "pretrained" and not LOCAL:
    ck = torch.load(hf_hub_download("polymathic-ai/walrus", "walrus.pt"), map_location="cpu", weights_only=True)["app"]["model"]
    ck = align_checkpoint_with_field_to_index_map(ck, model.state_dict(), fmap, nmap)
    log("load_state_dict:", model.load_state_dict(ck)); del ck
model.add_ft_options({"learnable_rope": True, "rope_per_axis": True})   # walrus finetuning_mods=all
model.to(dev)
# PERF FIX: under no_grad, nn.Linear on the permuted (non-contiguous) 5D activations dispatches to a slow bmm kernel
# (2.6 s vs ~0.4 s per forward on A100, torch 2.5.1); make inputs contiguous first.
for _m in model.modules():
    if isinstance(_m, torch.nn.Linear):
        _m.register_forward_pre_hook(lambda m, a: (a[0].contiguous(),) + tuple(a[1:]))
n_params = sum(p.numel() for p in model.parameters())
log(f"model {n_params/1e9:.3f}B params, init={INIT}, causal={model.causal_in_time}")

revin = RMSSamplewiseRevNormalization()
FIDX = torch.tensor([nmap["voltage"], nmap["recovery"], nmap["conductivity"]], device=dev)
BCS = [[0, 0], [0, 0], [2, 2]]   # WALL, WALL, PERIODIC(padded z)
META = WellMetadata(dataset_name="cardiac_electrophysiology_2D", n_spatial_dims=3, spatial_resolution=(128, 128, 1),
                    scalar_names=[], constant_scalar_names=[], field_names={0: ["voltage", "recovery"], 1: [], 2: []},
                    constant_field_names={0: ["conductivity"], 1: [], 2: []},
                    boundary_condition_types=["WALL", "WALL", "PERIODIC"], n_files=1, n_trajectories_per_file=[1],
                    n_steps_per_trajectory=[NT])


def forward(hist, cond):
    """hist B T H W 2, cond B H W -> (raw inputs T B C H W 1, normalised delta preds T B C H W 1, stats)."""
    B, T = hist.shape[:2]
    x = rearrange(hist, "b t h w c -> t b c h w").unsqueeze(-1)
    c = cond[None, :, None, :, :, None].expand(T, B, 1, *cond.shape[1:], 1)
    x = torch.cat([x, c], 2)
    with torch.no_grad():
        st = revin.compute_stats(x, META, epsilon=EPS)
    xn = revin.normalize_stdmean(x, st)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=AMP):
        y = model(xn, FIDX, [BCS] * B, metadata=META)
    return x, y.float(), st


def train_loss(xs, cs):
    """Replicates Trainer.rollout_model(train=True) + MAE loss, causal, delta, minimum_context=1."""
    hist = xs[:, :TIN]
    x, y, st = forward(hist, cs)
    tgt = rearrange(xs[:, 1:] - xs[:, :-1], "b t h w c -> t b c h w").unsqueeze(-1)   # deltas for frames 1..TIN
    tgt = (tgt - st.delta_mean[:, :, :2]) / st.delta_std[:, :, :2]
    err = (y[:, :, :2] - tgt).abs()[1:]                     # drop zero-context prediction
    return 100.0 * err.mean()


@torch.no_grad()
def predict_next(hist, cond):
    x, y, st = forward(hist, cond)
    d = revin.denormalize_delta(y[-1:], st)[0, :, :2, ..., 0]            # B 2 H W
    return hist[:, -1] + rearrange(d, "b c h w -> b h w c")


def vrmse(p, g, eps=1e-7):   # per sample, per channel; p,g: B H W C
    return (((p - g) ** 2).mean((1, 2)) / (g.var((1, 2), unbiased=False) + eps)).sqrt()


# ------------------------------------------------------------------ eval
@torch.no_grad()
def eval_one_step(split, windows, bs=16):
    model.eval(); out = []
    for k in range(0, len(windows), bs):
        xs, cs = split.batch(windows[k:k + bs])
        p = predict_next(xs[:, :TIN], cs); g = xs[:, TIN]
        out.append(torch.cat([vrmse(p, g), ((p - g) ** 2).mean((1, 2)).sqrt(), g.var((1, 2), unbiased=False)], 1).cpu())
    o = torch.cat(out).numpy()   # cols: vrmse_u, vrmse_v, rmse_u, rmse_v, var_u, var_v
    ok_u = o[:, 4] > 1e-6; ok_v = o[:, 5] > 1e-6
    return dict(vrmse_voltage=float(o[ok_u, 0].mean()), vrmse_recovery=float(o[ok_v, 1].mean()),
                vrmse_mean=float((o[ok_u, 0].mean() + o[ok_v, 1].mean()) / 2),
                vrmse_voltage_median=float(np.median(o[ok_u, 0])),
                rmse_voltage=float(o[:, 2].mean()), rmse_recovery=float(o[:, 3].mean()),
                n_windows=len(o), n_windows_nonconstant_voltage=int(ok_u.sum()))


@torch.no_grad()
def eval_rollouts(split, n_max):
    """Start each test rollout right after the last stimulus (S2 or last S1) and roll out autoregressively to frame 160."""
    model.eval(); rows = split.rows
    # subset: all video ids, then fill to n_max balanced between sustained re-entry / termination (by idx order)
    pick = [i for i in range(len(rows)) if int(rows.idx[i]) in VIDEO_IDS]
    for want in (True, False):
        for i in range(len(rows)):
            if sum(bool(rows.sustained_reentry[j]) == want for j in pick) >= n_max // 2: break
            if i not in pick and bool(rows.sustained_reentry[i]) == want: pick.append(i)
    order = sorted(pick[:max(n_max, 1)] if len(rows) > n_max else range(len(rows)), key=lambda i: rollout_start(rows.iloc[i]))
    res = {}; vids = {}
    for k in range(0, len(order), ROLL_B):
        ids = order[k:k + ROLL_B]; s0 = [rollout_start(rows.iloc[i]) for i in ids]
        hist = torch.stack([torch.from_numpy(split.X[i, s - TIN + 1:s + 1]) for i, s in zip(ids, s0)]).to(dev).float()
        cond = torch.from_numpy(split.C[ids]).to(dev).float()
        n = NT - 1 - min(s0)
        preds = []
        for _ in range(n):
            nxt = predict_next(hist, cond); preds.append(nxt[..., 0].half().cpu())   # keep voltage only
            hist = torch.cat([hist[:, 1:], nxt[:, None]], 1)
        P = torch.stack(preds, 1).float()                       # B n H W (voltage)
        for b, (i, s) in enumerate(zip(ids, s0)):
            L = NT - 1 - s; pu = P[b, :L].numpy(); gu = split.X[i, s + 1:, :, :, 0].astype(np.float32)
            err = np.sqrt(((pu - gu) ** 2).mean((1, 2))); var = gu.var((1, 2))
            vr = np.sqrt(((pu - gu) ** 2).mean((1, 2)) / (var + 1e-7))
            af_p = (pu > 0.5).mean((1, 2)); af_g = (gu > 0.5).mean((1, 2))
            r = rows.iloc[i]; last = max(stim_times(r)); tt = DT_FRAME * np.arange(s + 1, NT)
            late = tt > last + 150
            res[int(r["idx"])] = dict(start=s, rmse_u=err.tolist(), vrmse_u=vr.tolist(), af_pred=af_p.tolist(),
                                      af_true=af_g.tolist(), label=bool(r["sustained_reentry"]), tissue=r["tissue"],
                                      pred_final=bool(af_p[-1] > 0.002), true_final=bool(af_g[-1] > 0.002),
                                      pred_late=bool(late.any() and af_p[late].min() > 0.002))
            if int(r["idx"]) in VIDEO_IDS:
                vids[int(r["idx"])] = (s, pu.astype(np.float16))
        log(f"  rollout batch {k//ROLL_B+1}/{math.ceil(len(order)/ROLL_B)} start {min(s0)}..{max(s0)} n={n}")
    # summaries
    lab = np.array([v["label"] for v in res.values()]); pf = np.array([v["pred_final"] for v in res.values()])
    tf = np.array([v["true_final"] for v in res.values()]); pl = np.array([v["pred_late"] for v in res.values()])
    H = 100; acc = np.full(H, np.nan); cnt = np.zeros(H)
    sums = np.zeros(H)
    for v in res.values():
        e = np.array(v["rmse_u"][:H]); sums[:len(e)] += e; cnt[:len(e)] += 1
    m = cnt > 0; acc[m] = sums[m] / cnt[m]
    tp = int((pf & lab).sum()); tn = int((~pf & ~lab).sum()); fp = int((pf & ~lab).sum()); fn = int((~pf & lab).sum())
    summ = dict(n=len(res), reentry_acc_final=float((pf == lab).mean()), reentry_acc_late=float((pl == lab).mean()),
                gt_final_vs_label_acc=float((tf == lab).mean()), confusion_final=dict(tp=tp, tn=tn, fp=fp, fn=fn),
                rollout_rmse_u_vs_horizon=[None if np.isnan(a) else float(a) for a in acc],
                rollout_rmse_u_mean_all=float(np.mean([np.mean(v["rmse_u"]) for v in res.values()])),
                rollout_rmse_u_at={str(h): float(acc[h - 1]) for h in (1, 5, 10, 20, 40, 60, 80, 100) if h <= H and not np.isnan(acc[h - 1])})
    return summ, res, vids


# ------------------------------------------------------------------ main
def profile():
    H = 128
    from torch.profiler import profile as tprof, ProfilerActivity
    hist = torch.rand(1, TIN, H, H, 2, device=dev); cond = torch.ones(1, H, H, device=dev)
    for nm, ctx in (("nograd", torch.no_grad), ("grad", torch.enable_grad)):
        with ctx():
            forward(hist, cond); torch.cuda.synchronize()
            with tprof(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as pr:
                forward(hist, cond); torch.cuda.synchronize()
        print("=====", nm); print(pr.key_averages().table(sort_by="cuda_time_total", row_limit=18), flush=True)
    if E("PROF_ONLY"): return
    for B in (1, 2, 4, 8, 16):
        hist = torch.rand(B, TIN, H, H, 2, device=dev); cond = torch.ones(B, H, H, device=dev)
        for mode in ("eval_nograd", "train_nograd", "eval_inference", "train_grad"):
            if mode == "train_grad" and B > 4: continue
            model.train(mode.startswith("train"))
            ctx = torch.inference_mode() if "inference" in mode else (torch.no_grad() if "nograd" in mode else torch.enable_grad())
            ts = []
            for r in range(4):
                torch.cuda.synchronize(); t0 = time.time()
                with ctx:
                    _, y, _ = forward(hist, cond)
                    if mode == "train_grad": y.float().mean().backward()
                torch.cuda.synchronize(); ts.append(time.time() - t0)
            model.zero_grad(set_to_none=True)
            log(f"B={B:2d} {mode:15s} {np.median(ts[1:]):.3f}s  per-sample {np.median(ts[1:])/B:.3f}s  mem {torch.cuda.max_memory_allocated()/2**30:.1f}GB")


def main():
    import pandas as pd
    if MODE == "prof":
        return profile()
    if LOCAL:
        rows = pd.DataFrame([dict(idx=i, file="", pace_period=80.0, n_beats=1, s2_time=50.0, sustained_reentry=i % 2 == 0,
                                   tissue="healthy", split="x") for i in range(4)])
        tr = va = te = Split(rows, None); te.rows["idx"] = [19, 1, 2, 3]
    else:
        meta = pd.read_parquet(hf_hub_download(DATA, "meta/trajectories.parquet", repo_type="dataset"),
                               columns=["idx", "file", "split", "pace_period", "n_beats", "s2_time", "sustained_reentry", "tissue"])
        meta = meta.sort_values("idx")
        ntr = NTRAIN if MODE == "train" else 8; nva = 200 if MODE == "train" else 4; nte = 200 if MODE == "train" else 16
        trm = meta[meta.split == "train"].head(ntr); vam = meta[meta.split == "valid"].head(nva); tem = meta[meta.split == "test"].head(nte)
        files = list(trm.file) + list(vam.file) + list(tem.file)
        root = snapshot_download(DATA, repo_type="dataset", allow_patterns=files, max_workers=16)
        log(f"downloaded {len(files)} files"); tr = Split(trm, root); va = Split(vam, root); te = Split(tem, root)
        log(f"loaded train {len(tr.rows)} ({len(tr.windows)} windows) valid {len(va.rows)} test {len(te.rows)}; RAM-resident fp16")
    rng = np.random.default_rng(0)
    val_w = [va.windows[i] for i in rng.choice(len(va.windows), min(64, len(va.windows)), replace=False)]
    test_w = []
    for i in range(len(te.rows)):
        ws = [w for w in te.windows if w[0] == i]
        test_w += [ws[j] for j in np.linspace(0, len(ws) - 1, 4).round().astype(int)]

    steps = STEPS if MODE == "train" else (6 if LOCAL else 30)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4, eps=1e-10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARM) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    hist = dict(step=[], loss=[], val_step=[], val=[]); best = (float("inf"), None, -1)
    if MODE == "train":
        v0 = eval_one_step(va, val_w); log("valid @0", v0); hist["val_step"].append(0); hist["val"].append(v0)
    model.train(); t_tr = time.time(); run_loss = []
    for step in range(1, steps + 1):
        idx = [tr.windows[j] for j in rng.integers(0, len(tr.windows), BATCH)]
        xs, cs = tr.batch(idx)
        loss = train_loss(xs, cs)
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step(); sched.step(); run_loss.append(loss.item())
        if step % 25 == 0 or step <= 5:
            mem = torch.cuda.max_memory_allocated() / 2**30 if AMP else 0
            el = time.time() - t_tr
            log(f"step {step}/{steps} loss {np.mean(run_loss[-25:]):.3f} gn {gn:.2f} lr {sched.get_last_lr()[0]:.2e} "
                f"{el/step:.2f}s/step mem {mem:.1f}GB")
            hist["step"].append(step); hist["loss"].append(float(np.mean(run_loss[-25:])))
        if MODE == "train" and (step % VAL_EVERY == 0 or step == steps):
            v = eval_one_step(va, val_w); model.train()
            hist["val_step"].append(step); hist["val"].append(v); log(f"valid @{step}", v)
            if v["vrmse_mean"] < best[0]:
                best = (v["vrmse_mean"], {k: t.detach().to("cpu", copy=True) for k, t in model.state_dict().items()}, step)
        if (time.time() - t_tr) / 60 > MAX_MIN:
            log(f"time cap {MAX_MIN} min reached at step {step}"); steps = step; break
    train_s = time.time() - t_tr
    if best[1] is not None:
        model.load_state_dict(best[1]); log(f"restored best valid checkpoint @ step {best[2]} (vrmse_mean {best[0]:.4f})")

    log("test one-step ..."); t_ev = time.time()
    one = eval_one_step(te, test_w); log("test one-step", one)
    log("test rollouts ...")
    summ, per, vids = eval_rollouts(te, N_ROLL if MODE == "train" else 6)
    eval_s = time.time() - t_ev
    log("rollout summary", {k: v for k, v in summ.items() if k != "rollout_rmse_u_vs_horizon"})
    metrics = dict(run=RUN, init=INIT, n_train_traj=int(len(tr.rows)), steps=steps, best_step=best[2], batch=BATCH, lr=LR,
                   warmup=WARM, schedule="linear warmup + cosine to 1% (per step)", t_in=TIN, optimizer="AdamW wd1e-4 eps1e-10",
                   clip=10.0, loss="100*MAE on RMS-delta-normalised targets (causal, min context 1)", amp="bf16 autocast",
                   params=n_params, train_seconds=train_s, eval_seconds=eval_s, total_seconds=time.time() - t_start,
                   gpu=torch.cuda.get_device_name(0) if AMP else "cpu", test_one_step=one, test_rollout=summ, history=hist)
    json.dump(metrics, open(OUT / "metrics.json", "w"), indent=1)
    json.dump(per, open(OUT / "rollouts_per_traj.json", "w"))
    if vids:
        np.savez_compressed(OUT / "rollouts_video.npz", ids=np.array(list(vids)), starts=np.array([v[0] for v in vids.values()]),
                            **{f"pred_{k}": v[1] for k, v in vids.items()})
    log("saved", list(OUT.iterdir()))
    if MODE == "local":
        return
    api = HfApi()
    if MODE == "train":
        from safetensors.torch import save_file
        sd = {k: v.contiguous() for k, v in (best[1] or model.state_dict()).items()}
        save_file(sd, str(OUT / "model.safetensors"))
        OmegaConf.save(cfg, OUT / "model_config.yaml")
        json.dump(nmap, open(OUT / "field_to_index_map.json", "w"))
        api.upload_folder(repo_id=MODEL_REPO, repo_type="model", folder_path=str(OUT), path_in_repo=RUN,
                          commit_message=f"{RUN}: checkpoint + metrics")
    small = [p for p in OUT.iterdir() if p.suffix in (".json", ".npz")]
    for p in small:
        api.upload_file(path_or_fileobj=str(p), path_in_repo=f"walrus_heart/{RUN}/{p.name}", repo_id=RUNS_REPO, repo_type="dataset")
    log("uploaded; done")


if __name__ == "__main__":
    main()
