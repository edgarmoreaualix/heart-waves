"""Render Walrus-Heart comparison -> ../walrus_heart.mp4 / .png (1080x1080), style of ../render.py.

usage: python render_walrus.py [RES_DIR] [ID_TOP] [ID_BOTTOM]
RES_DIR holds <run>/rollouts_video.npz + rollouts_per_traj.json for runs FT and SCRATCH below; GT from gt_cache/.
"""
import sys, json
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

HERE = Path(__file__).parent
BG, FG, MUTED, ACCENT, RED, GREEN = "#0b0d12", "#e8e8ea", "#8a8f9c", "#ffb347", "#ff5a5f", "#7CFC9A"
FT_C, SC_C = "#5fb8ff", "#ff7a59"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#2a2e38",
                     "font.family": "DejaVu Sans", "font.size": 13})
CMAP = LinearSegmentedColormap.from_list("ap", ["#0b0d12", "#1b2a6b", "#3d6bd8", "#f2c14e", "#ff5a2a", "#fff2c2"])
SCAR = ListedColormap(["#00000000", "#8a8f9c"])
TU_MS = 12.9; DT = 3.0
FT, SC = "pretrained_n1600", "scratch_n1600"


def load(res, run, i):
    d = np.load(res / run / "rollouts_video.npz"); k = list(d["ids"]).index(i)
    per = json.load(open(res / run / "rollouts_per_traj.json"))[str(i)]
    return int(d["starts"][k]), d[f"pred_{i}"].astype(np.float32).transpose(0, 2, 1), per   # -> (t, y, x)


def main(res=HERE / "results", ids=(19, 1209), out=HERE.parent / "walrus_heart"):
    res = Path(res); m_ft = json.load(open(res / FT / "metrics.json")); m_sc = json.load(open(res / SC / "metrics.json"))
    rows = []
    for i in ids:
        with h5py.File(next((HERE / "gt_cache/data/test").glob(f"cardiac_ep_{i:05d}.hdf5"))) as h:
            gt = h["t0_fields/voltage"][0].transpose(0, 2, 1); cond = h["t0_fields/conductivity"][0].T
            s2 = float(h.attrs["s2_time"])
        s, pf, perf = load(res, FT, i); s_, ps, pers = load(res, SC, i); assert s == s_
        rows.append(dict(i=i, gt=gt, cond=cond, s=s, ft=pf, sc=ps, eft=np.array(perf["rmse_u"]), esc=np.array(pers["rmse_u"]),
                         s2=s2, label=perf["label"], tissue=perf["tissue"], pft=perf["pred_final"], psc=pers["pred_final"]))
    s0 = min(r["s"] for r in rows) - 5; nT = gt.shape[0]
    ms = lambda f: f * DT * TU_MS / 1000

    fig = plt.figure(figsize=(10.8, 10.8), dpi=100)
    fig.text(0.04, 0.965, "Walrus-Heart: can a fluid-physics AI learn the heartbeat?", fontsize=22, weight="bold", va="top")
    fig.text(0.04, 0.918, "Same 1.3B transformer, same training budget: pretrained on fluids (Walrus) vs from scratch · "
             "colour = voltage, grey = scar", fontsize=11.5, color=MUTED, va="top")
    cols = ["Simulation (truth)", "Walrus fine-tuned", "From scratch"]
    ccol = [FG, FT_C, SC_C]
    W, H0, gap = 0.29, 0.29, 0.015; x0 = 0.04; ytop = [0.585, 0.27]
    ims, badges = [], []
    for r_i, r in enumerate(rows):
        row_ims = []
        for c in range(3):
            ax = fig.add_axes([x0 + c * (W + gap), ytop[r_i], W, H0]); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_color("#2a2e38")
            im = ax.imshow(r["gt"][s0], cmap=CMAP, vmin=0, vmax=1, interpolation="bilinear")
            ax.imshow((r["cond"] == 0).astype(float), cmap=SCAR, vmin=0, vmax=1, interpolation="nearest")
            if r_i == 0:
                ax.set_title(cols[c], color=ccol[c], fontsize=14, weight="bold", pad=6)
            row_ims.append(im)
            if c > 0:
                b = ax.text(0.03, 0.04, "", transform=ax.transAxes, fontsize=10.5, color="white", weight="bold",
                            bbox=dict(fc="#000000b0", ec="none", pad=3)); badges.append((r_i, c, b))
        lab = "spiral re-entry" if r["label"] else "terminates"
        desc = {"healthy": "healthy tissue", "infarct_scar": "post-infarct scar", "diffuse_fibrosis": "fibrosis"}[r["tissue"]]
        fig.text(x0, ytop[r_i] - 0.012, f"test case #{r['i']} · {desc} · premature beat at {ms(r['s2']/DT):.2f} s → truth: {lab}",
                 fontsize=10.5, color=MUTED, va="top")
        ims.append(row_ims)
        if r_i == 0:
            clock = fig.axes[0].text(0.97, 0.04, "", transform=fig.axes[0].transAxes, ha="right", fontsize=11, color="white",
                                     bbox=dict(fc="#000000b0", ec="none", pad=3))

    axe = fig.add_axes([0.10, 0.075, 0.84, 0.13])
    for r_i, (r, ls) in enumerate(zip(rows, ["-", "--"])):
        t = ms(np.arange(r["s"] + 1, nT))
        axe.plot(t, r["eft"], color=FT_C, lw=1, alpha=0.25, ls=ls); axe.plot(t, r["esc"], color=SC_C, lw=1, alpha=0.25, ls=ls)
    lines = [(axe.plot([], [], color=FT_C, lw=2, ls=ls)[0], axe.plot([], [], color=SC_C, lw=2, ls=ls)[0]) for ls in ["-", "--"]]
    ymax = max(max(r["eft"].max(), r["esc"].max()) for r in rows) * 1.1
    axe.set_xlim(ms(s0), ms(nT - 1)); axe.set_ylim(0, ymax); axe.set_xlabel("seconds", fontsize=11)
    axe.set_ylabel("voltage error\n(RMSE)", fontsize=10)
    for sp in ("top", "right"): axe.spines[sp].set_visible(False)
    from matplotlib.lines import Line2D
    axe.legend(handles=[Line2D([], [], color=FT_C, lw=2, label="Walrus fine-tuned"), Line2D([], [], color=SC_C, lw=2, label="from scratch"),
                        Line2D([], [], color=MUTED, lw=1.5, label="top case"), Line2D([], [], color=MUTED, lw=1.5, ls="--", label="bottom case")],
               loc="upper left", ncol=4, fontsize=9.5, frameon=False, bbox_to_anchor=(0, 1.22), labelcolor=FG)
    a_ft = m_ft["test_rollout"]["reentry_acc_final"]; a_sc = m_sc["test_rollout"]["reentry_acc_final"]
    v_ft = m_ft["test_one_step"]["vrmse_mean"]; v_sc = m_sc["test_one_step"]["vrmse_mean"]
    n_r = m_ft["test_rollout"]["n"]
    fig.text(0.04, 0.012, f"Held-out sims · re-entry vs termination called right: Walrus {a_ft:.0%} · scratch {a_sc:.0%} (n={n_r}) · "
             f"next-frame VRMSE {v_ft:.3f} vs {v_sc:.3f}", fontsize=10.5, color=MUTED)

    w = FFMpegWriter(fps=24, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow"])
    still_at = min(nT - 1, max(r["s"] for r in rows) + int(__import__("os").environ.get("STILL_K", "10")))
    with w.saving(fig, f"{out}.mp4", dpi=100):
        for f in range(s0, nT):
            for r, row_ims, ln in zip(rows, ims, lines):
                row_ims[0].set_data(r["gt"][f])
                k = f - r["s"] - 1
                row_ims[1].set_data(r["ft"][k] if k >= 0 else r["gt"][f]); row_ims[2].set_data(r["sc"][k] if k >= 0 else r["gt"][f])
                if k >= 0:
                    t = ms(np.arange(r["s"] + 1, f + 1)); ln[0].set_data(t, r["eft"][:k + 1]); ln[1].set_data(t, r["esc"][:k + 1])
            for r_i, c, b in badges:
                r = rows[r_i]; k = f - r["s"] - 1
                if k < 0:
                    b.set_text("given (history)"); b.set_color(MUTED)
                else:
                    b.set_text(f"predicting · +{ms(k + 1):.1f} s"); b.set_color("white")
            clock.set_text(f"t = {ms(f):.2f} s")
            w.grab_frame()
            if f == still_at:
                fig.savefig(f"{out}.png", dpi=100)
        for _ in range(24):
            w.grab_frame()


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else HERE / "results", tuple(int(x) for x in a[1:3]) if len(a) >= 3 else (19, 1209))
