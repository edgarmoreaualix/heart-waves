"""Render story.npz -> heart_spiral.mp4 / .png (1080x1080, X-ready)."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import sim

HERE = Path(__file__).parent
BG, FG, MUTED, ACCENT, RED = "#0b0d12", "#e8e8ea", "#8a8f9c", "#ffb347", "#ff5a5f"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#2a2e38",
                     "font.family": "DejaVu Sans", "font.size": 13})
CMAP = LinearSegmentedColormap.from_list("ap", ["#0b0d12", "#1b2a6b", "#3d6bd8", "#f2c14e", "#ff5a2a", "#fff2c2"])


def main(src=HERE / "story.npz", out=HERE / "heart_spiral"):
    d = np.load(src)
    U, T, E, ET = d["u"].astype(np.float32), d["t"], d["ecg"], d["ecg_t"]
    s2 = float(d["s2"]); n = U.shape[1]
    ms = lambda t: t * sim.TU_MS / 1000
    E = E / np.abs(E[ET > 20]).max()

    fig = plt.figure(figsize=(10.8, 10.8), dpi=100)
    fig.text(0.04, 0.955, "How one mistimed heartbeat becomes an arrhythmia", fontsize=23, weight="bold", va="top")
    fig.text(0.04, 0.905, "2D heart tissue simulation (Aliev-Panfilov model) · colour = cell voltage · below: the ECG it produces",
             fontsize=12.5, color=MUTED, va="top")
    ax = fig.add_axes([0.20, 0.36, 0.60, 0.52])
    im = ax.imshow(U[0], cmap=CMAP, vmin=0, vmax=1, interpolation="bilinear", origin="upper")
    ax.set_xticks([]); ax.set_yticks([])
    cax = fig.add_axes([0.82, 0.40, 0.015, 0.44])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1]); cb.ax.set_yticklabels(["-80 mV\nrest", "-30", "+20 mV\nfiring"], color=MUTED, fontsize=10)
    cb.outline.set_visible(False)
    ex, ey = d["u"].shape[2] * 0.85, n * 0.5
    ax.plot([ex], [ey], marker="o", ms=9, mfc="none", mec="white", mew=2)
    ax.text(ex, ey - 9, "electrode", color="white", fontsize=10, ha="center")
    s2box = Rectangle((0, n / 2), n / 2, n / 2, fill=False, ec=RED, lw=2.5, ls="--", visible=False)
    ax.add_patch(s2box)
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)
    status = fig.text(0.5, 0.315, "", ha="center", fontsize=19, weight="bold")
    clock = ax.text(0.98, 0.03, "", transform=ax.transAxes, ha="right", fontsize=13, color="white",
                    bbox=dict(fc="#000000aa", ec="none", pad=3))

    axe = fig.add_axes([0.06, 0.095, 0.88, 0.19])
    axe.plot(ms(ET), E, color="#ffffff", alpha=0.07, lw=1)
    line, = axe.plot([], [], color="#7CFC9A", lw=1.8)
    dot, = axe.plot([], [], "o", color="#7CFC9A", ms=6)
    axe.axvline(ms(s2), color=RED, lw=1, ls=":", alpha=0.8)
    axe.text(ms(s2), 1.05, " premature beat", color=RED, fontsize=11, va="bottom")
    axe.set_xlim(0, ms(T[-1])); axe.set_ylim(-1.15, 1.25); axe.set_yticks([])
    axe.set_xlabel("seconds")
    for s in ("top", "right", "left"):
        axe.spines[s].set_visible(False)
    fig.text(0.04, 0.015, "Open simulation · normal pacing 58 bpm → spiral re-entry ~136 bpm (ventricular tachycardia)",
             fontsize=10.5, color=MUTED)

    w = FFMpegWriter(fps=30, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow"])
    still_at = int(np.searchsorted(T, s2 + 160))
    with w.saving(fig, f"{out}.mp4", dpi=100):
        for i, (u, t) in enumerate(zip(U, T)):
            im.set_data(u)
            m = ET <= t
            line.set_data(ms(ET[m]), E[m]); dot.set_data([ms(ET[m][-1])], [E[m][-1]])
            clock.set_text(f"t = {ms(t):.2f} s")
            s2box.set_visible(s2 - 4 <= t <= s2 + 12)
            if t < s2:
                status.set_text("Normal rhythm · each beat sweeps across, then the tissue rests"); status.set_color("#7CFC9A")
            elif t < s2 + 70:
                status.set_text("A premature stimulus hits tissue that is still recovering…"); status.set_color(RED)
            else:
                status.set_text("…the wave curls into a spiral that re-excites itself: 136 bpm"); status.set_color(ACCENT)
            w.grab_frame()
            if i == still_at:
                fig.savefig(f"{out}.png", dpi=100)
        for _ in range(20):
            w.grab_frame()
    plt.close(fig)


if __name__ == "__main__":
    main()
