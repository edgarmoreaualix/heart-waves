# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "h5py", "pyyaml", "huggingface_hub"]
# ///
"""Recompute The Well-style stats.yaml (mean/std/rms + mean_delta/rms_delta) over the train split, in float64."""
import glob, math, yaml, numpy as np, h5py
from huggingface_hub import snapshot_download, HfApi
REPO = "Ethgar/cardiac-ep-2d"
root = snapshot_download(REPO, repo_type="dataset", allow_patterns=["data/train/*"], max_workers=16)
acc = {k: np.zeros(5) for k in ("voltage", "recovery", "conductivity")}  # n, s, s2, nd, sd, (sd2 separately)
sd2 = {k: 0.0 for k in acc}
files = sorted(glob.glob(f"{root}/data/train/*.hdf5")); print(len(files), "files", flush=True)
for i, f in enumerate(files):
    with h5py.File(f) as h:
        for k in acc:
            a = h[f"t0_fields/{k}"][0].astype(np.float64)
            acc[k][0] += a.size; acc[k][1] += a.sum(); acc[k][2] += (a * a).sum()
            if k != "conductivity":
                d = np.diff(a, axis=0); acc[k][3] += d.size; acc[k][4] += d.sum(); sd2[k] += (d * d).sum()
    if i % 200 == 0: print(i, flush=True)
st = {"mean": {}, "std": {}, "rms": {}, "mean_delta": {}, "rms_delta": {}}
for k, (n, s, s2, nd, sd) in acc.items():
    m = s / n; st["mean"][k] = float(m); st["rms"][k] = float(math.sqrt(s2 / n)); st["std"][k] = float(math.sqrt(s2 / n - m * m))
    st["mean_delta"][k] = float(sd / nd) if nd else 0.0; st["rms_delta"][k] = float(math.sqrt(sd2[k] / nd)) if nd else 0.0
print(yaml.safe_dump(st))
open("/tmp/stats.yaml", "w").write(yaml.safe_dump(st))
HfApi().upload_file(path_or_fileobj="/tmp/stats.yaml", path_in_repo="stats.yaml", repo_id=REPO, repo_type="dataset",
                    commit_message="fix stats.yaml (float64 accumulation)")
print("uploaded", flush=True)
