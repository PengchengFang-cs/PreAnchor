"""Overlay the target chair mesh onto a generated motion; joint-level
contact/clearance metrics + top/side visual QA.

v1 metrics (no SMPL-X body surface yet -> joint-based proxies):
  - pelvis vs seat: horizontal offset to seat centroid, vertical gap to seat h
  - clearance: min joint->chair-vertex distance (cKDTree) for pelvis/knees
  - penetration proxy: pelvis below seat height while over the seat footprint

Usage: python scene_check.py [sit_index]
"""
import json
import sys
import glob
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from o2a.trumans import Shard

OUT = ROOT / "out"
idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
gen_path = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT / f"gen_sit_{idx}.npz"

anchors = [json.loads(l) for l in open(OUT / "anchors.jsonl")]
a = [x for x in anchors if x["verb"] == "sit"][idx]
gen = np.load(gen_path)
J = gen["posed_joints"]          # [T, nJ, 3], world (= TRUMANS world)
T, nJ, _ = J.shape
print(f"anchor: {a['object']} @ {a['shard']} frame={a['frame']}  joints={nJ}")

# --- chair mesh posed at commitment frame ---
tar = glob.glob(f"/scratch/pf2m24/data/trumans/shard_tars/{a['shard']}.tar")[0]
sh = Shard(tar)
name = a["object"].split("movable:")[-1]
if a["object"].startswith("movable:"):
    m = sh.movable[name]
    p = sh.obj_poses
    sel = np.where(p["sample_id"] == a["sample_id"])[0][0]
    lo, hi = int(p["sample_start"][sel]), int(p["sample_end"][sel])
    onames = list(p["object_names"])
    k = onames.index(name)
    rows = np.where(p["object_index"][lo:hi] == k)[0]
    fr = p["frames"][lo:hi][rows]
    Tm = p["transforms"][lo:hi][rows[np.argmin(np.abs(fr - a["frame"]))]]
    V = m["vertices"] @ Tm[:3, :3].T + Tm[:3, 3]
else:
    oid = sh.scene_names.index(name)
    vo = np.load(tar, allow_pickle=True)  # placeholder; static path unused v1
    raise SystemExit("static-object path not needed for this sample")

tree = cKDTree(V)
seat_h = a["seat_height"]

# --- joint groups (SMPL-X ordering assumed; verify with nJ) ---
IDX = {"pelvis": 0, "l_hip": 1, "r_hip": 2, "l_knee": 4, "r_knee": 5,
       "l_ankle": 7, "r_ankle": 8}

pelvis = J[:, IDX["pelvis"]]
last = slice(T - 30, T)  # final second

# metrics
seat_centroid = V[np.abs(V[:, 1] - seat_h) < 0.08].mean(0)
horiz_off = np.linalg.norm(pelvis[-1, [0, 2]] - seat_centroid[[0, 2]])
vert_gap = pelvis[-1, 1] - seat_h
print(f"seat height={seat_h:.3f}  seat centroid xz={seat_centroid[[0,2]].round(3)}")
print(f"pelvis end: horiz offset to seat centroid={horiz_off:.3f} m, "
      f"height above seat={vert_gap:.3f} m")

for jn in ("pelvis", "l_knee", "r_knee", "l_hip", "r_hip"):
    d = tree.query(J[last, IDX[jn]])[0]
    print(f"{jn:8s} min dist to chair (last 1s): {d.min():.3f} m "
          f"(mean {d.mean():.3f})")

over_seat = np.linalg.norm(pelvis[:, [0, 2]] - seat_centroid[None, [0, 2]],
                           axis=1) < 0.35
below = pelvis[:, 1] < seat_h - 0.03
pen_frames = int((over_seat & below).sum())
print(f"penetration proxy (pelvis below seat while over footprint): "
      f"{pen_frames}/{T} frames")

# --- visuals: top view + side view at final frame ---
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
ax = axes[0]
ax.plot(V[::23, 0], V[::23, 2], ".", ms=0.8, color="0.7", zorder=0)
ax.plot(pelvis[:, 0], pelvis[:, 2], "b-", lw=1)
ax.plot(J[-1, :, 0], J[-1, :, 2], "r.", ms=4)
ax.set_title("top view (x-z)"); ax.set_aspect("equal")
ax = axes[1]
near = np.abs(V[:, 0] - pelvis[-1, 0]) < 0.6
ax.plot(V[near][::11, 2], V[near][::11, 1], ".", ms=0.8, color="0.7", zorder=0)
ax.plot(J[-1, :, 2], J[-1, :, 1], "r.", ms=5)
for f in range(T - 90, T, 15):  # skeleton root descent trace
    ax.plot(J[f, 0, 2], J[f, 0, 1], "b.", ms=3, alpha=0.5)
ax.axhline(seat_h, color="g", lw=0.8, ls="--", label=f"seat h={seat_h:.2f}")
ax.set_title("side view (z-y), final pose"); ax.set_aspect("equal"); ax.legend()
out = OUT / f"scene_check_{gen_path.stem}.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved {out}")
