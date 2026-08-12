"""3D visualization: chair mesh + SMPL-X 22-joint skeleton, frame strips + GIFs.

Usage: python viz3d.py            # renders all three arms of anchor 0
Outputs to out/viz3d/: {tag}_strip.png (5 key frames) and {tag}.gif (~60 frames).
"""
import json
import sys
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from o2a.trumans import Shard

OUT = ROOT / "out"
VIZ = OUT / "viz3d"
VIZ.mkdir(exist_ok=True)

# SMPL-X body: (child, parent)
BONES = [(1, 0), (2, 0), (3, 0), (4, 1), (5, 2), (6, 3), (7, 4), (8, 5),
         (9, 6), (10, 7), (11, 8), (12, 9), (13, 9), (14, 9), (15, 12),
         (16, 13), (17, 14), (18, 16), (19, 17), (20, 18), (21, 19)]

anchors = [json.loads(l) for l in open(OUT / "anchors.jsonl")]
a = [x for x in anchors if x["verb"] == "sit"][0]

# --- chair mesh posed at commitment frame (decimated for rendering) ---
tar = glob.glob(f"/scratch/pf2m24/data/trumans/shard_tars/{a['shard']}.tar")[0]
sh = Shard(tar)
name = a["object"].split("movable:")[-1]
m = sh.movable[name]
p = sh.obj_poses
sel = np.where(p["sample_id"] == a["sample_id"])[0][0]
lo, hi = int(p["sample_start"][sel]), int(p["sample_end"][sel])
k = list(p["object_names"]).index(name)
rows = np.where(p["object_index"][lo:hi] == k)[0]
fr = p["frames"][lo:hi][rows]
T4 = p["transforms"][lo:hi][rows[np.argmin(np.abs(fr - a["frame"]))]]
V = m["vertices"] @ T4[:3, :3].T + T4[:3, 3]
chair_pts = V[::18]                           # ~9k points for rendering

cx, cz = a["world"][0], a["world"][2]
XL = (cx - 1.3, cx + 1.3)
ZL = (cz - 2.8, cz + 0.8)


def draw_frame(ax, J, ghost=None):
    """Draw chair + floor + one skeleton pose (world Y-up -> plot z up)."""
    # floor grid
    for gx in np.arange(np.floor(XL[0]), np.ceil(XL[1]) + 0.01, 0.5):
        ax.plot([gx, gx], list(ZL), [0, 0], color="0.88", lw=0.6)
    for gz in np.arange(np.floor(ZL[0]), np.ceil(ZL[1]) + 0.01, 0.5):
        ax.plot(list(XL), [gz, gz], [0, 0], color="0.88", lw=0.6)
    ax.scatter(chair_pts[:, 0], chair_pts[:, 2], chair_pts[:, 1],
               color="0.55", s=1.2, alpha=0.5, linewidths=0)
    if ghost is not None:                     # faded earlier poses
        for g in ghost:
            for c, pa in BONES:
                ax.plot([g[c, 0], g[pa, 0]], [g[c, 2], g[pa, 2]],
                        [g[c, 1], g[pa, 1]], color="steelblue",
                        lw=1.0, alpha=0.25)
    for c, pa in BONES:
        ax.plot([J[c, 0], J[pa, 0]], [J[c, 2], J[pa, 2]],
                [J[c, 1], J[pa, 1]], color="crimson", lw=2.4)
    ax.scatter(J[:, 0], J[:, 2], J[:, 1], color="crimson", s=9)
    ax.set_xlim(*XL)
    ax.set_ylim(*ZL)
    ax.set_zlim(0, 1.9)
    ax.set_box_aspect((2.6, 3.6, 1.9))
    ax.axis("off")


def render(tag, npz_path):
    J = np.load(npz_path)["posed_joints"]     # [T, 22, 3]
    T = len(J)

    # --- strip: 5 key frames ---
    keys = [0, int(0.5 * T), int(0.75 * T), int(0.9 * T), T - 1]
    fig = plt.figure(figsize=(16, 4.2))
    for i, f in enumerate(keys):
        ax = fig.add_subplot(1, 5, i + 1, projection="3d")
        ax.view_init(elev=12, azim=-15)
        draw_frame(ax, J[f])
        ax.set_title(f"frame {f} ({f/30:.1f}s)", fontsize=9)
    fig.suptitle(tag, fontsize=12)
    fig.tight_layout()
    fig.savefig(VIZ / f"{tag}_strip.png", dpi=95, bbox_inches="tight")
    plt.close(fig)

    # --- GIF ---
    frames = []
    sub = range(0, T, 4)                       # 60 frames @ ~7.5 fps playback
    for f in sub:
        fig = plt.figure(figsize=(4.4, 4.4))
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=12, azim=-15)
        ghost = [J[g] for g in range(0, f, 45)]
        draw_frame(ax, J[f], ghost=ghost)
        ax.set_title(f"{tag}  {f/30:.1f}s", fontsize=9)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(Image.fromarray(img))
        plt.close(fig)
    frames[0].save(VIZ / f"{tag}.gif", save_all=True, append_images=frames[1:],
                   duration=130, loop=0)
    print(f"{tag}: strip + gif done ({len(frames)} frames)")


if "--test" in sys.argv:
    J = np.load(OUT / "gen_sit_0_fb.npz")["posed_joints"]
    fig = plt.figure(figsize=(15, 4.2))
    for i, az in enumerate((-55, -15, 30, 75)):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        ax.view_init(elev=14, azim=az)
        draw_frame(ax, J[-1])
        ax.set_title(f"azim={az}", fontsize=9)
    fig.savefig(VIZ / "view_test.png", dpi=100, bbox_inches="tight")
    print("view test saved")
    raise SystemExit

for tag, path in [("arm1_root_xz_yaw", OUT / "gen_sit_0.npz"),
                  ("arm1b_root_xyz_yaw", OUT / "gen_sit_0_h.npz"),
                  ("arm2_fullbody_keyframe", OUT / "gen_sit_0_fb.npz")]:
    if Path(path).exists():
        render(tag, path)
print("all visualizations written to", VIZ)
