"""Object2Anchor — step 1: explore TRUMANS coordinate conventions and extract
first sit-down anchors from one shard.

Outputs (printed):
  1. up-axis / facing-direction convention check
  2. per-"sit down"-segment: commitment frame, pelvis (x,y,z,yaw), nearest
     scene object under pelvis, pelvis expressed in that object's local frame
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

SHARD_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/iridisfs/scratch/pf2m24/.xdg-runtime/claude-56935/"
    "-iridisfs-scratch-pf2m24-projects-GR00T-WholeBodyControl/"
    "219d4785-a8c7-432c-828a-2d7d07d97f91/scratchpad/trumans_peek")
FPS_GUESS = 30.0

humans = np.load(SHARD_DIR / "humans.npz")
scene = np.load(SHARD_DIR / "scene.npz", allow_pickle=True)
obj_poses = np.load(SHARD_DIR / "objects.poses.npz", allow_pickle=True)
text = json.loads((SHARD_DIR / "text.json").read_text())

sample_ids = humans["sample_id"]
sample_start, sample_end = humans["sample_start"], humans["sample_end"]
frame_id = humans["frame_id"]
transl = humans["transl"]
global_orient = humans["global_orient"]

# ---------- 1. coordinate conventions ----------
print("=== coordinate check ===")
print(f"transl mean {transl.mean(0).round(3)}  std {transl.std(0).round(3)}")
print(f"transl min  {transl.min(0).round(3)}  max {transl.max(0).round(3)}")
# height axis = the axis with small std and mean around ~1.0 (standing pelvis)
up_axis = int(np.argmin(transl.std(0)))
print(f"guessed up axis: {['x','y','z'][up_axis]}")

rot0 = R.from_rotvec(global_orient[:200]).as_matrix()  # first 200 frames
for name, vec in [("+Y(body up?)", [0, 1, 0]), ("+Z(body fwd?)", [0, 0, 1])]:
    world = rot0 @ np.array(vec, float)
    print(f"canonical {name} -> world mean {world.mean(0).round(3)}")

# ---------- helpers ----------
scene_v = scene["vertices"]
scene_names = list(scene["object_names"])
v_off = scene["object_vertex_offsets"]

def yaw_of(rotvec):
    """heading angle of the body's forward direction projected on ground."""
    fwd = R.from_rotvec(rotvec).as_matrix() @ np.array([0.0, 0.0, 1.0])
    ground = [i for i in range(3) if i != up_axis]
    return float(np.arctan2(fwd[ground[1]], fwd[ground[0]]))

def nearest_objects(point, k=4, max_dist=1.0):
    """scene objects with any vertex within max_dist of point (coarse)."""
    out = []
    for i, name in enumerate(scene_names):
        verts = scene_v[v_off[i]:v_off[i + 1]]
        if len(verts) == 0:
            continue
        d = np.linalg.norm(verts - point[None], axis=1).min()
        if d < max_dist:
            out.append((d, name, i))
    return sorted(out)[:k]

# ---------- 2. sit-down segments ----------
print("\n=== sit-down anchors (v0 commitment = min-height + low-velocity) ===")
for s in text["samples"]:
    sid = s["sample_id"]
    row = np.where(sample_ids == sid)[0]
    if len(row) == 0:
        continue
    lo, hi = int(sample_start[row[0]]), int(sample_end[row[0]])
    fids = frame_id[lo:hi]
    for seg in s["segments"]:
        if "sit down" not in seg["text"].lower() or seg.get("human", 0) != 0:
            continue
        # rows of this segment (match by frame_id), plus 1s of context after
        m = (fids >= seg["start"]) & (fids <= seg["end"] + int(FPS_GUESS))
        rows = np.arange(lo, hi)[m]
        if len(rows) < 5:
            continue
        h = transl[rows, up_axis]
        vel = np.abs(np.gradient(h)) * FPS_GUESS
        # commitment: lowest 20% of heights with low vertical speed
        stable = vel < 0.25
        cand = np.where(stable & (h < np.quantile(h, 0.2) + 0.05))[0]
        ci = int(cand[len(cand) // 2]) if len(cand) else int(np.argmin(h))
        crow = rows[ci]
        p = transl[crow]
        yaw = yaw_of(global_orient[crow])
        near = nearest_objects(p)
        near_str = ", ".join(f"{n}({d:.2f}m)" for d, n, _ in near) or "none<1m"
        print(f"sample {sid} seg[{seg['start']}-{seg['end']}] "
              f"commit_frame={fids[m][ci]} pelvis={p.round(3)} "
              f"yaw={np.degrees(yaw):.0f}deg  h_start={h[0]:.2f} h_commit={h[ci]:.2f}")
        print(f"    near: {near_str}")
