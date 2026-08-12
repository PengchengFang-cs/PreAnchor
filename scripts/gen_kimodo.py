"""M3 first loop: one GT sit anchor -> Kimodo constraints -> generate.

v1 uses root2d constraints only (start + waypoints + terminal anchor with
heading).  Start is synthesized 2.5 m in front of the anchor's facing
direction, walking in.  Height of sitting comes from the text prompt.

Usage:
  python gen_kimodo.py [anchor_index] [--duration 8]
NOTE on conventions: our yaw = atan2(fwd_x, fwd_z), heading vec = (sin, cos).
Kimodo heading (cos t, sin t) with t from compute_heading_angle -- verify sign
on first render and flip HERE if the character faces backward.
"""
import json
import sys
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FPS = 30
MODEL = "kimodo-smplx-rp"

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
duration = 8.0
if "--duration" in sys.argv:
    duration = float(sys.argv[sys.argv.index("--duration") + 1])
T = int(duration * FPS)

anchors = [json.loads(l) for l in open(OUT / "anchors.jsonl")]
sits = [a for a in anchors if a["verb"] == "sit"]
a = sits[idx]
ax, ay, az, ayaw = a["world"]
print(f"anchor #{idx}: {a['shard']} obj={a['object']} "
      f"world=({ax:.2f},{az:.2f}) yaw={np.degrees(ayaw):.0f}deg text='{a['text']}'")

# start: 2.5 m in front of where the seated character faces (i.e. approach
# from the open side), walking toward the seat.
fwd = np.array([np.sin(ayaw), np.cos(ayaw)])      # (x, z) facing of seated pose
start_xz = np.array([ax, az]) + 2.5 * fwd
walk_dir = -fwd                                    # walking toward the seat
walk_yaw = float(np.arctan2(walk_dir[0], walk_dir[1]))

def heading(yaw):
    return [float(np.cos(yaw)), float(np.sin(yaw))]

frames, points, heads = [], [], []
# start (2 frames)
for f in (0, 1):
    frames.append(f); points.append(list(start_xz)); heads.append(heading(walk_yaw))
# waypoints at 40% / 70% of the approach, on the straight line
for frac, tfrac in ((0.4, 0.35), (0.75, 0.6)):
    p = start_xz + frac * (np.array([ax, az]) - start_xz)
    frames.append(int(tfrac * T)); points.append(list(map(float, p)))
    heads.append(heading(walk_yaw))
# terminal anchor: last 6 frames seated at (x,z) facing ayaw
for f in range(T - 6, T):
    frames.append(f); points.append([float(ax), float(az)])
    heads.append(heading(ayaw))

constraints = [{
    "type": "root2d",
    "frame_indices": frames,
    "smooth_root_2d": points,
    "global_root_heading": heads,
}]
cpath = OUT / f"constraints_sit_{idx}.json"
cpath.write_text(json.dumps(constraints, indent=1))
print(f"wrote {cpath}  ({len(frames)} constrained frames / {T} total)")

prompt = "walk forward and sit down"
out_stem = OUT / f"gen_sit_{idx}"
cmd = [sys.executable, "-m", "kimodo.scripts.generate", prompt,
       "--model", MODEL, "--constraints", str(cpath),
       "--duration", str(duration), "--output", str(out_stem),
       "--seed", "1"]
print("running:", " ".join(cmd))
subprocess.run(cmd, check=True)
