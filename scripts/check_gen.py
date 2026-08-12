"""Verify a Kimodo generation against its anchor constraints.

Usage: python check_gen.py out/gen_sit_0.npz out/constraints_sit_0.json
Prints terminal root error vs anchor and saves a top-down trajectory plot.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

gen_path, cons_path = Path(sys.argv[1]), Path(sys.argv[2])
d = np.load(gen_path, allow_pickle=True)
print("npz keys:", list(d.keys()))
root = d["root_positions"]            # [T, 3]
heading = d.get("global_root_heading")  # [T, 2] (cos, sin) if present
T = len(root)

cons = json.loads(cons_path.read_text())[0]
fi = np.array(cons["frame_indices"])
pts = np.array(cons["smooth_root_2d"])
heads = np.array(cons.get("global_root_heading", []))

term = fi >= T - 8
tgt_xz = pts[term].mean(0)
end_xz = root[-1, [0, 2]]
print(f"terminal target (x,z)={tgt_xz.round(3)}  generated end={end_xz.round(3)}"
      f"  err={np.linalg.norm(tgt_xz - end_xz):.3f} m")
print(f"root height: start={root[0,1]:.3f}  end={root[-1,1]:.3f} "
      f"(sit expected ~0.45-0.65 raw pelvis)")
if heading is not None and len(heads):
    tgt = heads[term].mean(0)
    tgt_ang = np.arctan2(tgt[1], tgt[0])
    end_ang = np.arctan2(heading[-1, 1], heading[-1, 0])
    err = np.degrees(abs((end_ang - tgt_ang + np.pi) % (2 * np.pi) - np.pi))
    print(f"terminal heading err = {err:.1f} deg")

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(root[:, 0], root[:, 2], "b-", lw=1, label="generated root")
ax.plot(pts[:, 0], pts[:, 1], "rx", ms=8, label="constraints")
ax.plot(root[0, 0], root[0, 2], "g^", ms=10, label="start")
ax.plot(root[-1, 0], root[-1, 2], "rs", ms=10, label="end")
ax.legend(); ax.set_aspect("equal"); ax.grid(alpha=0.3)
out = gen_path.with_suffix(".traj.png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved {out}")
