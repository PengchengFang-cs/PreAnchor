"""M1 wrap-up: per-object multimodal clustering + top-down QA visualizations.

Clustering: greedy mode-merge in object-local (x, z, yaw) with 0.25 m / 30 deg
thresholds (no sklearn dependency).  Viz: world-frame top-down plots of a few
scenes (scene faces as grey dots, anchors as red arrows) + per-object local
scatter for the most-used objects.
"""
import json
import sys
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from o2a.trumans import Shard

OUT = ROOT / "out"
VIZ = OUT / "viz"
VIZ.mkdir(parents=True, exist_ok=True)

anchors = [json.loads(l) for l in open(OUT / "anchors.jsonl")]

# ---------------- clustering ----------------
POS_TH, YAW_TH = 0.25, np.radians(30)

def cluster(items):
    modes = []  # each: [sum_xz(2), sum_sin, sum_cos, n, members]
    for a in items:
        x, _, z, yaw = a["anchor"]
        placed = False
        for m in modes:
            cx, cz = m[0] / m[3]
            cyaw = np.arctan2(m[1], m[2])
            dyaw = abs((yaw - cyaw + np.pi) % (2 * np.pi) - np.pi)
            if np.hypot(x - cx, z - cz) < POS_TH and dyaw < YAW_TH:
                m[0] += np.array([x, z]); m[1] += np.sin(yaw); m[2] += np.cos(yaw)
                m[3] += 1; m[4].append(a); placed = True
                break
        if not placed:
            modes.append([np.array([x, z]), np.sin(yaw), np.cos(yaw), 1, [a]])
    return modes

groups = defaultdict(list)
for a in anchors:
    groups[(a["shard"], a["object"], a["verb"])].append(a)

n_modes = []
report = []
for (shard, obj, verb), items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    modes = cluster(items)
    n_modes.append(len(modes))
    report.append((shard, obj, verb, len(items), len(modes)))
print(f"groups={len(groups)}  modes/group: mean={np.mean(n_modes):.2f} "
      f"median={np.median(n_modes):.0f} max={max(n_modes)}")
print("top groups (shard, object, verb, n_anchors, n_modes):")
for r in report[:10]:
    print("  ", r)

# ---------------- per-object local scatter (top 6 groups) ----------------
for shard, obj, verb, n, k in report[:6]:
    items = groups[(shard, obj, verb)]
    fig, ax = plt.subplots(figsize=(5, 5))
    for a in items:
        x, _, z, yaw = a["anchor"]
        ax.plot(x, z, "r.", ms=6)
        ax.arrow(x, z, 0.12 * np.sin(yaw), 0.12 * np.cos(yaw),
                 head_width=0.03, color="b", alpha=0.6)
    ax.set_title(f"{obj} [{verb}] n={n} modes={k}\n{shard}", fontsize=8)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    safe = obj.replace("/", "_").replace(":", "_")
    fig.savefig(VIZ / f"local_{shard}_{safe}_{verb}.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)

# ---------------- world-frame QA for 3 shards ----------------
by_shard = defaultdict(list)
for a in anchors:
    by_shard[a["shard"]].append(a)
top_shards = sorted(by_shard, key=lambda s: -len(by_shard[s]))[:3]
for skey in top_shards:
    tar = glob.glob(f"/scratch/pf2m24/data/trumans/shard_tars/{skey}.tar")
    if not tar:
        continue
    sh = Shard(tar[0])
    c = sh.face_c[::37]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(c[:, 0], c[:, 2], ".", ms=0.5, color="0.75", zorder=0)
    for a in by_shard[skey]:
        x, _, z, yaw = a["world"]
        col = "r" if a["verb"] == "sit" else "m"
        ax.plot(x, z, col + ".", ms=7, zorder=2)
        ax.arrow(x, z, 0.2 * np.sin(yaw), 0.2 * np.cos(yaw),
                 head_width=0.05, color=col, alpha=0.8, zorder=2)
    ax.set_title(f"{skey}: {len(by_shard[skey])} anchors (red=sit magenta=lie)")
    ax.set_aspect("equal")
    fig.savefig(VIZ / f"world_{skey}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
print(f"viz written to {VIZ}")
