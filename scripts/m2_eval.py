"""M2 acceptance: construct_sit() candidates vs GT anchor modes (Table 1).

Scope v1: STATIC objects only (536/778 sit anchors) — they don't move, so GT
modes are well-defined by world-frame clustering (0.25 m / 30 deg greedy,
same rule as M1).  Movable chairs need per-frame object frames -> later.

Metrics over (shard, object) groups:
  hit@k       fraction of groups with >=1 GT mode matched in top-k candidates
  coverage@k  fraction of all GT modes matched by top-k candidates
  (+ anchor-weighted coverage, candidate counts, precision proxy)

Usage: python m2_eval.py [--thresh 0.30]
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from o2a.trumans import Shard
from o2a.construct import construct_sit

OUT = ROOT / "out"
TARS = "/scratch/pf2m24/data/trumans/shard_tars"
POS_TH = float(sys.argv[sys.argv.index("--thresh") + 1]) if "--thresh" in sys.argv else 0.30
YAW_TH = np.radians(30)


def cluster_world(items):
    """greedy mode-merge on world (x,z,yaw); returns [(x,z,yaw,n), ...]."""
    modes = []
    for a in items:
        x, _, z, yaw = a["world"]
        for m in modes:
            cx, cz = m[0] / m[3]
            cyaw = np.arctan2(m[1], m[2])
            if (np.hypot(x - cx, z - cz) < 0.25 and
                    abs((yaw - cyaw + np.pi) % (2 * np.pi) - np.pi) < YAW_TH):
                m[0] += np.array([x, z]); m[1] += np.sin(yaw)
                m[2] += np.cos(yaw); m[3] += 1
                break
        else:
            modes.append([np.array([x, z]), np.sin(yaw), np.cos(yaw), 1])
    return [(m[0][0] / m[3], m[0][1] / m[3],
             float(np.arctan2(m[1], m[2])), m[3]) for m in modes]


def match_rank(mode, cands):
    """rank (1-based) of the first candidate matching the mode, or None."""
    mx, mz, myaw, _ = mode
    for r, c in enumerate(cands, 1):
        if (np.hypot(c["x"] - mx, c["z"] - mz) < POS_TH and
                abs((c["yaw"] - myaw + np.pi) % (2 * np.pi) - np.pi) < YAW_TH):
            return r
    return None


def main():
    anchors = [json.loads(l) for l in open(OUT / "anchors.jsonl")]
    sit = [a for a in anchors if a["verb"] == "sit"
           and not a["object"].startswith("movable:")]
    groups = defaultdict(list)
    for a in sit:
        groups[(a["shard"], a["object"])].append(a)
    print(f"{len(sit)} static sit anchors, {len(groups)} (shard,object) groups")

    rows = []
    for shard_name in sorted(set(k[0] for k in groups)):
        sh = Shard(glob.glob(f"{TARS}/{shard_name}.tar")[0])
        for (sname, obj), items in sorted(groups.items()):
            if sname != shard_name:
                continue
            oid = sh.scene_names.index(obj)
            sel = sh.face_obj == oid
            cands = construct_sit(sh.face_c[sel], sh.face_n[sel],
                                  sh.face_a[sel], sh.face_c,
                                  scene_other=sh.face_c[~sel])
            modes = cluster_world(items)
            ranks = [match_rank(m, cands) for m in modes]
            rows.append({"shard": sname, "object": obj,
                         "n_anchors": len(items), "n_modes": len(modes),
                         "n_cands": len(cands),
                         "mode_sizes": [m[3] for m in modes],
                         "ranks": ranks})
            rank_s = ",".join("-" if r is None else str(r) for r in ranks)
            print(f"{sname} {obj:34s} anchors={len(items):3d} "
                  f"modes={len(modes)} cands={len(cands):3d} ranks=[{rank_s}]")

    with open(OUT / "m2_eval.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # ---------------- aggregates ----------------
    def agg(k):
        hit = [any(r0 is not None and r0 <= k for r0 in r["ranks"]) for r in rows]
        matched = sum(1 for r in rows for r0 in r["ranks"]
                      if r0 is not None and r0 <= k)
        total = sum(r["n_modes"] for r in rows)
        wm = sum(s for r in rows for r0, s in zip(r["ranks"], r["mode_sizes"])
                 if r0 is not None and r0 <= k)
        wt = sum(sum(r["mode_sizes"]) for r in rows)
        return 100 * np.mean(hit), 100 * matched / total, 100 * wm / wt

    print(f"\nthresh: pos<{POS_TH} m, yaw<30 deg   "
          f"({len(rows)} groups, {sum(r['n_modes'] for r in rows)} GT modes)")
    print("| k | hit@k (组) | coverage@k (模式) | coverage@k (anchor加权) |")
    print("|---|---|---|---|")
    for k in (1, 3, 5, 10**9):
        h, c, w = agg(k)
        print(f"| {'all' if k > 100 else k} | {h:.0f}% | {c:.0f}% | {w:.0f}% |")
    nc = np.array([r["n_cands"] for r in rows])
    print(f"\ncandidates/object: median={np.median(nc):.0f} max={nc.max()}  "
          f"zero-candidate groups={int((nc == 0).sum())}")


if __name__ == "__main__":
    main()
