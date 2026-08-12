"""M1: full-dataset GT anchor extraction (S category: sit / lie).

For every classified segment: detect commitment -> bind support object ->
object local frame -> anchor (x,y,z,yaw)@object + delta (root above seat).
Writes anchors.jsonl and prints summary stats.
"""
import json
import sys
import glob
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from o2a.taxonomy import classify
from o2a.trumans import Shard
from o2a.support import (detect_commitment, bind_support, object_frame,
                         to_object_frame, yaw_of)

DATA = Path("/scratch/pf2m24/data/trumans/shard_tars")
OUT = Path(__file__).resolve().parents[1] / "out"
OUT.mkdir(exist_ok=True)

anchors, stats = [], {"segments": 0, "no_commit": 0, "no_bind": 0, "ok": 0}
for tar in sorted(glob.glob(str(DATA / "*.tar"))):
    try:
        shard = Shard(tar)
    except Exception as e:
        print(f"[skip shard] {Path(tar).name}: {e}")
        continue
    scene_key = Path(tar).stem
    for sample in shard.samples():
        sid = sample["sample_id"]
        for seg in shard.segments(sid):
            spec = classify(seg["text"])
            if spec is None:
                continue
            stats["segments"] += 1
            ci = detect_commitment(seg, sample, spec.verb)
            if ci is None:
                stats["no_commit"] += 1
                continue
            root = sample["transl"][ci]
            yaw = yaw_of(sample["global_orient"][ci])
            frame = int(sample["frame_id"][ci])
            bound = bind_support(shard, sid, frame, root)
            if bound is None:
                stats["no_bind"] += 1
                continue
            obj_key, centers, seat_h = bound
            origin, yaw0 = object_frame(centers)
            local, yaw_l = to_object_frame(root, yaw, origin, yaw0)
            stats["ok"] += 1
            anchors.append({
                "shard": scene_key, "sample_id": sid, "text": seg["text"],
                "verb": spec.verb, "frame": frame,
                "object": obj_key, "seat_height": round(seat_h, 4),
                "delta": round(float(root[1] - seat_h), 4),
                "world": [round(float(v), 4) for v in root] + [round(yaw, 4)],
                "obj_origin": [round(float(v), 4) for v in origin],
                "obj_yaw0": round(yaw0, 4),
                "anchor": [round(float(v), 4) for v in local] + [round(yaw_l, 4)],
            })
    print(f"{scene_key}: cumulative ok={stats['ok']}")

with open(OUT / "anchors.jsonl", "w") as f:
    for a in anchors:
        f.write(json.dumps(a) + "\n")

print("\n=== M1 extraction summary ===")
print(stats)
for verb in ("sit", "lie"):
    sub = [a for a in anchors if a["verb"] == verb]
    if not sub:
        continue
    d = np.array([a["delta"] for a in sub])
    sh = np.array([a["seat_height"] for a in sub])
    print(f"{verb}: n={len(sub)}  seat_h median={np.median(sh):.3f} "
          f"(p10={np.quantile(sh,0.1):.3f}, p90={np.quantile(sh,0.9):.3f})  "
          f"delta median={np.median(d):.3f} (p10={np.quantile(d,0.1):.3f}, "
          f"p90={np.quantile(d,0.9):.3f})")
objs = {}
for a in anchors:
    objs.setdefault((a["shard"], a["object"]), []).append(a)
multi = sum(1 for v in objs.values() if len(v) >= 3)
print(f"unique (scene,object) pairs: {len(objs)}; with >=3 anchors: {multi}")
