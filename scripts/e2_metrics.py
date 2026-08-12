"""Compliance metrics for batch E2 generations — movable AND static objects.

Extends scene_check.py's joint-proxy metrics to the full anchor set:
  - movable objects: mesh vertices posed at the commitment frame
  - static objects:  scene face centers (Shard.face_c) of that object
Seat slice is LOCAL to the target anchor (within 0.5 m of anchor x,z) so
multi-seat furniture (sofas, beds) is judged at the intended mode only.

Reads out/e2/gen_s{sit_idx}_{tag}.npz, writes out/e2/e2_results.jsonl and
prints + saves a per-arm summary (out/e2/E2_summary.md).

Usage: python e2_metrics.py
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from o2a.trumans import Shard

OUT = ROOT / "out"
E2 = OUT / "e2"
SEAT_TO_PELVIS = 0.17
IDX = {"pelvis": 0, "l_knee": 4, "r_knee": 5}
TARS = "/scratch/pf2m24/data/trumans/shard_tars"

_shards = {}


def shard(name):
    if name not in _shards:
        _shards[name] = Shard(glob.glob(f"{TARS}/{name}.tar")[0])
    return _shards[name]


def object_points(a):
    """world-space point cloud of the anchor's object at the commitment frame."""
    sh = shard(a["shard"])
    name = a["object"].split("movable:")[-1]
    if a["object"].startswith("movable:"):
        m = sh.movable[name]
        p = sh.obj_poses
        sel = np.where(p["sample_id"] == a["sample_id"])[0][0]
        lo, hi = int(p["sample_start"][sel]), int(p["sample_end"][sel])
        k = list(p["object_names"]).index(name)
        rows = np.where(p["object_index"][lo:hi] == k)[0]
        fr = p["frames"][lo:hi][rows]
        Tm = p["transforms"][lo:hi][rows[np.argmin(np.abs(fr - a["frame"]))]]
        return m["vertices"] @ Tm[:3, :3].T + Tm[:3, 3]
    oid = sh.scene_names.index(name)
    return sh.face_c[sh.face_obj == oid]


def seat_centroid(pts, a):
    """centroid of the seat surface near the TARGET spot (local mode)."""
    axz = np.array(a["world"])[[0, 2]]
    near = np.linalg.norm(pts[:, [0, 2]] - axz, axis=1) < 0.5
    for tol in (0.08, 0.15, 0.25):
        band = near & (np.abs(pts[:, 1] - a["seat_height"]) < tol)
        if band.sum() >= 3:
            return pts[band].mean(0)
    return np.array([axz[0], a["seat_height"], axz[1]])


def metrics(a, gen):
    J = gen["posed_joints"]
    root = gen["root_positions"]
    T = len(root)
    pts = object_points(a)
    cen = seat_centroid(pts, a)
    seat_h = a["seat_height"]
    ax, _, az, ayaw = a["world"]

    r = {"xz_err": float(np.linalg.norm(root[-1, [0, 2]] - [ax, az])),
         "y_err": float(root[-1, 1] - (seat_h + SEAT_TO_PELVIS)),
         "horiz_off": float(np.linalg.norm(J[-1, 0, [0, 2]] - cen[[0, 2]])),
         "vert_gap": float(J[-1, 0, 1] - seat_h)}

    hd = gen.get("global_root_heading")
    if hd is not None and len(np.shape(hd)) == 2:
        end = np.arctan2(hd[-1, 1], hd[-1, 0])
        r["yaw_err_deg"] = float(np.degrees(
            abs((end - ayaw + np.pi) % (2 * np.pi) - np.pi)))

    pelvis = J[:, IDX["pelvis"]]
    over = np.linalg.norm(pelvis[:, [0, 2]] - cen[None, [0, 2]], axis=1) < 0.35
    below = pelvis[:, 1] < seat_h - 0.03
    r["pen_frames"] = int((over & below).sum())

    tree = cKDTree(pts)
    last = slice(T - 30, T)
    r["knee_min_d"] = float(min(tree.query(J[last, IDX[j]])[0].min()
                                for j in ("l_knee", "r_knee")))
    return r


def fmt(rows, key, scale=1.0, pct=False):
    v = np.array([r[key] for r in rows if key in r]) * scale
    if pct:
        return f"{100 * v.mean():.0f}%"
    return f"{np.median(v):.3f}" if len(v) else "-"


def main():
    anchors = [json.loads(l) for l in open(OUT / "anchors.jsonl")]
    sit = [x for x in anchors if x["verb"] == "sit"]

    rows = []
    for f in sorted(glob.glob(str(E2 / "gen_s*_*.npz"))):
        m = re.match(r"gen_s(\d+)_(\w+)\.npz", Path(f).name)
        i, tag = int(m.group(1)), m.group(2)
        a = sit[i]
        gen = np.load(f, allow_pickle=True)
        r = {"sit_idx": i, "arm": tag, "object": a["object"],
             "seat_h": a["seat_height"], **metrics(a, gen)}
        rows.append(r)
        print(f"s{i:<4d} {tag:3s} h={a['seat_height']:.3f} "
              f"xz={r['xz_err']:.3f} y={r['y_err']:+.3f} "
              f"pen={r['pen_frames']:3d} knee={r['knee_min_d']:.3f} "
              f"{a['object']}")

    with open(E2 / "e2_results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # ---- per-arm summary (median unless noted) ----
    lines = ["| 臂 | n | xz误差 | y误差(带符号中位) | \\|y误差\\|<5cm | 穿透帧(均值) | 零穿透率 | xz<10cm |",
             "|---|---|---|---|---|---|---|---|"]
    for tag, label in (("noh", "1 root-(x,z,yaw)"), ("h", "1b +root_y"),
                       ("fb", "2 fullbody 自举")):
        g = [r for r in rows if r["arm"] == tag]
        if not g:
            continue
        ye = np.array([r["y_err"] for r in g])
        pen = np.array([r["pen_frames"] for r in g])
        xz = np.array([r["xz_err"] for r in g])
        lines.append(
            f"| {label} | {len(g)} | {np.median(xz):.3f} | "
            f"{np.median(ye):+.3f} | {100*(np.abs(ye)<0.05).mean():.0f}% | "
            f"{pen.mean():.1f} | {100*(pen==0).mean():.0f}% | "
            f"{100*(xz<0.10).mean():.0f}% |")
    # seat-height terciles for the height story
    hs = sorted(set(r["seat_h"] for r in rows))
    if len(hs) >= 3:
        t1, t2 = np.quantile(hs, [1 / 3, 2 / 3])
        lines.append("")
        lines.append("| 臂 × 座高 | 低(≤%.2f) y误差 | 中 y误差 | 高(>%.2f) y误差 |" % (t1, t2))
        lines.append("|---|---|---|---|")
        for tag in ("noh", "h", "fb"):
            g = [r for r in rows if r["arm"] == tag]
            cells = []
            for lo, hi in ((0, t1), (t1, t2), (t2, 9)):
                ys = [r["y_err"] for r in g if lo < r["seat_h"] <= hi]
                cells.append(f"{np.median(ys):+.3f}" if ys else "-")
            lines.append(f"| {tag} | " + " | ".join(cells) + " |")

    summary = "\n".join(lines)
    print("\n" + summary)
    (E2 / "E2_summary.md").write_text(
        "# E2 批量三臂结果\n\n" + summary + "\n")
    print(f"\nsaved {E2/'e2_results.jsonl'} and {E2/'E2_summary.md'}")


if __name__ == "__main__":
    main()
