"""Batch E2: ~24 sit anchors x 3 arms (noh / h / fb) through frozen Kimodo.

Anchor selection (deterministic): one representative anchor per object
(median seat height within the object's group), objects quantile-spaced
across the seat-height range, E1's anchor (sit #0, static_chair_03)
force-included so E1 stays a strict subset.

Arms per anchor (E1 protocol, seed 1, cfg (2.0, 5.0), 100 steps, T=240):
  noh  root-(x,z,yaw) only
  h    + root_y channel on last 24 frames
  fb   fullbody terminal keyframe, self-bootstrapped from THIS anchor's h run

Resumable: existing out/e2/gen_s{sit_idx}_{tag}.npz are skipped.

Usage: python batch_e2.py [--n 24] [--limit K] [--dry]
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
E2 = OUT / "e2"

from gen_kimodo2 import build_constraints, SEAT_TO_PELVIS

E1_OBJECT = "movable:static_chair_03"  # its representative is forced to sit #0


def select_anchors(sit, n):
    groups = {}
    for i, a in enumerate(sit):
        o = a["object"]
        # movable handheld items (pen/book/cup) as sit supports are M1
        # binding errors -- only chair-like movables are valid targets
        if o.startswith("movable:") and "chair" not in o:
            continue
        groups.setdefault(o, []).append((i, a))
    reps = {}
    for obj, g in groups.items():
        g = sorted(g, key=lambda t: (t[1]["seat_height"], t[1]["shard"], t[1]["frame"]))
        reps[obj] = g[len(g) // 2]
    reps[E1_OBJECT] = (0, sit[0])

    objs = sorted(reps, key=lambda o: reps[o][1]["seat_height"])
    pick = sorted(set(np.linspace(0, len(objs) - 1, n).round().astype(int)))
    chosen = [objs[i] for i in pick]
    if E1_OBJECT not in chosen:
        chosen.append(E1_OBJECT)
    sel = sorted((reps[o] for o in chosen), key=lambda t: t[1]["seat_height"])
    return sel


def to_cuda(cons):
    for c in cons:
        for attr in ("smooth_root_2d", "global_root_heading", "root_y",
                     "root_y_pos", "global_joints_positions", "global_joints_rots"):
            v = getattr(c, attr, None)
            if v is not None:
                setattr(c, attr, v.cuda())


def generate(model, a, T, with_height, boot_path=None):
    cons = build_constraints(a, T, model.skeleton, with_height, boot_path)
    to_cuda(cons)
    torch.manual_seed(1)
    out = model(["walk forward and sit down"], [T], constraint_lst=cons,
                num_denoising_steps=100, num_samples=1, multi_prompt=True,
                post_processing=True, return_numpy=True,
                cfg_type="separated", cfg_weight=(2.0, 5.0))
    return {k: (v[0] if hasattr(v, "shape") and len(v.shape) > 0 else v)
            for k, v in out.items()}


def main():
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 24
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    T = 240

    anchors = [json.loads(l) for l in open(OUT / "anchors.jsonl")]
    sit = [x for x in anchors if x["verb"] == "sit"]
    sel = select_anchors(sit, n)
    if limit:
        # keep E1's anchor first under --limit so the smoke test hits it
        sel = sorted(sel, key=lambda t: t[0] != 0)[:limit]

    E2.mkdir(exist_ok=True)
    manifest = [{"sit_idx": i, **{k: a[k] for k in
                 ("object", "seat_height", "shard", "sample_id", "frame", "world")}}
                for i, a in sel]
    (E2 / "e2_anchors.json").write_text(json.dumps(manifest, indent=1))
    print(f"selected {len(sel)} anchors, seat_h "
          f"{sel[0][1]['seat_height']:.3f}..{sel[-1][1]['seat_height']:.3f}")
    for i, a in sel:
        print(f"  s{i:<4d} h={a['seat_height']:.3f}  {a['object']}")
    if "--dry" in sys.argv:
        return

    from kimodo import load_model
    model = load_model("kimodo-smplx-rp", device="cuda")

    for i, a in sel:
        h_path = E2 / f"gen_s{i}_h.npz"
        for tag, wh in (("noh", False), ("h", True), ("fb", "fullbody")):
            path = E2 / f"gen_s{i}_{tag}.npz"
            if path.exists():
                print(f"[skip] {path.name}")
                continue
            if tag == "fb" and not h_path.exists():
                print(f"[warn] {h_path.name} missing, skip fb for s{i}")
                continue
            t0 = time.time()
            try:
                single = generate(model, a, T, wh,
                                  boot_path=h_path if tag == "fb" else None)
            except Exception:
                print(f"[fail] s{i} {tag}")
                traceback.print_exc()
                continue
            np.savez(path, **single)
            root = single["root_positions"]
            tgt_y = a["seat_height"] + SEAT_TO_PELVIS
            xz_err = float(np.linalg.norm(
                root[-1, [0, 2]] - np.array(a["world"])[[0, 2]]))
            print(f"[done] s{i} {tag:3s} {time.time()-t0:5.1f}s  "
                  f"xz_err={xz_err:.3f}  y_end={root[-1,1]:.3f} tgt={tgt_y:.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
