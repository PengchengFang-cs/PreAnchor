"""In-process Kimodo generation with full root-only anchor = (x, y, z, yaw).

Adds a custom Root3DConstraintSet (= Kimodo's root2d + the independent
`root_y_pos` conditioning channel) so height is constrained WITHOUT any
full-body keyframe -- true root-only, all four anchor DoF.

Height calibration: TRUMANS anchors store SMPL-X `transl`; Kimodo's root is
the pelvis joint.  TRANSL_TO_PELVIS = 0.26 m (standing transl ~1.13 in
TRUMANS vs standing pelvis 0.872 in Kimodo gen).  Refine with body model later.

Usage: python gen_kimodo2.py [sit_index] [--no-height]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
FPS = 30
SEAT_TO_PELVIS = 0.17    # pelvis joint above support surface when seated
TRANSL_TO_PELVIS = 0.26  # legacy; only used for reporting

from kimodo import load_model
from kimodo.constraints import Root2DConstraintSet, FullBodyConstraintSet

class Root3DConstraintSet(Root2DConstraintSet):
    """root2d (x,z + heading) + root height via the root_y_pos channel."""
    name = "root3d"

    def __init__(self, skeleton, frame_indices, smooth_root_2d, root_y,
                 global_root_heading=None):
        super().__init__(skeleton, frame_indices, smooth_root_2d,
                         global_root_heading=global_root_heading)
        self.root_y = root_y

    def update_constraints(self, data_dict, index_dict):
        super().update_constraints(data_dict, index_dict)
        data_dict["root_y_pos"].append(self.root_y)
        index_dict["root_y_pos"].append(self.frame_indices)

    def to(self, device=None, dtype=None):
        super().to(device, dtype)
        self.root_y = self.root_y.to(device=device) if device is not None else self.root_y
        if dtype is not None:
            self.root_y = self.root_y.to(dtype=dtype)
        return self


def build_constraints(a, T, skeleton, with_height=True, boot_path=None):
    ax, ay, az, ayaw = a["world"]
    fwd = np.array([np.sin(ayaw), np.cos(ayaw)])
    start_xz = np.array([ax, az]) + 2.5 * fwd
    walk_yaw = float(np.arctan2(-fwd[0], -fwd[1]))

    def heading(yaw, n):
        return torch.tensor([[np.cos(yaw), np.sin(yaw)]] * n, dtype=torch.float32)

    cons = []
    # start + waypoints (ground plane only)
    fr, pts, hd = [0, 1], [start_xz, start_xz], [walk_yaw, walk_yaw]
    for frac, tfrac in ((0.4, 0.35), (0.75, 0.6)):
        fr.append(int(tfrac * T))
        pts.append(start_xz + frac * (np.array([ax, az]) - start_xz))
        hd.append(walk_yaw)
    cons.append(Root2DConstraintSet(
        skeleton, torch.tensor(fr),
        torch.tensor(np.array(pts), dtype=torch.float32),
        global_root_heading=torch.stack(
            [heading(y, 1)[0] for y in hd]),
    ))
    # terminal anchor (last `yframes` frames, seated still)
    yframes = 24
    tf = torch.arange(T - yframes, T)
    txz = torch.tensor([[ax, az]] * yframes, dtype=torch.float32)
    thead = heading(ayaw, yframes)
    if with_height is True:
        # height from OBJECT GEOMETRY, not cross-dataset transl transfer:
        # pelvis-joint sits ~SEAT_TO_PELVIS above the support surface
        # (calibrated on Kimodo's own body: default sit 0.626 - 0.45 chair).
        ty = torch.full((yframes,), float(a["seat_height"]) + SEAT_TO_PELVIS)
        cons.append(Root3DConstraintSet(skeleton, tf, txz, ty,
                                        global_root_heading=thead))
    elif with_height == "fullbody":
        # arm-2 self-bootstrap: previous generation's seated end pose,
        # raised so the pelvis lands at seat_h + SEAT_TO_PELVIS
        prev = np.load(boot_path or OUT / "gen_sit_0_h.npz")
        pj = torch.tensor(prev["posed_joints"][-1], dtype=torch.float32)
        rj = torch.tensor(prev["global_rot_mats"][-1], dtype=torch.float32)
        dy = (a["seat_height"] + SEAT_TO_PELVIS) - float(prev["root_positions"][-1, 1])
        pj[:, 1] += dy
        n = 6
        cons.append(FullBodyConstraintSet(
            skeleton, torch.arange(T - n, T),
            pj[None].expand(n, -1, -1).clone(),
            rj[None].expand(n, -1, -1, -1).clone()))
        print(f"fullbody keyframe: raised prev end pose by dy={dy:+.3f} m")
    else:
        cons.append(Root2DConstraintSet(skeleton, tf, txz,
                                        global_root_heading=thead))
    return cons


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if "--fullbody" in sys.argv:
        with_height = "fullbody"
    else:
        with_height = "--no-height" not in sys.argv
    T = 240

    anchors = [json.loads(l) for l in open(OUT / "anchors.jsonl")]
    a = [x for x in anchors if x["verb"] == "sit"][idx]
    print(f"anchor #{idx}: {a['object']} world_y={a['world'][1]:.3f} "
          f"seat_h={a['seat_height']:.3f} with_height={with_height}")

    model = load_model("kimodo-smplx-rp", device="cuda")
    cons = build_constraints(a, T, model.skeleton, with_height)
    # match from_dict semantics: constraint DATA on model device, frame indices on cpu
    for c in cons:
        for attr in ("smooth_root_2d", "global_root_heading", "root_y",
                     "root_y_pos", "global_joints_positions", "global_joints_rots"):
            v = getattr(c, attr, None)
            if v is not None:
                setattr(c, attr, v.cuda())
    torch.manual_seed(1)
    out = model(["walk forward and sit down"], [T], constraint_lst=cons,
                num_denoising_steps=100, num_samples=1, multi_prompt=True,
                post_processing=True, return_numpy=True,
                cfg_type="separated", cfg_weight=(2.0, 5.0))

    single = {k: (v[0] if hasattr(v, "shape") and len(v.shape) > 0 else v)
              for k, v in out.items()}
    tag = {True: "h", False: "noh", "fullbody": "fb"}[with_height]
    path = OUT / f"gen_sit_{idx}_{tag}.npz"
    np.savez(path, **single)

    root = single["root_positions"]
    tgt_y = a["seat_height"] + SEAT_TO_PELVIS
    print(f"saved {path}")
    print(f"terminal: (x,z)=({root[-1,0]:.3f},{root[-1,2]:.3f}) "
          f"target=({a['world'][0]:.3f},{a['world'][2]:.3f})")
    print(f"root y end={root[-1,1]:.3f}  target={tgt_y:.3f}  "
          f"err={abs(root[-1,1]-tgt_y):.3f}")


if __name__ == "__main__":
    main()
