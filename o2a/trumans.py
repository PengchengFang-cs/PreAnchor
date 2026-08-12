"""TRUMANS (zirobtc mirror) shard reader.

Layout per shard tar: humans.npz, scene.npz, objects/*.npz, objects.poses.npz,
samples.npz, text.json.  World is Y-up; SMPL-X params; 30 fps.
"""
import io
import json
import tarfile
from pathlib import Path

import numpy as np

FPS = 30.0
UP = 1  # Y-up


def _npz(tf, name):
    return np.load(io.BytesIO(tf.extractfile(name).read()), allow_pickle=True)


class Shard:
    def __init__(self, tar_path):
        self.path = Path(tar_path)
        with tarfile.open(tar_path) as tf:
            names = tf.getnames()
            self.humans = _npz(tf, "humans.npz")
            self.text = json.loads(tf.extractfile("text.json").read())
            scene = _npz(tf, "scene.npz")
            self.obj_poses = _npz(tf, "objects.poses.npz")
            self.movable = {}
            for n in names:
                if n.startswith("objects/") and n.endswith(".npz"):
                    self.movable[Path(n).stem] = _npz(tf, n)

        # --- static scene: per-face centers / normals / object ids ---
        v, f = scene["vertices"], scene["faces"]
        self.scene_names = list(scene["object_names"])
        fo = scene["object_face_offsets"]
        self.face_obj = np.zeros(len(f), dtype=np.int32)
        for i in range(len(self.scene_names)):
            self.face_obj[fo[i]:fo[i + 1]] = i
        tri = v[f]                                     # [F,3,3]
        self.face_c = tri.mean(1)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area2 = np.linalg.norm(n, axis=1)
        self.face_n = n / np.clip(area2, 1e-9, None)[:, None]
        self.face_a = 0.5 * area2

    # ------------------------------------------------------------------
    def samples(self):
        h = self.humans
        for i, sid in enumerate(h["sample_id"]):
            lo, hi = int(h["sample_start"][i]), int(h["sample_end"][i])
            yield {
                "sample_id": int(sid),
                "rows": np.arange(lo, hi),
                "frame_id": h["frame_id"][lo:hi],
                "transl": h["transl"][lo:hi],
                "global_orient": h["global_orient"][lo:hi],
                "betas": h["betas"][lo:hi],
            }

    def segments(self, sample_id):
        for s in self.text["samples"]:
            if s["sample_id"] == sample_id:
                return [seg for seg in s["segments"] if seg.get("human", 0) == 0]
        return []

    # ------------------------------------------------------------------
    def movable_faces_at(self, sample_id, frame):
        """face centers/normals/areas/name for movable objects posed at frame."""
        p = self.obj_poses
        sel = np.where(p["sample_id"] == sample_id)[0]
        if len(sel) == 0:
            return []
        i = sel[0]
        lo, hi = int(p["sample_start"][i]), int(p["sample_end"][i])
        frames, oidx = p["frames"][lo:hi], p["object_index"][lo:hi]
        tfs = p["transforms"][lo:hi]
        out = []
        for k, name in enumerate(p["object_names"]):
            if name not in self.movable:
                continue
            rows = np.where(oidx == k)[0]
            if len(rows) == 0:
                continue
            r = rows[np.argmin(np.abs(frames[rows] - frame))]
            T = tfs[r]
            m = self.movable[name]
            v, f = m["vertices"], m["faces"]
            step = max(1, len(f) // 4000)              # decimate for speed
            f = f[::step]
            tri = v[f] @ T[:3, :3].T + T[:3, 3]
            c = tri.mean(1)
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            a2 = np.linalg.norm(n, axis=1)
            out.append((c, n / np.clip(a2, 1e-9, None)[:, None],
                        0.5 * a2 * step, str(name)))
        return out
