"""S-category (support-transfer) predicate: detect() on TRUMANS motion.

Anchor convention (v1): "root" = SMPL-X `transl` (consistent proxy for pelvis;
exact pelvis joint needs body-model files and can be converted later).
World is Y-up. yaw = atan2(fwd_x, fwd_z) of body-forward (R @ +Z) on XZ plane.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

from .trumans import FPS

SUPPORT_NY = 0.80          # face normal y-component to count as "up-facing"
BIND_RADIUS = 0.45         # horizontal search radius around root (m)
BIND_BELOW = (0.05, 0.50)  # support surface must be this far below root (m)


def yaw_of(rotvec):
    fwd = R.from_rotvec(rotvec).as_matrix() @ np.array([0.0, 0.0, 1.0])
    return float(np.arctan2(fwd[0], fwd[2]))


def body_up_y(rotvec):
    return float((R.from_rotvec(rotvec).as_matrix() @ np.array([0.0, 1.0, 0.0]))[1])


def detect_commitment(seg, sample, verb, pad_after_s=1.0):
    """Return the commitment row (index into sample arrays) or None."""
    fids = sample["frame_id"]
    m = (fids >= seg["start"]) & (fids <= seg["end"] + int(pad_after_s * FPS))
    idx = np.where(m)[0]
    if len(idx) < 5:
        return None
    h = sample["transl"][idx, 1]
    vel = np.abs(np.gradient(h)) * FPS
    stable = vel < 0.25
    low = h < np.quantile(h, 0.2) + 0.05
    if verb == "lie":  # additionally require torso near-horizontal
        upy = np.array([body_up_y(sample["global_orient"][i]) for i in idx])
        stable = stable & (np.abs(upy) < 0.55)
    cand = np.where(stable & low)[0]
    if len(cand) == 0:
        cand = np.array([int(np.argmin(h))])
    return int(idx[cand[len(cand) // 2]])


def bind_support(shard, sample_id, frame, root):
    """Find the object whose up-facing surface supports `root`.

    Returns (object_key, support_face_centers, seat_height) or None.
    Candidates = static scene objects + movable objects posed at `frame`.
    """
    pools = [(shard.face_c, shard.face_n, shard.face_a, None)]
    pools += [(c, n, a, name) for c, n, a, name in
              shard.movable_faces_at(sample_id, frame)]

    best = None
    for c, n, a, movable_name in pools:
        up = n[:, 1] > SUPPORT_NY
        dxz = np.linalg.norm((c[:, [0, 2]] - root[None, [0, 2]]), axis=1)
        below = root[1] - c[:, 1]
        sel = up & (dxz < BIND_RADIUS) & (below > BIND_BELOW[0]) & (below < BIND_BELOW[1])
        if not sel.any():
            continue
        if movable_name is None:  # group static faces by object id
            for oid in np.unique(shard.face_obj[sel]):
                s = sel & (shard.face_obj == oid)
                area = a[s].sum()
                if best is None or area > best[0]:
                    best = (area, str(shard.scene_names[oid]), c[s])
        else:
            area = a[sel].sum()
            if best is None or area > best[0]:
                best = (area, "movable:" + movable_name, c[sel])
    if best is None:
        return None
    _, key, centers = best
    seat_h = float(np.median(centers[:, 1]))
    band = np.abs(centers[:, 1] - seat_h) < 0.10
    return key, centers[band], seat_h


def object_frame(support_centers):
    """Canonical frame of the support region: origin = centroid, axes = PCA
    on XZ (deterministic sign), y = world up.  Returns (origin, yaw0)."""
    xz = support_centers[:, [0, 2]]
    origin = support_centers.mean(0)
    c = xz - xz.mean(0)
    cov = c.T @ c / max(len(c), 1)
    w, v = np.linalg.eigh(cov)
    major = v[:, np.argmax(w)]              # [x, z] of major axis
    if major[0] < 0 or (abs(major[0]) < 1e-6 and major[1] < 0):
        major = -major
    yaw0 = float(np.arctan2(major[0], major[1]))
    return origin, yaw0


def to_object_frame(root, yaw, origin, yaw0):
    """Express world root/yaw in the object frame (yaw convention atan2(x,z):
    frame forward = (sin yaw0, 0, cos yaw0), right = (cos yaw0, 0, -sin yaw0))."""
    d = root - origin
    fwd = np.array([np.sin(yaw0), 0.0, np.cos(yaw0)])
    right = np.array([np.cos(yaw0), 0.0, -np.sin(yaw0)])
    local = np.array([float(d @ right), float(d[1]), float(d @ fwd)])
    yaw_local = float((yaw - yaw0 + np.pi) % (2 * np.pi) - np.pi)
    return local, yaw_local
