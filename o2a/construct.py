"""M2 construct(): analytic sit-anchor candidates from object geometry alone.

Pipeline (S-class, sit): parse up-facing seat patches -> sample (x,z) seat
points -> yaw grid + hard constraints (front edge reachable, shin box free,
torso box free) -> rank by soft score (backrest, edge distance, height prior).
Height channel: pelvis_y = patch_height + SEAT_TO_PELVIS (E1/E2-validated).

All geometry is face-center point clouds (Shard convention). `scene` faces
are used for clearance tests so tucked-in / blocked spots are rejected.
"""
import numpy as np
from scipy.spatial import cKDTree

SEAT_TO_PELVIS = 0.17
UP_NY = 0.80                 # up-facing face normal threshold
SEAT_BAND = (0.20, 1.00)     # sittable surface height range (m, floor=0)
LEVEL_TOL = 0.05             # faces within one seat level
GRID = 0.12                  # xz connected-component cell size
MIN_AREA = 0.06              # m^2, minimum seat patch area
POINT_STEP = 0.30            # interior sampling step for large patches
SMALL_PATCH = 0.55           # patch extent below which centroid is the seat
N_YAW = 12
EDGE_RANGE = (0.03, 0.45)    # front edge distance from pelvis point
EDGE_BEST = 0.15
CORRIDOR = 0.15              # half-width of the edge-probing corridor


def _patches(c, n, a):
    """up-facing faces -> list of (height, pts, area) seat patches."""
    up = (n[:, 1] > UP_NY) & (c[:, 1] > SEAT_BAND[0]) & (c[:, 1] < SEAT_BAND[1])
    c, a = c[up], a[up]
    if len(c) == 0:
        return []
    out = []
    order = np.argsort(c[:, 1])
    c, a = c[order], a[order]
    lo = 0
    for hi in range(1, len(c) + 1):  # greedy height levels
        if hi == len(c) or c[hi, 1] - c[hi - 1, 1] > LEVEL_TOL:
            lvl_c, lvl_a = c[lo:hi], a[lo:hi]
            lo = hi
            # xz connected components on a grid
            cell = np.floor(lvl_c[:, [0, 2]] / GRID).astype(int)
            key = {}
            parent = list(range(len(cell)))

            def find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            for i, (cx, cz) in enumerate(cell):
                for dx in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        j = key.get((cx + dx, cz + dz))
                        if j is not None:
                            pi, pj = find(i), find(j)
                            if pi != pj:
                                parent[pi] = pj
                key[(cx, cz)] = i
            comp = {}
            for i in range(len(cell)):
                comp.setdefault(find(i), []).append(i)
            for idx in comp.values():
                idx = np.array(idx)
                area = float(lvl_a[idx].sum())
                if area >= MIN_AREA:
                    pts = lvl_c[idx]
                    out.append((float(np.median(pts[:, 1])), pts, area))
    return out


def _seat_points(pts):
    """candidate pelvis (x,z) positions on a patch."""
    xz = pts[:, [0, 2]]
    ext = xz.max(0) - xz.min(0)
    if max(ext) < SMALL_PATCH:
        return [xz.mean(0)]
    out = []
    tree = cKDTree(xz)
    for gx in np.arange(xz[:, 0].min(), xz[:, 0].max() + 1e-6, POINT_STEP):
        for gz in np.arange(xz[:, 1].min(), xz[:, 1].max() + 1e-6, POINT_STEP):
            if len(tree.query_ball_point([gx, gz], 0.12)) >= 3:
                out.append(np.array([gx, gz]))
    return out or [xz.mean(0)]


def _box_occupied(scene_xz_tree, scene, p, fwd, right, f0, f1, lat, y0, y1):
    """any scene face-center inside the oriented box? (probe via KDTree)"""
    center = p + fwd * (f0 + f1) / 2
    rad = np.hypot((f1 - f0) / 2, lat) + 0.01
    idx = scene_xz_tree.query_ball_point(center, rad)
    if not idx:
        return False
    q = scene[idx]
    d = q[:, [0, 2]] - p
    fo, la = d @ fwd, d @ right
    return bool(np.any((fo > f0) & (fo < f1) & (np.abs(la) < lat) &
                       (q[:, 1] > y0) & (q[:, 1] < y1)))


def construct_sit(obj_c, obj_n, obj_a, scene_c, scene_other=None):
    """-> list of dicts {x, z, yaw, y, seat_h, score}, ranked best-first.

    scene_other: scene faces EXCLUDING this object -- shins brush the
    object's own legs/stretchers when seated, so only foreign geometry
    blocks the shin box.  Defaults to scene_c (conservative)."""
    if scene_other is None:
        scene_other = scene_c
    tree = cKDTree(scene_c[:, [0, 2]])
    tree_o = cKDTree(scene_other[:, [0, 2]])
    cands = []
    for seat_h, pts, area in _patches(obj_c, obj_n, obj_a):
        xz = pts[:, [0, 2]]
        for p in _seat_points(pts):
            d = xz - p
            point_cands = []
            for yaw in np.arange(0, 2 * np.pi, 2 * np.pi / N_YAW):
                fwd = np.array([np.sin(yaw), np.cos(yaw)])
                right = np.array([np.cos(yaw), -np.sin(yaw)])
                fo, la = d @ fwd, d @ right
                ahead = fo[(np.abs(la) < CORRIDOR) & (fo > 0)]
                d_edge = float(ahead.max()) if len(ahead) else 0.0
                if not EDGE_RANGE[0] < d_edge < EDGE_RANGE[1]:
                    continue
                # feet strip past the front edge: foreign geometry reaching
                # the floor (walls, cabinets) blocks; table tops/aprons and
                # the object's own legs/stretchers do not
                if _box_occupied(tree_o, scene_other, p, fwd, right,
                                 d_edge + 0.05, d_edge + 0.35, 0.15,
                                 0.10, min(0.30, seat_h - 0.05)):
                    continue
                # torso box above the seat, front half free
                if _box_occupied(tree, scene_c, p, fwd, right,
                                 -0.05, 0.25, 0.25,
                                 seat_h + 0.25, seat_h + 0.95):
                    continue
                back = _box_occupied(tree, scene_c, p, fwd, right,
                                     -0.45, -0.10, 0.25,
                                     seat_h + 0.15, seat_h + 0.60)
                open_front = not _box_occupied(tree, scene_c, p, fwd, right,
                                               d_edge + 0.10, d_edge + 0.90,
                                               0.25, 0.30, 1.50)
                score = (1.0 * back
                         + 0.15 * open_front
                         - abs(d_edge - EDGE_BEST)
                         + 0.2 * min(area, 1.0)
                         - 0.8 * abs(seat_h - 0.45))
                point_cands.append({"x": float(p[0]), "z": float(p[1]),
                                    "yaw": float((yaw + np.pi) % (2 * np.pi)
                                                 - np.pi),
                                    "y": seat_h + SEAT_TO_PELVIS,
                                    "seat_h": seat_h, "back": back,
                                    "score": float(score)})
            # a seat with a back is sat facing away from it: when any yaw
            # at this point has a backrest, demote the yaws that don't
            if any(c["back"] for c in point_cands):
                for c in point_cands:
                    if not c["back"]:
                        c["score"] -= 0.8
            cands.extend(point_cands)
    # dedupe near-identical candidates, keep best score
    cands.sort(key=lambda r: -r["score"])
    kept = []
    for r in cands:
        dup = any(np.hypot(r["x"] - k["x"], r["z"] - k["z"]) < 0.25 and
                  abs((r["yaw"] - k["yaw"] + np.pi) % (2 * np.pi) - np.pi)
                  < np.radians(30) for k in kept)
        if not dup:
            kept.append(r)
    return kept
