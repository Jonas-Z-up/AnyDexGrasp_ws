#!/usr/bin/env python3
"""接触组搜索 — 为每个抓取类型枚举 (fa, fb) 指节组合, 选与 Inspire 查表约定
面外倾斜最小的组合, 锁定 TYPE_CONTACT_GROUPS。

对每类型: 在该类型闭合位姿下一次性算出全部指节表面质心, 枚举 fa≠fb 组合,
按 angles2stl 公式构造 (C, n, u) → R_own, 与 R_lk 比较 (3 宽度平均):
  面外倾斜小 = 构造与上游一致。拇指固定用 thumb_2_r。

用法: python own_hand/tools/search_contact_groups.py [--types 5 6 7 8]
"""

import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(WS_ROOT, "own_hand"))          # own_hand_grasp
sys.path.insert(0, os.path.join(WS_ROOT, "scripts", "tools"))  # compute_hand_geometry

import own_hand_grasp as ohg          # noqa: E402
import compute_hand_geometry as chg   # noqa: E402

FINGER_LINKS = ["index_1_r", "index_2_r", "middle_1_r", "middle_2_r",
                "ring_1_r", "ring_2_r", "little_1_r", "little_2_r"]
PALM_LINKS = ["base_link", "thumb_1_r"]
THUMB = "thumb_2_r"

LOOKUP_JSON = os.path.join(WS_ROOT, "assets", "inspire_urdf",
                           "width_12Dangle_6Dangle.json")
TYPE_NAMES = {1: "Ring", 2: "Prismatic_2_Finger", 3: "Prismatic_3_Finger",
              4: "Large_Diameter", 5: "Medium_Wrap", 6: "Tripod",
              7: "Sphere_3_Finger", 8: "Distal_Type"}
WIDTHS_CM = ["4.0", "6.0", "8.0"]


def link_centroids(st, links, samples=2500):
    import open3d as o3d
    fr = chg.link_frames(st["p"], st["hand"])
    out = {}
    for ln in links:
        mesh = o3d.io.read_triangle_mesh(os.path.join(chg.MESH_DIR, ln + ".STL"))
        pc = mesh.sample_points_uniformly(number_of_points=samples)
        pts = np.asarray(pc.points)
        R, t = fr[ln][1], fr[ln][0]
        out[ln] = (pts @ R.T + t).mean(axis=0)
    return out


def tilt_for(R_lk_list, p0, p1, p2, c_palm):
    v1, v2 = p1 - p0, p2 - p0
    n = np.cross(v1, v2)
    nn = np.linalg.norm(n)
    if nn < 1e-6:                       # 三点近共线, 退化
        return None
    n = n / nn
    C = 0.5 * (p0 + 0.5 * (p1 + p2))
    u = p0 - C
    un = np.linalg.norm(u)
    if un < 1e-6:
        return None
    u = u / un
    if np.dot(n, c_palm - C) < 0:       # 掌体在杯口负侧
        n = -n
    z = np.cross(n, u)
    zn = np.linalg.norm(z)
    if zn < 1e-6:
        return None
    R_own = np.column_stack([n, u, z / zn])
    tilts = []
    for R_lk in R_lk_list:
        R_rel = R_own @ R_lk.T
        rv = Rotation_rel_axis(R_rel)
        tilts.append(np.degrees(np.arctan2(np.linalg.norm(rv[1:]), abs(rv[0]))))
    return float(np.mean(tilts))


def Rotation_rel_axis(R_rel):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R_rel).as_rotvec()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", type=int, nargs="+", default=list(TYPE_NAMES))
    args = ap.parse_args()

    with open(LOOKUP_JSON) as f:
        lookup = json.load(f)
    st = ohg._init()

    for tid in args.types:
        name = TYPE_NAMES[tid]
        R_lks = []
        for wcm in WIDTHS_CM:
            entry = lookup.get(name, {}).get(wcm)
            if entry:
                R_lks.append(np.array(entry["rotation"], dtype=float))
        motors = chg.TYPE_CLOSE_MOTORS[tid]["motors"]
        chg.set_motors(st["p"], st["hand"], st["jidx"], motors)
        cent = link_centroids(st, [THUMB] + FINGER_LINKS + PALM_LINKS)
        p0 = cent[THUMB]
        c_palm = np.mean([cent[ln] for ln in PALM_LINKS], axis=0)

        results = []
        for i, fa in enumerate(FINGER_LINKS):
            for fb in FINGER_LINKS[i + 1:]:
                tilt = tilt_for(R_lks, p0, cent[fa], cent[fb], c_palm)
                if tilt is not None:
                    results.append((tilt, fa, fb))
        results.sort()
        print(f"\nType {tid} {name} — 最优5组:")
        for tilt, fa, fb in results[:5]:
            print(f"  tilt={tilt:5.1f}°  fa={fa:<11} fb={fb}")
        print(f"  最差1组: tilt={results[-1][0]:.1f}°  "
              f"fa={results[-1][1]} fb={results[-1][2]}")


if __name__ == "__main__":
    main()
