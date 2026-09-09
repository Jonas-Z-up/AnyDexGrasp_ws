#!/usr/bin/env python3
"""Ring 手势锚定渲染 — 验证对用户描述的理解。

用户描述: 采用"Ring — 拇指+食指+中指形成环"(hand_bridge Type 2), 用
食指与中指之间的缝隙和大拇指对应蓝色两指线框实现抓取。

锚定构造 (同一三角形公式, 接触组换为环开口三点):
  p0 = 拇指接触质心 (thumb_2_r)
  p1 = 食指远节质心 (index_2_r)   ┐ 环的另一侧 = 食/中指缝隙两缘
  p2 = 中指远节质心 (middle_2_r)  ┘
  C  = (p0 + (p1+p2)/2)/2  = 环心 (物体中心目标位)
  接近轴 = 三角形法向, 开合轴 = 拇指方向。

场景 = 真实预览输出 (Type 3 抓取点/朝向, width), 圆柱示意香蕉。
电机方案 A: hand_bridge Type 2 动作库值 (实机调好)
电机方案 B: Inspire "Ring" 6d 宽度化 2× 映射
输出: own_hand/renders/hand/ring_compare_{handbridge,6d}.png (4 视角)
"""

import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(WS_ROOT, "own_hand"))          # own_hand_grasp
sys.path.insert(0, os.path.join(WS_ROOT, "scripts", "tools"))  # compute_hand_geometry

import own_hand_grasp as ohg   # noqa: E402
import compute_hand_geometry as chg  # noqa: E402

OUT_DIR = os.path.join(WS_ROOT, "own_hand", "renders", "hand")
GRASP_JSON = os.path.join(WS_ROOT, "logs", "preview", "latest_inspire_grasp.json")
LOOKUP_JSON = os.path.join(WS_ROOT, "assets", "inspire_urdf",
                           "width_12Dangle_6Dangle.json")
TYPE_NAMES = {1: "Ring", 2: "Prismatic_2_Finger", 3: "Prismatic_3_Finger",
              4: "Large_Diameter", 5: "Medium_Wrap", 6: "Tripod",
              7: "Sphere_3_Finger", 8: "Distal_Type"}

# Ring 手势的环开口三点 (拇指 | 食/中指缝隙两缘)
RING_CONTACT = {"p0": ["thumb_2_r"], "p1": ["index_2_r"], "p2": ["middle_2_r"]}


def ring_anchor(motors, t_g, R_g):
    """Ring 手势锚定: 环心对准抓取点, 三角形法向=接近轴, 拇指方向=开合轴。"""
    st = ohg._init()
    chg.set_motors(st["p"], st["hand"], st["jidx"], motors)

    def cen(links, n):
        return chg.link_surface_points(st["p"], st["hand"], links, samples=n).mean(axis=0)

    p0 = cen(RING_CONTACT["p0"], 2000)
    p1 = cen(RING_CONTACT["p1"], 1500)
    p2 = cen(RING_CONTACT["p2"], 1500)
    C = 0.5 * (p0 + 0.5 * (p1 + p2))
    u = p0 - C
    u = u / np.linalg.norm(u)
    n = np.cross(p1 - p0, p2 - p0)
    n = n / np.linalg.norm(n)
    if np.dot(C, n) < 0:      # 环心须在 base 原点 +n 侧 (2026-09-08 修正, 同 own_hand_grasp)
        n = -n
    z = np.cross(n, u)
    z = z / np.linalg.norm(z)
    R_own = np.column_stack([n, u, z])
    R_wb = R_g @ R_own.T
    t_wb = t_g - R_wb @ C
    return t_wb, R_wb


def width_motors_ring(width):
    """方案 B: Inspire "Ring" 6d → our 0-2000 (2×映射, 1000=未用→2000)。"""
    with open(LOOKUP_JSON) as f:
        lk = json.load(f)
    entries = lk["Ring"]
    wcm = float(width) * 100.0
    key = min(entries.keys(), key=lambda k: abs(float(k) - wcm))
    six = entries[key]["6d"]
    print(f"  Inspire Ring @{key}cm 6d = {six}")

    def m(v):
        v = float(v)
        return 2000 if v >= 1000 else int(round(2.0 * v))

    return {5: m(six[0]), 4: m(six[1]), 3: m(six[2]), 2: m(six[3]),
            1: m(six[4]), 6: m(six[5])}


def build_hand_mesh(motors, T_base):
    import open3d as o3d
    st = ohg._init()
    chg.set_motors(st["p"], st["hand"], st["jidx"], motors)
    frames = chg.link_frames(st["p"], st["hand"])
    mesh = o3d.geometry.TriangleMesh()
    for ln in ohg.ALL_LINKS:
        m = o3d.io.read_triangle_mesh(os.path.join(chg.MESH_DIR, ln + ".STL"))
        R_l, t_l = frames[ln][1], frames[ln][0]
        T_l = np.vstack([np.hstack([R_l, t_l.reshape(3, 1)]), [0, 0, 0, 1]])
        m.transform(T_l)
        mesh += m
    mesh.transform(T_base)
    return mesh


def main():
    import open3d as o3d

    with open(GRASP_JSON) as f:
        r = json.load(f)
    width = float(r.get("two_finger_width", r["width"]))
    t_g = np.array(r["two_finger_translation"])
    R_g = np.array(r["two_finger_rotation_matrix"]).reshape(3, 3)
    print(f"场景: width={width:.3f} 抓取点={np.round(t_g*1000,0)} (相机系)")

    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=0.03, height=0.16)
    zax = R_g[:, 2]
    ax = np.array([0.0, 0.0, 1.0])
    v = np.cross(ax, zax)
    c = float(np.dot(ax, zax))
    if abs(c) < 0.999:
        v = v / np.linalg.norm(v)
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        Rz = np.eye(3) + K + K @ K * (1 / (1 + c))
        cyl.rotate(Rz, center=(0, 0, 0))
    cyl.translate(t_g)
    cyl.paint_uniform_color([0.25, 0.55, 0.9])

    arr = np.zeros(17)
    arr[0], arr[1], arr[2], arr[3] = 1.0, width, 0.03, 0.02
    arr[4:13] = R_g.reshape(-1)
    arr[13:16] = t_g
    arr[16] = -1
    from graspnetAPI import Grasp
    gripper = Grasp(arr).to_open3d_geometry()
    gripper.paint_uniform_color([0.1, 0.4, 1.0])
    gripper_pts = np.asarray(gripper.sample_points_uniformly(8000).points)

    ohg._init()
    variants = {
        "handbridge": dict(ohg.chg.TYPE_CLOSE_MOTORS[2]["motors"]),  # Ring=Type2 动作库值
        "6d": width_motors_ring(width),
    }

    views = [("相机视角", -88, -90), ("侧视", 15, 110),
             ("俯视", -35, -90), ("正视", 5, 0)]

    for tag, motors in variants.items():
        t_b, R_b = ring_anchor(motors, t_g, R_g)
        T = np.vstack([np.hstack([R_b, t_b.reshape(3, 1)]), [0, 0, 0, 1]])
        mesh = build_hand_mesh(motors, T)
        print(f"方案[{tag}] 电机={motors}")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(15, 13))
        for k, (title, elev, azim) in enumerate(views):
            ax = fig.add_subplot(2, 2, k + 1, projection="3d")
            pts = np.asarray(mesh.sample_points_uniformly(12000).points)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.2, c="red", alpha=0.5)
            cpts = np.asarray(cyl.sample_points_uniformly(5000).points)
            ax.scatter(cpts[:, 0], cpts[:, 1], cpts[:, 2], s=1.2,
                       c="tab:blue", alpha=0.5)
            ax.scatter(gripper_pts[:, 0], gripper_pts[:, 1], gripper_pts[:, 2],
                       s=1.5, c="royalblue", alpha=0.9)
            for vec, col in zip([R_g[:, 0], R_g[:, 1], R_g[:, 2]],
                                ["r", "g", "b"]):
                seg = np.linspace(t_g, t_g + vec * 0.08, 10)
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], c=col, lw=2)
            ax.set_title(f"Ring[{tag}] — {title}")
            ax.set_xlabel("X_cam"); ax.set_ylabel("Y_cam"); ax.set_zlabel("Z_cam")
            ctr = t_g
            ax.set_xlim(ctr[0] - 0.22, ctr[0] + 0.22)
            ax.set_ylim(ctr[1] - 0.22, ctr[1] + 0.22)
            ax.set_zlim(ctr[2] - 0.22, ctr[2] + 0.22)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=elev, azim=azim)
        fp = os.path.join(OUT_DIR, f"ring_compare_{tag}.png")
        plt.tight_layout()
        plt.savefig(fp, dpi=110)
        plt.close(fig)
        print(f"[输出] {fp}")
    print("\n红色=自研手  蓝点=目标物体  深蓝=两指线框  RGB线=抓取系XYZ")


if __name__ == "__main__":
    main()
