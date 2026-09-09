#!/usr/bin/env python3
"""锚定姿态对比渲染 — 当前代码 vs 提议的宽度化锚定, 供人工判定。

场景取自真实预览输出 logs/preview/latest_inspire_grasp.json (相机系):
两指抓取中心/朝向 + Type 3 + width, 目标用圆柱示意 (沿开合轴法向横放)。
每姿态两个视角: 相机视角(从相机原点看过去) + 侧视。

用法: python own_hand/tools/render_anchor_compare.py
输出: own_hand/renders/hand/anchor_compare_{current,proposed}.png
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
TYPE_NAMES = {1: "Ring", 2: "Prismatic_2_Finger", 3: "Prismatic_3_Finger",
              4: "Large_Diameter", 5: "Medium_Wrap", 6: "Tripod",
              7: "Sphere_3_Finger", 8: "Distal_Type"}


def build_hand_mesh(motors, T_base):
    """按电机向量摆手, 各 link mesh 变换到 T_base, 合并为一个 open3d mesh。"""
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


def anchor_from_motors(motors, grasp_type, t_g, R_g):
    """在给定摆位下按接触三角形公式构造锚定 (与 own_hand_grasp 同式)。"""
    st = ohg._init()
    chg.set_motors(st["p"], st["hand"], st["jidx"], motors)
    spec = ohg.TYPE_CONTACT_GROUPS[grasp_type]

    def cen(links, n):
        return chg.link_surface_points(st["p"], st["hand"], links, samples=n).mean(axis=0)

    p0 = cen(spec["thumb"], 2000)
    p1 = cen(spec["fa"], 1500)
    p2 = cen(spec["fb"], 1500)
    C = 0.5 * (p0 + 0.5 * (p1 + p2))
    u = p0 - C
    u = u / np.linalg.norm(u)
    n = np.cross(p1 - p0, p2 - p0)
    n = n / np.linalg.norm(n)
    c_palm = cen(["base_link"], 2000)
    if np.dot(n, c_palm - C) < 0:
        n = -n
    z = np.cross(n, u)
    z = z / np.linalg.norm(z)
    R_own = np.column_stack([n, u, z])
    R_wb = R_g @ R_own.T
    t_wb = t_g - R_wb @ C
    return t_wb, R_wb


def width_motors_local(grasp_type, width):
    """提议的宽度化摆位: Inspire 6d 表 → our 0-2000 制 (2×映射, 1000=未用→2000)。

    6d 顺序 [小, 无名, 中, 食, 拇屈, 拇旋]; 宽度取查表最近档。
    """
    lookup_json = os.path.join(WS_ROOT, "assets", "inspire_urdf",
                               "width_12Dangle_6Dangle.json")
    with open(lookup_json) as f:
        lk = json.load(f)
    entries = lk[TYPE_NAMES[grasp_type]]
    wcm = float(width) * 100.0
    key = min(entries.keys(), key=lambda k: abs(float(k) - wcm))
    six = entries[key]["6d"]

    def m(v):
        v = float(v)
        return 2000 if v >= 1000 else int(round(2.0 * v))

    return {5: m(six[0]), 4: m(six[1]), 3: m(six[2]), 2: m(six[3]),
            1: m(six[4]), 6: m(six[5])}


def main():
    import open3d as o3d

    with open(GRASP_JSON) as f:
        r = json.load(f)
    gt = int(r["grasp_type"])
    width = float(r.get("two_finger_width", r["width"]))
    t_g = np.array(r["two_finger_translation"])
    R_g = np.array(r["two_finger_rotation_matrix"]).reshape(3, 3)
    print(f"场景: Type {gt} width={width:.3f} 抓取点={np.round(t_g*1000,0)} (相机系)")
    print(f"  接近X={np.round(R_g[:,0],2)} 开合Y={np.round(R_g[:,1],2)}")

    # 目标: 圆柱沿 grip Z 横放 (香蕉类物体示意), 半径3cm
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=0.03, height=0.16)
    Rz = np.eye(3)
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

    # 蓝色两指线框 (graspnetAPI, 17列数组构造)
    try:
        from graspnetAPI import Grasp
        arr = np.zeros(17)
        arr[0], arr[1], arr[2], arr[3] = 1.0, width, 0.03, 0.02
        arr[4:13] = R_g.reshape(-1)
        arr[13:16] = t_g
        arr[16] = -1
        g = Grasp(arr)
        gripper = g.to_open3d_geometry()
        gripper.paint_uniform_color([0.1, 0.4, 1.0])
        gripper_pts = np.asarray(gripper.sample_points_uniformly(8000).points)
    except Exception as e:
        print(f"[警告] graspnetAPI 线框失败({e}), 跳过")
        gripper_pts = None

    ohg._init()   # 构建 TYPE_CLOSE_MOTORS
    # 姿态A: 当前代码 (hand_bridge 闭合值 = TYPE_CLOSE_MOTORS)
    motors_a = dict(ohg.chg.TYPE_CLOSE_MOTORS[gt]["motors"])
    t_a, R_a = anchor_from_motors(motors_a, gt, t_g, R_g)
    # 姿态B: 提议 (2×6d 宽度化)
    motors_b = width_motors_local(gt, width)
    t_b, R_b = anchor_from_motors(motors_b, gt, t_g, R_g)
    print(f"姿态A(当前闭合值) 电机={motors_a}")
    print(f"姿态B(提议2×6d)   电机={motors_b}")

    st = ohg._init()
    meshes = {
        "current": build_hand_mesh(motors_a, np.vstack(
            [np.hstack([R_a, t_a.reshape(3, 1)]), [0, 0, 0, 1]])),
        "proposed": build_hand_mesh(motors_b, np.vstack(
            [np.hstack([R_b, t_b.reshape(3, 1)]), [0, 0, 0, 1]])),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    for tag, mesh in meshes.items():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(15, 7))
        for k, (title, elev, azim) in enumerate(
                [("相机视角 (从相机原点看)", -88, -90), ("侧视", 15, 110)]):
            ax = fig.add_subplot(1, 2, k + 1, projection="3d")
            pts = np.asarray(mesh.sample_points_uniformly(12000).points)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.2, c="red", alpha=0.5)
            cpts = np.asarray(cyl.sample_points_uniformly(5000).points)
            ax.scatter(cpts[:, 0], cpts[:, 1], cpts[:, 2], s=1.2, c="tab:blue", alpha=0.5)
            if gripper_pts is not None:
                ax.scatter(gripper_pts[:, 0], gripper_pts[:, 1], gripper_pts[:, 2],
                           s=1.2, c="royalblue", alpha=0.9)
            # 抓取系坐标轴
            for vec, col in zip([R_g[:, 0], R_g[:, 1], R_g[:, 2]],
                                ["r", "g", "b"]):
                seg = np.linspace(t_g, t_g + vec * 0.08, 10)
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], c=col, lw=2)
            ax.set_title(f"{tag} — {title}")
            ax.set_xlabel("X_cam"); ax.set_ylabel("Y_cam"); ax.set_zlabel("Z_cam")
            ctr = t_g
            ax.set_xlim(ctr[0] - 0.22, ctr[0] + 0.22)
            ax.set_ylim(ctr[1] - 0.22, ctr[1] + 0.22)
            ax.set_zlim(ctr[2] - 0.22, ctr[2] + 0.22)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=elev, azim=azim)
        fp = os.path.join(OUT_DIR, f"anchor_compare_{tag}.png")
        plt.tight_layout()
        plt.savefig(fp, dpi=110)
        plt.close(fig)
        print(f"[输出] {fp}")
    print("\n红色=自研手  蓝点=目标物体  深蓝=两指线框  RGB线=抓取系XYZ")


if __name__ == "__main__":
    main()
