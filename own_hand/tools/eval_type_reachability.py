#!/usr/bin/env python3
"""8 种抓取类型的"桌面顶抓可达性"对比评估。

背景 (2026-09-09): Type 2 Ring 竖直顶抓低矮物体时, 法兰被拖到台面高度
(实测 2.6cm, 腕部半径 3-4cm ⇒ 不可达)。本工具逐类型计算:
  - 包络几何: C(包络中心), n(接近轴/杯口法向), u(对向轴, 全在 base 系)
  - 结构指标: C·n = 锚定点沿接近轴的深度(顶抓时=免费高度),
              |C⊥| = 垂直于接近轴的横向拖拽量
  - 可达性: 锚定一个标准竖直顶抓(物体中心 2cm 高)后, 法兰高度 /
            整手 mesh 最低点 vs 桌面, 4 个绕接近轴滚转取最优
  - 渲染: 侧视+俯视 PNG (红=手, 蓝=物体, 灰=桌面)

用法: python own_hand/tools/eval_type_reachability.py [--types 1 2 ...] [--render-topk N]
"""
import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(WS_ROOT, "own_hand"))
sys.path.insert(0, os.path.join(WS_ROOT, "scripts"))
sys.path.insert(0, os.path.join(WS_ROOT, "scripts", "tools"))

import own_hand_grasp as ohg  # noqa: E402

HAND_MOUNT_Z = 0.05       # config hand_mount_z_own (法兰在 base 后方 5cm)
TABLE_CLEARANCE_WANT = 0.08  # 法兰至少应高于桌面 8cm (腕体半径余量)


class _G:
    pass


def make_grasp(approach, opening):
    """两指抓取系: X=接近(指向物体), Y=开合。均自动正交化。"""
    x = np.asarray(approach, float); x /= np.linalg.norm(x)
    y = np.asarray(opening, float); y -= (y @ x) * x; y /= np.linalg.norm(y)
    g = _G()
    g.translation = np.zeros(3)          # 物体中心在原点
    g.depth = 0.0
    g.rotation_matrix = np.column_stack([x, y, np.cross(x, y)])
    return g


def roll_grasp(g, deg):
    """绕接近轴滚转开合轴。"""
    x = g.rotation_matrix[:, 0]
    y0 = np.array([1.0, 0, 0])
    y0 -= (y0 @ x) * x; y0 /= np.linalg.norm(y0)
    a = np.radians(deg)
    y = np.cos(a) * y0 + np.sin(a) * np.cross(x, y0)
    return make_grasp(x, y)


def flange_of(t_base, R_base):
    return t_base - HAND_MOUNT_Z * R_base[:, 0]


def eval_type(t, verbose=True):
    """返回 (最优滚转deg, 指标dict); 对标准竖直顶抓评估。"""
    C, R_own = ohg.envelope_data(t)
    n, u = R_own[:, 0], R_own[:, 1]
    Cpar = float(C @ n)
    Cperp = float(np.linalg.norm(C - Cpar * n))
    ang_nX = float(np.degrees(np.arccos(np.clip(n @ np.array([1., 0, 0]), -1, 1))))

    # 标准竖直顶抓: 接近轴竖直向下, 物体中心在原点, 桌面 z=-0.02
    TABLE_Z = -0.02
    best = None
    for roll in (0, 90, 180, 270):
        g = roll_grasp(make_grasp([0, 0, -1], [1, 0, 0]), roll)
        t_b, R_b = ohg.base_pose_from_grasp(g, t)
        p_fl = flange_of(t_b, R_b)
        fl_h = (p_fl[2] - TABLE_Z) * 100            # 法兰高于桌面 cm
        fl_d = float(np.linalg.norm(p_fl[:2])) * 100
        mesh = ohg.hand_mesh_at_grasp(g, t)
        zmin = float(np.asarray(mesh.vertices)[:, 2].min())
        pen = (TABLE_Z - zmin) * 100                 # >0 = 穿桌 cm
        score = fl_h - 2 * max(pen, 0)               # 高法兰+不穿桌优先
        if best is None or score > best[1]["score"]:
            best = (roll, dict(fl_h=fl_h, fl_d=fl_d, pen=pen, score=score))
    roll, m = best
    if verbose:
        ok = "✅" if (m["fl_h"] > TABLE_CLEARANCE_WANT * 100 and m["pen"] < 0.5) else "❌"
        print(f"Type {t}: n·X={ang_nX:5.1f}°  C·n={Cpar*100:5.1f}cm  |C⊥|={Cperp*100:5.1f}cm"
              f"  |最优滚转 {roll:3d}°| 法兰高={m['fl_h']:5.1f}cm  水平距={m['fl_d']:5.1f}cm"
              f"  穿桌={m['pen']:5.1f}cm  {ok}")
    return roll, m, (C, R_own)


def render_type(t, roll, out_png, anchor_raise_m=0.0):
    """合成香蕉(11×4×3cm 贴桌) + 桌面, 按滚转渲染侧视+斜视。

    anchor_raise_m: 锚定点沿竖直上抬量(正=抓物体上半部), 供拇指避桌对比。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import open3d as o3d

    TABLE_Z = -0.02
    table = o3d.geometry.TriangleMesh.create_box(0.7, 0.5, 0.01)
    table.translate((-0.35, -0.25, TABLE_Z - 0.01))
    table.paint_uniform_color([0.6, 0.6, 0.6])
    # 香蕉 = 逐轴缩放椭球 (半轴 5.5×2×1.5cm, 底面贴桌)。open3d 的 scale()
    # 是三轴均匀缩放, 逐轴必须直接改顶点坐标。
    banana = o3d.geometry.TriangleMesh.create_sphere(1.0, 24)
    bv = np.asarray(banana.vertices) * np.array([0.055, 0.020, 0.015])
    banana.vertices = o3d.utility.Vector3dVector(bv)
    banana.translate((0, 0, -0.005))      # 底面 z=-0.02 贴桌
    banana.paint_uniform_color([0.2, 0.4, 0.9])

    g = roll_grasp(make_grasp([0, 0, -1], [1, 0, 0]), roll)
    if anchor_raise_m:
        g.translation = g.translation + np.array([0, 0, anchor_raise_m])  # z 向上
    hand = ohg.hand_mesh_at_grasp(g, t, color=[0.9, 0.2, 0.15])

    geoms = [table, banana, hand,
             o3d.geometry.TriangleMesh.create_coordinate_frame(0.06)]
    fig = plt.figure(figsize=(14, 6.5))
    for k, (el, az, name) in enumerate([(8, -80, "侧视"), (55, -60, "斜视")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        for i, gm in enumerate(geoms):
            if isinstance(gm, o3d.geometry.TriangleMesh):
                pts = np.asarray(gm.sample_points_uniformly(4000).points)
                col = np.asarray(gm.vertex_colors[0]) if len(gm.vertex_colors) \
                    else ("tab:gray" if i == 0 else "tab:blue")
            else:
                continue
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.2, c=[col], alpha=0.5)
        ax.set_title(f"Type {t} (滚转 {roll}°, 锚定抬升 {anchor_raise_m*100:.0f}cm) — {name}")
        ax.view_init(elev=el, azim=az)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"  渲染 → {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", type=int, nargs="+", default=list(range(1, 9)))
    ap.add_argument("--render-topk", type=int, default=2,
                    help="渲染法兰高度最高的前 N 个类型 (0=不渲染)")
    args = ap.parse_args()

    print("== 结构指标 (base 系) + 竖直顶抓可达性 (4 滚转取优) ==")
    print("n·X: 接近轴与手指延伸轴夹角 (0°=杯口沿手臂, 90°=杯口垂直手臂)")
    print("C·n: 锚定点沿接近轴深度 ≈ 顶抓时的免费腕高; |C⊥|: 横向拖拽量\n")
    results = {}
    for t in args.types:
        try:
            roll, m, _ = eval_type(t)
            results[t] = (roll, m)
        except Exception as e:
            print(f"Type {t}: 评估失败 {e}")

    if args.render_topk:
        out_dir = os.path.join(WS_ROOT, "own_hand", "renders", "hand")
        os.makedirs(out_dir, exist_ok=True)
        print("\n== 渲染: 法兰高度 Top%d + 锚定抬升对比 ==" % args.render_topk)
        top = sorted(results.items(), key=lambda kv: -kv[1][1]["fl_h"])[:args.render_topk]
        for t, (roll, _m) in top:
            render_type(t, roll, os.path.join(out_dir, f"reach_t{t}.png"))
            if t == 5:  # 拇指避桌变体: 锚定抬升 2cm
                render_type(t, roll, os.path.join(out_dir, f"reach_t{t}_raised.png"),
                            anchor_raise_m=0.02)


if __name__ == "__main__":
    main()
