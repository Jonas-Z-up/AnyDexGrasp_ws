#!/usr/bin/env python3
"""抓取位姿实体渲染（pybullet TinyRenderer, 离线无需显示器）。

替代 matplotlib 点云散点图: 加载真实 URDF/STL 网格, 实体着色渲染
红=自研手(闭合位姿), 蓝=香蕉(11×4×3cm 贴桌), 灰=桌面, RGB线=抓取系。

用法:
  python own_hand/tools/render_solid_grasp.py                 # t5 / t5_raised / t2
  python own_hand/tools/render_solid_grasp.py --types 5 --raise 0.02
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

import own_hand_grasp as ohg   # noqa: E402
import compute_hand_geometry as chg  # noqa: E402

TABLE_Z = -0.02            # 桌面顶面 (物体中心在原点上方 2cm)
OUT_DIR = os.path.join(WS_ROOT, "own_hand", "renders", "hand")
BANANA_OBJ = os.path.join(OUT_DIR, "banana_ellipsoid.obj")


def make_grasp(approach, opening):
    x = np.asarray(approach, float); x /= np.linalg.norm(x)
    y = np.asarray(opening, float); y -= (y @ x) * x; y /= np.linalg.norm(y)
    class _G: pass
    g = _G()
    g.translation = np.zeros(3)
    g.depth = 0.0
    g.rotation_matrix = np.column_stack([x, y, np.cross(x, y)])
    return g


def roll_grasp(g, deg):
    x = g.rotation_matrix[:, 0]
    y0 = np.array([1.0, 0, 0]); y0 -= (y0 @ x) * x; y0 /= np.linalg.norm(y0)
    a = np.radians(deg)
    return make_grasp(x, np.cos(a) * y0 + np.sin(a) * np.cross(x, y0))


def write_banana_obj():
    """半轴 5.5×2×1.5cm 椭球 OBJ (重心在原点)。"""
    import pybullet as pb  # 仅确保目录时机
    os.makedirs(OUT_DIR, exist_ok=True)
    u = np.linspace(0, 2 * np.pi, 32)
    v = np.linspace(0, np.pi, 16)
    U, V = np.meshgrid(u, v)
    P = np.stack([np.cos(U) * np.sin(V) * 0.055,
                  np.sin(U) * np.sin(V) * 0.020,
                  np.cos(V) * 0.015], axis=-1).reshape(-1, 3)
    with open(BANANA_OBJ, "w") as f:
        for p3 in P:
            f.write(f"v {p3[0]:.5f} {p3[1]:.5f} {p3[2]:.5f}\n")
        for i in range(len(v) - 1):
            for j in range(len(u) - 1):
                a = i * len(u) + j + 1
                b, c, d = a + 1, a + len(u), a + len(u) + 1
                f.write(f"f {a} {b} {d}\nf {a} {d} {c}\n")


def build_scene(pb):
    """桌面 + 香蕉 (纯 visual body, 零质量)。返回创建的 body id 列表。"""
    table_vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.35, 0.25, 0.005],
                                     rgbaColor=[0.62, 0.62, 0.62, 1.0])
    t1 = pb.createMultiBody(baseMass=0, baseVisualShapeIndex=table_vis,
                            basePosition=[0, 0, TABLE_Z - 0.005])
    banana_vis = pb.createVisualShape(pb.GEOM_MESH, fileName=BANANA_OBJ,
                                      rgbaColor=[0.16, 0.38, 0.92, 1.0],
                                      meshScale=[1, 1, 1])
    t2 = pb.createMultiBody(baseMass=0, baseVisualShapeIndex=banana_vis,
                            basePosition=[0, 0, -0.005])
    return [t1, t2]


def place_hand(pb, st, t, g):
    """按类型闭合 + 锚定位姿摆手, 并整体染红。返回 (t_base, R_base)。"""
    chg.set_motors(st["p"], st["hand"], st["jidx"], chg.TYPE_CLOSE_MOTORS[t]["motors"])
    t_b, R_b = ohg.base_pose_from_grasp(g, t)
    from scipy.spatial.transform import Rotation
    quat = Rotation.from_matrix(np.asarray(R_b, float)).as_quat().tolist()
    pb.resetBasePositionAndOrientation(st["hand"], t_b.tolist(), quat)
    for li in range(-1, pb.getNumJoints(st["hand"])):
        try:
            pb.changeVisualShape(st["hand"], li, rgbaColor=[0.88, 0.14, 0.10, 1.0])
        except Exception:
            pass
    return t_b, R_b


def draw_grasp_frame(pb, size=0.06):
    axes = [(size, 0, 0), (0, size, 0), (0, 0, size)]
    cols = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for a, c in zip(axes, cols):
        pb.addUserDebugLine([0, 0, 0], list(a), lineColorRGB=c, lineWidth=2)


def capture(pb, yaw, pitch, dist, out_png, w=1440, h=1080):
    import matplotlib.image as mimg
    view = pb.computeViewMatrixFromYawPitchRoll([0, 0, 0.01], dist, yaw, pitch, 0, 2)
    proj = pb.computeProjectionMatrixFOV(45, w / h, 0.03, 3.0)
    out = pb.getCameraImage(w, h, viewMatrix=view, projectionMatrix=proj,
                            renderer=pb.ER_TINY_RENDERER)
    rgb = np.reshape(out[2], (h, w, 4))[:, :, :3]
    mimg.imsave(out_png, rgb)
    return rgb


def pixel_report(name, rgb):
    """程序化自检: 各要素像素占比 (本会话无法读图, 以数值代替目检)。"""
    r, g_, b = rgb[:, :, 0].astype(int), rgb[:, :, 1].astype(int), rgb[:, :, 2].astype(int)
    red = (r > 120) & (r > 1.5 * g_) & (r > 1.5 * b)
    blue = (b > 110) & (b > 1.4 * r) & (b > g_)
    gray = (abs(r - g_) < 14) & (abs(g_ - b) < 14) & (r > 90) & (r < 215)
    tot = rgb.shape[0] * rgb.shape[1]
    print(f"[自检] {name}: 红(手) {red.sum()/tot*100:.1f}%  蓝(香蕉) {blue.sum()/tot*100:.2f}%"
          f"  灰(桌面) {gray.sum()/tot*100:.1f}%")
    return red.sum(), blue.sum(), gray.sum()


def render_variant(st, pb, tag, t, roll=0, raise_m=0.0, scene_bodies=None):
    # 不能 resetSimulation(): 会销毁 ohg._init() 加载的手体, 关节索引随之失效。
    # 只清除上一变体的桌面/香蕉/调试线, 手体保留复用。
    for b in scene_bodies or []:
        pb.removeBody(b)
    pb.removeAllUserDebugItems()
    scene_bodies = build_scene(pb)
    g = roll_grasp(make_grasp([0, 0, -1], [1, 0, 0]), roll)
    if raise_m:
        g.translation = g.translation + np.array([0, 0, raise_m])  # 场景 z 向上, 桌面在 -0.02
    t_b, R_b = place_hand(pb, st, t, g)
    draw_grasp_frame(pb)
    p_fl = t_b - 0.05 * R_b[:, 0]
    print(f"\n== {tag}: Type {t} 滚转{roll}° 抬升{raise_m*100:.1f}cm | "
          f"法兰高 {(p_fl[2]-TABLE_Z)*100:.1f}cm ==")
    # pitch 正值 = 相机在水平面下方 (该版本 computeViewMatrixFromYawPitchRoll
    # 约定), 会钻到桌面底下拍 — 俯视一律用负 pitch。
    views = [("侧视", 0, -6), ("斜视", 42, -32)]
    rgbs = []
    for name, yaw, pitch in views:
        rgb = capture(pb, yaw, pitch, 0.52,
                      os.path.join(OUT_DIR, f"_tmp_{tag}_{name}.png"))
        pixel_report(f"{tag}-{name}", rgb)
        rgbs.append(rgb)
    import PIL.Image as Image
    gap = np.full((rgbs[0].shape[0], 12, 3), 255, np.uint8)
    Image.fromarray(np.hstack([rgbs[0], gap, rgbs[1]])).save(
        os.path.join(OUT_DIR, f"reach_solid_{tag}.png"))
    print(f"  → {os.path.join(OUT_DIR, f'reach_solid_{tag}.png')}")
    return scene_bodies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", type=int, nargs="+", default=[5, 2])
    ap.add_argument("--raise", dest="raise_m", type=float, default=0.0)
    ap.add_argument("--roll", type=int, default=0)
    args = ap.parse_args()

    write_banana_obj()
    pb = __import__("pybullet")
    st = ohg._init()                       # 复用管线的 URDF 会话 (DIRECT)

    plan = [(f"t{args.types[0]}", args.types[0], args.roll, 0.0)]
    if args.raise_m > 0:
        plan[0] = (f"t{args.types[0]}_raised", args.types[0], args.roll, args.raise_m)
    elif args.types[0] == 5:               # 默认补拇指避桌对比变体 (扫描定值 2.5cm)
        plan.append((f"t{args.types[0]}_raised", args.types[0], args.roll, 0.025))
    for t in args.types[1:]:
        plan.append((f"t{t}", t, args.roll, 0.0))

    scene = []
    for tag, t, roll, rm in plan:
        scene = render_variant(st, pb, tag, t, roll=roll, raise_m=rm,
                               scene_bodies=scene)


if __name__ == "__main__":
    main()
