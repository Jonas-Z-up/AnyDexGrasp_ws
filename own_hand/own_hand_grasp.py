#!/usr/bin/env python3
"""自研手抓取锚定模块 — 用自研手 URDF 几何替代 Inspire 查表，把两指抓取位姿
换算为自研手 base 位姿（写 JSON 顶层 translation/rotation_matrix 的语义）。

变换数学（与 InspireHandR_grasp._graspTR_2_InspireHandRTR 同构）:

  Inspire:  T_world←palm = T_world←grasp @ inv(T_lookup)
            T_lookup = [R_lookup | t_lookup] = 两指抓取系在 Inspire 掌心系中的位姿
            (InspireHandR_grasp.py:325-333)

  自研手:   T_world←base = T_world←grasp @ inv(T_base←grasp_own)
            构造公式移植自上游 angles2stl.py common_grasp_information:
              t_own = 包络锚点 C_base = (拇指质心 + 指组质心中点)/2 (闭合位姿)
              R_own 列 = [n, u, n×u] (抓取系[XYZ]在 base 系):
                n = 接触三角形(拇指,指A,指B)法向 = 包络"杯口"朝向(指向物体),
                    符号定向: C·n > 0 (环心/包络中心在 base 原点 +n 侧, 即手从
                    后方够物体) —— **不是手指延伸轴**(v1 的"指尖插桌"bug 即源于
                    把 base+X 对齐抓取X); 旧"掌体在杯口负侧"规则对 Ring 手势
                    (接近轴=环轴, ⊥掌面法向)会翻转 180°, 2026-09-08 已修
                u = C_base → 拇指质心 方向
            两指抓取系约定 (graspnetAPI): X=接近(杯口法向), Y=开合, 原点=抓取中心。

  +90° 装配滚转被自然吸收: 接触组几何来自真实 FK, 不需要任何常数。

用法:
  python own_hand/own_hand_grasp.py --selftest            # 合成圆柱自测 + 渲染 PNG
  python own_hand/own_hand_grasp.py --selftest --types 3 5

接口:
  base_pose_from_grasp(two_finger_grasp, grasp_type)
      two_finger_grasp: 含 .translation/.rotation_matrix 的对象 (graspnetAPI.Grasp 兼容)
      → (translation(3,), rotation_matrix(3,3)) 自研手 base 在两指抓取同一坐标系下的位姿
"""

import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(WS_ROOT, "scripts"))          # hand_bridge
sys.path.insert(0, os.path.join(WS_ROOT, "scripts", "tools"))  # compute_hand_geometry

import compute_hand_geometry as chg  # noqa: E402  (FK/电机映射/闭合位姿工具)

# ── 各抓取类型的接触组定义（hand_bridge 类型语义）──────────────────
# 抓取系构造公式（移植自上游 angles2stl.py common_grasp_information）:
#   p0=拇指接触质心, p1/p2=对向侧两处接触质心
#   C      = (p0 + (p1+p2)/2)/2            ← 包络中心(锚点, 物体中心目标位)
#   tool_x = normalize(normal(p0,p1,p2))   ← 接近轴 = 接触三角形法向(杯口朝向)
#   tool_y = normalize(p0 − C)             ← 开合轴 = 拇指方向
# 注意 grasp_type 按 hand_bridge 语义解释 (执行器本来就直传 hand_bridge):
#   2=Ring, 3=Prismatic_3_Finger ... 与 Inspire 查表的类型名不完全对齐。
#
# Type 2 (Ring) = 用户确认的抓取形态 (2026-09-08 渲染判定):
#   拇指+食指+中指围环, 物体入"拇指 ↔ 食/中指缝隙"的对向区间;
#   三角形 = 拇指远节 + 食指远节 + 中指远节 (环开口三点)。
# 其余类型沿用 search_contact_groups.py 对 Inspire 查表的数值锁定结果,
#   待逐类型真机判定后更新。
TYPE_CONTACT_GROUPS = {
    1: {"thumb": ["thumb_2_r"], "fa": ["index_1_r"], "fb": ["index_2_r"]},
    2: {"thumb": ["thumb_2_r"], "fa": ["index_2_r"], "fb": ["middle_2_r"]},
    3: {"thumb": ["thumb_2_r"], "fa": ["index_1_r"], "fb": ["index_2_r"]},
    4: {"thumb": ["thumb_2_r"], "fa": ["index_2_r"], "fb": ["middle_2_r"]},
    5: {"thumb": ["thumb_2_r"], "fa": ["index_1_r"], "fb": ["index_2_r"]},
    6: {"thumb": ["thumb_2_r"], "fa": ["index_1_r"], "fb": ["index_2_r"]},
    7: {"thumb": ["thumb_2_r"], "fa": ["ring_1_r"], "fb": ["little_2_r"]},
    8: {"thumb": ["thumb_2_r"], "fa": ["index_1_r"], "fb": ["middle_2_r"]},
}

ALL_LINKS = ["base_link", "thumb_1_r", "thumb_2_r", "index_1_r", "index_2_r",
             "middle_1_r", "middle_2_r", "ring_1_r", "ring_2_r",
             "little_1_r", "little_2_r"]

_state = None   # 惰性初始化的 pybullet 会话缓存


def _init():
    """惰性初始化: 加载 URDF、算贴掌端角、建闭合电机表。仅首次调用开销大。"""
    global _state
    if _state is not None:
        return _state
    p, cid, hand, jidx, lidx = chg.load_hand_pybullet()
    closed = {}
    for mid in [1, 2, 3, 4, 5]:
        a, _d = chg.find_closed_angle(p, hand, jidx, lidx, mid)
        closed[mid] = a
    a, _d = chg.find_closed_angle(p, hand, jidx, lidx, 6,
                                  pre_angles=chg.angles_for_motor(1, closed[1]))
    closed[6] = a
    chg.CLOSED_ANGLE.update(closed)          # set_motors 依赖该全局
    chg._build_type_close_motors()           # hand_bridge 预塑形/闭合电机表
    _state = {"p": p, "hand": hand, "jidx": jidx, "lidx": lidx}
    return _state


_env_cache = {}


def envelope_data(grasp_type, samples=4000):
    """闭合位姿下的包络锚定几何（手 base 系）。结果按类型缓存。

    构造公式移植自上游 angles2stl.py:275-306 common_grasp_information:
      point0 = 拇指接触质心; point1/point2 = 两组对向手指接触质心
      center = (point0 + (point1+point2)/2) / 2
      u      = normalize(point0 − center)                  (拇指方向, 抓取系Y在base系)
      n      = normalize(cross(p1−p0, p2−p0))              (接触三角形法向, 杯口朝向)
      n 符号定向: C·n > 0 (包络中心在 base 原点 +n 侧, 手从后方够物体)
    返回 (center(3,), frame_axes(3,3)):
      frame_axes 列 = [n, u, n×u] = 抓取系 [X,Y,Z] 在 base 系中的方向。
    """
    key = int(grasp_type)
    if key in _env_cache:
        return _env_cache[key]
    st = _init()
    spec = TYPE_CONTACT_GROUPS.get(key)
    if spec is None:
        raise ValueError(f"未知抓取类型 {grasp_type} (需 1-8)")
    motors = chg.TYPE_CLOSE_MOTORS[key]["motors"]
    chg.set_motors(st["p"], st["hand"], st["jidx"], motors)

    def _centroid(links, n):
        pts = chg.link_surface_points(st["p"], st["hand"], links, samples=n)
        return pts.mean(axis=0)

    p0 = _centroid(spec["thumb"], max(samples // 2, 500))
    p1 = _centroid(spec["fa"], max(samples // 3, 400))
    p2 = _centroid(spec["fb"], max(samples // 3, 400))

    center = 0.5 * (p0 + 0.5 * (p1 + p2))
    u = p0 - center
    u = u / np.linalg.norm(u)
    n = np.cross(p1 - p0, p2 - p0)
    n = n / np.linalg.norm(n)

    # 符号定向: 直接判据 C·n > 0 —— 包络中心(物体位)必须在 base 原点的 +n 侧,
    # 即手从后方够物体、接近轴指向物体 (2026-09-08 修正)。
    # 旧规则"n 指向远离 base_link 一侧"对 Ring 手势是反的: Ring 的接近轴=环轴
    # (≈手指方向), 与掌面法向(食指蜷曲方向)近垂直 —— 用掌体在平面哪一侧定符号
    # 会翻转 180°, 实测 base 跑到抓取点前方 +52mm(反向)。
    # C·n 裕度实测 +51mm, 确定性强。
    if np.dot(center, n) < 0:
        n = -n
    z = np.cross(n, u)
    z = z / np.linalg.norm(z)
    # n 与 u 理论正交(中点构造), 数值再正交化一次 u
    u = u - np.dot(u, n) * n
    u = u / np.linalg.norm(u)
    z = np.cross(n, u)

    _env_cache[key] = (center, np.column_stack([n, u, z]))
    return _env_cache[key]


def base_pose_from_grasp(two_finger_grasp, grasp_type):
    """两指抓取位姿 → 自研手 base 位姿（同一世界/相机坐标系）。

    two_finger_grasp 需提供 .translation(3,) 与 .rotation_matrix(3,3)
    （graspnetAPI.Grasp 兼容; 碰撞调整后的值可直接用）。若提供 .depth
    （两指进深, 米）, 锚定点沿接近轴推进 depth —— 物体接触点在
    translation + depth×接近轴（对齐 Inspire 执行语义 ur.py:441 的
    depth+常数推进; 残余常数误差交 jog 标定）。
    返回 (translation(3,), rotation_matrix(3,3))。
    """
    center, R_own = envelope_data(grasp_type)   # R_own 列 = 抓取系[XYZ]在base系

    R_g = np.asarray(two_finger_grasp.rotation_matrix, dtype=float).reshape(3, 3)
    t_g = np.asarray(two_finger_grasp.translation, dtype=float).reshape(3)
    depth = float(getattr(two_finger_grasp, "depth", 0.0) or 0.0)
    t_contact = t_g + R_g[:, 0] * depth         # 物体接触点(沿接近轴推进 depth)

    # T_world←base = T_world←grasp @ inv(T_base←grasp)
    R_wb = R_g @ R_own.T
    t_wb = t_contact - R_wb @ center
    return t_wb, R_wb


def motor_positions(grasp_type, stage="preshape"):
    """The same motor commands used by hand_bridge, without connecting hardware."""
    from hand_bridge import PRESHAPE_LIBRARY
    preset = PRESHAPE_LIBRARY[int(grasp_type)]
    motors = {i + 1: int(v) for i, v in enumerate(preset["broadcast"])}
    if stage == "closed":
        for mid, _force, _speed, position in preset["close"]:
            motors[mid] = int(position)
    elif stage != "preshape":
        raise ValueError("stage must be preshape or closed")
    return motors


def link_meshes_for_motors(motors):
    """Link meshes in base_link coordinates at an explicit six-motor posture."""
    import open3d as o3d
    st = _init()
    # Complete vectors prevent stale joints leaking between candidate types.
    chg.set_motors(st["p"], st["hand"], st["jidx"], motors)
    frames = chg.link_frames(st["p"], st["hand"])
    meshes = []
    for name in ALL_LINKS:
        m = o3d.io.read_triangle_mesh(os.path.join(chg.MESH_DIR, name + ".STL"))
        if m.is_empty():
            raise ValueError("Missing hand mesh: " + name)
        m.remove_duplicated_vertices()
        m.remove_degenerate_triangles()
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = frames[name][1], frames[name][0]
        m.transform(T)
        meshes.append(m)
    return meshes


def hand_mesh_at_pose(translation, rotation, grasp_type, stage="preshape", color=None):
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    for link in link_meshes_for_motors(motor_positions(grasp_type, stage)):
        mesh += link
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rotation, translation
    mesh.transform(T)
    if color is not None:
        mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def hand_mesh_at_grasp(two_finger_grasp, grasp_type, color=None):
    """按锚定位姿摆好的整手 open3d TriangleMesh（世界/相机系，闭合位姿手型）。

    depth 进深已由 base_pose_from_grasp 内部消费（锚定点推进），此处不再叠加。
    供 run_inspire_preview 的 own 模式可视化使用。
    """
    t_base, R_base = base_pose_from_grasp(two_finger_grasp, grasp_type)
    return hand_mesh_at_pose(t_base, R_base, grasp_type, stage="closed", color=color)


# ── selftest ──────────────────────────────────────────────

class _FakeGrasp:
    def __init__(self, translation, rotation_matrix):
        self.translation = np.asarray(translation, dtype=float)
        self.rotation_matrix = np.asarray(rotation_matrix, dtype=float)


def _axis_angle(a, b):
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def selftest(types=(5,), out_dir=None):
    """合成圆柱 + 已知两指抓取 → 锚定位姿 + 数值校验 + 渲染 PNG。

    校验项(Debug 判据):
      1) 包络中心重投影误差 < 1e-6 m (构造性应为 ~0)
      2) 变换后对向轴与抓取系 Y 轴夹角 < 1°
      3) 渲染图: 手包住圆柱, 拇指在闭合轴上
    """
    import open3d as o3d

    if out_dir is None:
        out_dir = os.path.join(WS_ROOT, "own_hand", "renders", "hand")
    os.makedirs(out_dir, exist_ok=True)

    # 场景: 竖直圆柱 (轴沿世界 Y), 中心 c_obj; 抓取从上前方接近
    c_obj = np.array([0.0, 0.0, 0.0])
    RADIUS, HALF_H = 0.035, 0.06
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=RADIUS, height=2 * HALF_H)
    R_cyl = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)  # Z→Y
    cyl.rotate(R_cyl, center=(0, 0, 0))
    cyl.translate(c_obj)
    cyl.paint_uniform_color([0.3, 0.6, 0.9])

    # 两指抓取系: X=接近(世界 -Z, 从上向下), Y=开合(世界 +X), Z=X×Y
    x_g = np.array([0.0, 0.0, -1.0])
    y_g = np.array([1.0, 0.0, 0.0])
    z_g = np.cross(x_g, y_g)
    R_g = np.column_stack([x_g, y_g, z_g])
    grasp = _FakeGrasp(c_obj.copy(), R_g)

    all_links = ["base_link", "thumb_1_r", "thumb_2_r", "index_1_r", "index_2_r",
                 "middle_1_r", "middle_2_r", "ring_1_r", "ring_2_r",
                 "little_1_r", "little_2_r"]

    for gt in types:
        t_base, R_base = base_pose_from_grasp(grasp, gt)
        center, R_axes = envelope_data(gt)
        u = R_axes[:, 1]                      # 拇指方向(抓取系Y)在 base 系

        # 校验 1: 包络中心变换后应落在抓取中心
        err_c = np.linalg.norm(R_base @ center + t_base - c_obj)
        # 校验 2: 拇指方向变换后应与抓取系 Y 同向
        ang_u = _axis_angle(R_base @ u, y_g)
        ok = (err_c < 1e-6) and (ang_u < 1.0)
        print(f"[Type {gt}] err_center={err_c*1000:.4f}mm (<0.001)  "
              f"ang_opposition={ang_u:.4f}° (<1)  {'✅ PASS' if ok else '❌ FAIL'}")

        # 渲染: 闭合位姿整手各 link mesh → 变换到 base 位姿
        st = _init()
        motors = chg.TYPE_CLOSE_MOTORS[gt]["motors"]
        chg.set_motors(st["p"], st["hand"], st["jidx"], motors)
        frames = chg.link_frames(st["p"], st["hand"])
        T = np.vstack([np.hstack([R_base, t_base.reshape(3, 1)]), [0, 0, 0, 1]])
        hand_mesh = o3d.geometry.TriangleMesh()
        for ln in all_links:
            m = o3d.io.read_triangle_mesh(os.path.join(chg.MESH_DIR, ln + ".STL"))
            R_l, t_l = frames[ln][1], frames[ln][0]
            T_l = np.vstack([np.hstack([R_l, t_l.reshape(3, 1)]), [0, 0, 0, 1]])
            m.transform(T_l)      # 摆到闭合关节角
            m.transform(T)        # 摆到锚定位姿
            hand_mesh += m
        hand_mesh.paint_uniform_color([0.9, 0.25, 0.2])
        hand_mesh.compute_vertex_normals()

        # 抓取坐标系三轴 (0.1m)
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        axes.transform(np.vstack([np.hstack([R_g, c_obj.reshape(3, 1)]), [0, 0, 0, 1]]))
        frame_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        frame_base.transform(T)

        out_png = os.path.join(out_dir, f"own_anchor_selftest_t{gt}.png")
        _render_scenes([cyl, hand_mesh, axes, frame_base], out_png,
                       labels=[f"Type {gt} 自研手锚定自测"])
        print(f"[Type {gt}] 渲染 → {out_png}")


def _render_scenes(geoms, out_png, labels=None):
    """matplotlib 点云散点离线渲染（确定性强、无 GUI 依赖）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import open3d as o3d
    fig = plt.figure(figsize=(16, 7))
    for k, (el, az) in enumerate([("front", -60), ("side", 30)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        for i, g in enumerate(geoms):
            pts = None
            if isinstance(g, o3d.geometry.TriangleMesh):
                pts = np.asarray(g.sample_points_uniformly(6000).points)
            elif isinstance(g, o3d.geometry.PointCloud):
                pts = np.asarray(g.points)
            if pts is None or len(pts) == 0:
                continue
            base_color = np.asarray(
                g.colors[0]) if len(getattr(g, "colors", [])) else None
            color = tuple(base_color[:3]) if base_color is not None \
                else ("tab:red" if i == 1 else "tab:gray")
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.5, c=[color], alpha=0.6)
        if labels:
            ax.set_title(f"{labels[0]} — {el}")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.view_init(elev=25, azim=az)
        ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="自研手抓取锚定模块 (selftest / 接口演示)")
    ap.add_argument("--selftest", action="store_true", help="运行合成圆柱自测")
    ap.add_argument("--types", type=int, nargs="+", default=[5],
                    help="自测的抓取类型 (默认 5=Medium Wrap)")
    args = ap.parse_args()
    if not args.selftest:
        print(__doc__)
        print("接口: base_pose_from_grasp(two_finger_grasp, grasp_type)")
        print("运行 --selftest 做合成场景验证。")
        return
    selftest(types=args.types)


if __name__ == "__main__":
    main()
