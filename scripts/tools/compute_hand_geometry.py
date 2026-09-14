#!/usr/bin/env python3
"""自研手几何计算工具 — 从 URDF 计算标定所需的全部几何数据（零手工测量）。

依据 assets/own_urdf/dexterous_hand/urdf/dexterous_hand.urdf（SolidWorks 名义几何），
pybullet FK + 距离场分析，输出 docs/自研手几何数据.md：
  1. 各自由度贴掌端角：远端指尖节到掌面距离最小处的关节角（若距离 < 1.5mm 即真实
     接触，否则为名义闭合位姿）；ID2~ID5 贴掌端在 p=0，ID6 贴掌端在 p=2000
  2. 电机位置 (0-2000) → 关节角线性映射表（ID2~ID5: 2000=张开=0rad；
     ID6: p 增大→内旋加深, p=2000=贴掌=120°）
  3. 各电机档位的指尖开口距离（拇-食、拇-中、四指横向跨度）
  4. 抓取类型闭合位姿（close 值）下的包络中心与实际闭合开口

用法:
  python scripts/tools/compute_hand_geometry.py                 # 计算并写文档
  python scripts/tools/compute_hand_geometry.py --render 1000   # 另存该电机档渲染图(照片核验用)
  python scripts/tools/compute_hand_geometry.py --render all    # 渲染 2000/1500/1000/500/0 五档

注意:
  - URDF 为名义 CAD 几何（含装配公差与欠驱动耦合假设），结果是高精度初值而非真值；
    落点类残差由 jog 标定闭环吸收，开度类残差用本工具 --render 渲染图与实机照片比对核验。
  - ID6 方向以实机核对为准（2026-09-06）：p 增大→内旋加深（p=2000 贴掌, p=0 外展）。
    hand_bridge.py 旧注释（2000=垂直掌心）方向写反，仅注释有误、PRESHAPE_LIBRARY
    数值是实机调好的、不受影响。
  - 四指的 *_joint_2 为 mimic 关节，pybullet 不支持，本工具按 multiplier=1.0 手动同步。
"""

import argparse
import datetime
import os
import sys
import tempfile

import numpy as np

WS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URDF_PATH = os.path.join(WS_ROOT, "assets", "own_urdf", "dexterous_hand",
                         "urdf", "dexterous_hand.urdf")
MESH_DIR = os.path.join(WS_ROOT, "assets", "own_urdf", "dexterous_hand", "meshes")
OUT_MD = os.path.join(WS_ROOT, "docs", "自研手几何数据.md")
RENDER_DIR = os.path.join(WS_ROOT, "own_hand", "renders", "hand")

sys.path.insert(0, os.path.join(WS_ROOT, "scripts"))

# ── 电机 → 关节 映射（依据 hand_bridge.py 注释与 CLAUDE.md 电机表）──────
# ID1:拇指屈伸 ID2:食指 ID3:中指 ID4:无名指 ID5:小指 ID6:拇指内旋
# 四指关节2为 mimic(multiplier=1.0)，与关节1同步设角。
# dir=-1: p 增大→张开(角度减小)；dir=+1: p 增大→内旋加深(角度增大)。
# ID6 方向 2026-09-06 实机核对修正（旧注释方向写反）。
MOTOR_JOINTS = {
    1: {"name": "拇指屈伸", "joints": ["thumb_joint_2"], "distal": "thumb_2_r", "dir": -1},
    2: {"name": "食指", "joints": ["index_joint_1", "index_joint_2"], "distal": "index_2_r", "dir": -1},
    3: {"name": "中指", "joints": ["middle_joint_1", "middle_joint_2"], "distal": "middle_2_r", "dir": -1},
    4: {"name": "无名指", "joints": ["ring_joint_1", "ring_joint_2"], "distal": "ring_2_r", "dir": -1},
    5: {"name": "小指", "joints": ["little_joint_1", "little_joint_2"], "distal": "little_2_r", "dir": -1},
    6: {"name": "拇指内旋", "joints": ["thumb_joint_1"], "distal": "thumb_2_r", "dir": +1},
}
# 全开/复位姿态：ID1~ID5 全伸直，ID6 外展（p=0）
OPEN_POSE = {1: 2000, 2: 2000, 3: 2000, 4: 2000, 5: 2000, 6: 0}
TIP_LINKS = {"thumb": "thumb_2_r", "index": "index_2_r", "middle": "middle_2_r",
             "ring": "ring_2_r", "little": "little_2_r"}
MOTOR_POS_GRID = [2000, 1500, 1000, 500, 0]
CONTACT_THRESH_M = 0.0015   # 远端指节-掌面距离低于此视为真实接触

CLOSED_ANGLE = {}           # motor_id → 闭合端点角 (rad)，set_motors 依赖


def load_hand_pybullet():
    """加载 URDF（改写 package:// 为绝对路径），返回 (p, cid, hand, joint索引表, link索引表)。"""
    import pybullet as p

    with open(URDF_PATH, encoding="utf-8-sig") as f:
        text = f.read()
    patched = text.replace("package://dexterous_hand/meshes/", MESH_DIR.rstrip("/") + "/")
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
    tmp.write(patched)
    tmp.close()

    cid = p.connect(p.DIRECT)
    hand = p.loadURDF(tmp.name, useFixedBase=True,
                      flags=p.URDF_USE_SELF_COLLISION | p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    os.unlink(tmp.name)

    joint_idx, link_idx = {}, {}
    for i in range(p.getNumJoints(hand)):
        info = p.getJointInfo(hand, i)
        joint_idx[info[1].decode()] = i          # 关节名
        link_idx[info[12].decode()] = i          # child link 名
    link_idx["base_link"] = -1
    return p, cid, hand, joint_idx, link_idx


def min_link_palm_distance(p, hand, link_idx, link_names, query=0.5):
    """若干 link 与 base_link 的真实最小距离 (m)；无最近点返回 inf。"""
    d_min = float("inf")
    for ln in link_names:
        pts = p.getClosestPoints(hand, hand, distance=query,
                                 linkIndexA=link_idx[ln], linkIndexB=-1)
        for pt in pts:
            d_min = min(d_min, pt[8])            # contactDistance
    return d_min


def set_joint_angles(p, hand, joint_idx, angles):
    for jname, a in angles.items():
        p.resetJointState(hand, joint_idx[jname], a)


def angles_for_motor(mid, a):
    return {j: a for j in MOTOR_JOINTS[mid]["joints"]}


def find_closed_angle(p, hand, joint_idx, link_idx, motor_id, pre_angles=None):
    """闭合端点角 = 远端指节-掌面距离最小时的关节角。

    该自由度从 0 扫到上限，其余自由度保持 pre_angles（缺省全 0 = 全开）；
    粗扫取距离 argmin，再局部细化。返回 (angle_rad, min_dist_m)。
    """
    m = MOTOR_JOINTS[motor_id]
    upper = min(p.getJointInfo(hand, joint_idx[j])[9] for j in m["joints"])  # [9]=上限

    def dist_at(a):
        angles = dict(pre_angles or {})
        angles.update(angles_for_motor(motor_id, a))
        set_joint_angles(p, hand, joint_idx, angles)
        return min_link_palm_distance(p, hand, link_idx, [m["distal"]])

    n = 60
    best_a, best_d = 0.0, dist_at(0.0)
    step = upper / n
    for k in range(1, n + 1):
        a = upper * k / n
        d = dist_at(a)
        if d < best_d:
            best_a, best_d = a, d
    lo, hi = max(0.0, best_a - step), min(upper, best_a + step)
    for k in range(41):
        a = lo + (hi - lo) * k / 40
        d = dist_at(a)
        if d < best_d:
            best_a, best_d = a, d
    return best_a, best_d


def set_motors(p, hand, joint_idx, motor_positions):
    """按电机位置字典 {motor_id: pos} 设定整手姿态（线性映射 + mimic 同步）。

    未给出的电机默认取 OPEN_POSE（ID1~ID5=2000 伸直, ID6=0 外展）。
    dir=-1: 角度 = 贴掌端角 × (1 - p/2000)；dir=+1: 角度 = 贴掌端角 × p/2000。
    """
    angles = {}
    for mid, m in MOTOR_JOINTS.items():
        pos = motor_positions.get(mid, OPEN_POSE[mid])
        frac = (1.0 - float(pos) / 2000.0) if m["dir"] < 0 else (float(pos) / 2000.0)
        a = CLOSED_ANGLE.get(mid, 0.0) * frac
        for j in m["joints"]:
            angles[j] = a
    set_joint_angles(p, hand, joint_idx, angles)


def link_frames(p, hand):
    """所有 link 的 base 系位姿 → {link名: (pos(3,), R(3,3))}。

    pybullet getLinkState 给的是 world 位姿; 此处必须用 base_link 的 world
    位姿归一。旧实现把 base_link 硬编码为原点, 仅当手停在 world 原点时成立;
    手被 resetBasePositionAndOrientation 移动后再采样 (如渲染脚本先摆 t5
    再算 t2 包络), 接触点全部落在漂浮坐标系里, 包络几何被静默污染。
    手在原点时结果与旧实现逐字节相同。
    """
    # 注意: getBasePositionAndOrientation 返回的是 base 的质心(COM)系, 不是
    # URDF link 系; 两者差 getDynamicsInfo 的局部惯性偏移, 必须补偿, 否则
    # 归一会引入惯性偏移量的系统误差。
    pos_com, orn_com = p.getBasePositionAndOrientation(hand)
    R_com = np.array(p.getMatrixFromQuaternion(orn_com)).reshape(3, 3)
    di = p.getDynamicsInfo(hand, -1)
    p_loc, q_loc = np.array(di[3]), np.array(di[4])
    R_loc = np.array(p.getMatrixFromQuaternion(q_loc)).reshape(3, 3)
    R0 = R_com @ R_loc.T
    p0 = np.array(pos_com) - R0 @ p_loc
    frames = {"base_link": (np.zeros(3), np.eye(3))}
    for i in range(p.getNumJoints(hand)):
        st = p.getLinkState(hand, i, computeForwardKinematics=1)
        pos = R0.T @ (np.array(st[4]) - p0)
        R = R0.T @ np.array(p.getMatrixFromQuaternion(st[5])).reshape(3, 3)
        name = p.getJointInfo(hand, i)[12].decode()
        frames[name] = (pos, R)
    return frames


def tip_positions(p, hand, joint_idx, motor_positions):
    set_motors(p, hand, joint_idx, motor_positions)
    fr = link_frames(p, hand)
    return {k: fr[v][0] for k, v in TIP_LINKS.items()}


def link_surface_points(p, hand, link_names, samples=3000):
    """参与指节 mesh 表面采样点（手坐标系）。open3d 加载 STL。

    采样 = 面积加权三角面采样 + 固定种子。旧实现用 sample_points_uniformly,
    其 Python 绑定无 seed 参数, 跨进程质心抖动会让接触三角形法向摆动
    (近退化组合尤甚, 如 Type 2 拇指/食指/中指尖三点近共线), 导致锚定位姿
    厘米级漂移、法兰高度逐次运行不一致 (实测 Type 2 在 1.1~7.1cm 间跳动)。
    """
    import open3d as o3d
    fr = link_frames(p, hand)
    rng = np.random.RandomState(0)
    pts_all = []
    for ln in link_names:
        mesh = o3d.io.read_triangle_mesh(os.path.join(MESH_DIR, ln + ".STL"))
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.triangles)
        e1 = v[f[:, 1]] - v[f[:, 0]]
        e2 = v[f[:, 2]] - v[f[:, 0]]
        area = np.clip(0.5 * np.linalg.norm(np.cross(e1, e2), axis=1), 1e-12, None)
        tri = rng.choice(len(f), size=samples, p=area / area.sum())
        r1, r2 = rng.rand(samples), rng.rand(samples)
        s = np.sqrt(r1)
        pts = ((1 - s)[:, None] * v[f[tri, 0]]
               + (s * (1 - r2))[:, None] * v[f[tri, 1]]
               + (s * r2)[:, None] * v[f[tri, 2]])
        R, t = fr[ln][1], fr[ln][0]
        pts_all.append(pts @ R.T + t)
    return np.vstack(pts_all)


# ── 抓取类型闭合位姿（来自 scripts/hand_bridge.py 的 PRESHAPE_LIBRARY）────
# 电机向量 = broadcast 为底、close 列表逐指覆盖；未列出的保持 broadcast 值。
TYPE_CLOSE_MOTORS = {}


def _build_type_close_motors():
    try:
        from hand_bridge import PRESHAPE_LIBRARY
    except Exception as e:
        print(f"[警告] 无法导入 hand_bridge（{e}），跳过包络计算")
        return
    for tid, preset in PRESHAPE_LIBRARY.items():
        vec = {i + 1: v for i, v in enumerate(preset["broadcast"])}
        for mid, _f, _s, pos in preset["close"]:
            vec[mid] = pos
        TYPE_CLOSE_MOTORS[tid] = {"name": preset["name"], "motors": vec}


def render_motor(p, hand, motor_positions, path):
    """tiny renderer 离线渲染当前姿态：侧视图 + 顶视图（照片核验用）。"""
    set_motors(p, hand, joint_idx_global, motor_positions)
    aabb_min, aabb_max = np.array(p.getAABB(hand, -1)[0]), np.array(p.getAABB(hand, -1)[1])
    for i in range(p.getNumJoints(hand)):
        lo, hi = p.getAABB(hand, i)
        aabb_min, aabb_max = np.minimum(aabb_min, lo), np.maximum(aabb_max, hi)
    center = (aabb_min + aabb_max) / 2
    size = float(np.max(aabb_max - aabb_min))
    views = {"side": (0, -35), "top": (-90, -35)}
    for tag, (yaw, pitch) in views.items():
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=center.tolist(), distance=size * 2.2,
            yaw=yaw, pitch=pitch, roll=0, upAxisIndex=2)
        proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0,
                                            nearVal=size * 0.1, farVal=size * 5)
        img = p.getCameraImage(640, 480, viewMatrix=view, projectionMatrix=proj,
                               renderer=p.ER_TINY_RENDERER)[2]
        arr = np.reshape(img, (480, 640, 4))[:, :, :3]
        try:
            from PIL import Image
            Image.fromarray(arr.astype(np.uint8)).save(path.replace(".png", f"_{tag}.png"))
        except ImportError:
            with open(path.replace(".png", f"_{tag}.raw"), "wb") as f:
                f.write(arr.astype(np.uint8).tobytes())


# ── 主流程 ────────────────────────────────────────────────

def main():
    global joint_idx_global
    ap = argparse.ArgumentParser(description="自研手 URDF 几何计算（端点角/电机映射/开口/包络）")
    ap.add_argument("--render", default="",
                    help="渲染指定电机位置 (如 1000) 或 all，用于与实机照片比对核验")
    args = ap.parse_args()

    p, cid, hand, joint_idx, link_idx = load_hand_pybullet()
    joint_idx_global = joint_idx
    print(f"[URDF] 已加载: {URDF_PATH}")

    # 1. 闭合端点角（ID1-5 各自独立测：其余手指全开；ID6 在拇指屈伸全闭下测）
    endpoint_dist = {}
    print("\n[贴掌端] 远端指节-掌面最小距离分析（ID2~ID5 贴掌端=p0, ID6 贴掌端=p2000）:")
    for mid in [1, 2, 3, 4, 5]:
        a, d = find_closed_angle(p, hand, joint_idx, link_idx, mid)
        CLOSED_ANGLE[mid] = a
        endpoint_dist[mid] = d
    pre_thumb = angles_for_motor(1, CLOSED_ANGLE[1])     # 拇指屈伸全闭
    a, d = find_closed_angle(p, hand, joint_idx, link_idx, 6, pre_angles=pre_thumb)
    CLOSED_ANGLE[6] = a
    endpoint_dist[6] = d
    for mid in sorted(MOTOR_JOINTS):
        d = endpoint_dist[mid]
        note = "" if d < CONTACT_THRESH_M else \
            f"  ⚠ 未实际接触(最近 {d*1000:.1f}mm)，取最近点姿态为名义贴掌"
        if MOTOR_JOINTS[mid]["dir"] > 0:
            note += "  [贴掌端在 p=2000]"
        print(f"  ID{mid} {MOTOR_JOINTS[mid]['name']:　<4}: 贴掌端角 = {CLOSED_ANGLE[mid]:.4f} rad = "
              f"{np.degrees(CLOSED_ANGLE[mid]):6.1f}°{note}")

    # 2. 电机→关节角线性映射表 + 3. 开口距离（拇指取全开/复位姿态 ID6=0 外展）
    rows_map, rows_open = [], []
    for pos in MOTOR_POS_GRID:
        row = {"pos": pos}
        for mid in sorted(MOTOR_JOINTS):
            m = MOTOR_JOINTS[mid]
            frac = (1.0 - pos / 2000.0) if m["dir"] < 0 else (pos / 2000.0)
            row[mid] = CLOSED_ANGLE[mid] * frac
        rows_map.append(row)
        tips = tip_positions(p, hand, joint_idx, {i: pos for i in range(1, 6)})
        rows_open.append({
            "pos": pos,
            "d_thumb_index": float(np.linalg.norm(tips["thumb"] - tips["index"])),
            "d_thumb_middle": float(np.linalg.norm(tips["thumb"] - tips["middle"])),
            "d_index_little": float(np.linalg.norm(tips["index"] - tips["little"])),
        })

    # 4. 抓取类型闭合位姿包络
    _build_type_close_motors()
    envelopes = {}
    for tid, spec in sorted(TYPE_CLOSE_MOTORS.items()):
        set_motors(p, hand, joint_idx, spec["motors"])
        links = []
        for mid in sorted(spec["motors"]):
            if spec["motors"][mid] < 1900:        # 参与闭合的手指
                links.append(MOTOR_JOINTS[mid]["distal"])
                links.append(MOTOR_JOINTS[mid]["distal"].replace("_2_r", "_1_r"))
        links = sorted(set(links))
        pts = link_surface_points(p, hand, links)
        centroid = pts.mean(axis=0)
        fr = link_frames(p, hand)
        tip_pos = {k: fr[v][0] for k, v in TIP_LINKS.items()}
        act = []
        for a2, b2 in [("thumb", "index"), ("thumb", "middle"), ("index", "middle")]:
            act.append((f"{a2}-{b2}", float(np.linalg.norm(tip_pos[a2] - tip_pos[b2]))))
        envelopes[tid] = {"name": spec["name"], "motors": spec["motors"],
                          "centroid": centroid, "reach_x": float(centroid[0]),
                          "openings": act}
        print(f"\n[包络] Type {tid} {spec['name']}: 电机={spec['motors']}")
        print(f"  包络中心(手系): [{centroid[0]*1000:.1f}, {centroid[1]*1000:.1f}, "
              f"{centroid[2]*1000:.1f}] mm  (距掌原点沿+X {centroid[0]*1000:.1f} mm)")

    # 渲染
    if args.render:
        os.makedirs(RENDER_DIR, exist_ok=True)
        positions = MOTOR_POS_GRID if args.render == "all" else [int(args.render)]
        for pos in positions:
            fp = os.path.join(RENDER_DIR, f"motor_{pos:04d}.png")
            render_motor(p, hand, {i: pos for i in range(1, 6)}, fp)
            print(f"[渲染] 电机={pos} → {fp.replace('.png', '_side.png')} / _top.png")

    write_markdown(rows_map, rows_open, envelopes, endpoint_dist)
    p.disconnect()
    print(f"\n[输出] {OUT_MD}")


def write_markdown(rows_map, rows_open, envelopes, endpoint_dist):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# 自研手几何数据（URDF 计算生成）",
        "",
        f"> 由 `scripts/tools/compute_hand_geometry.py` 自动生成于 {now}，",
        f"> 依据 `assets/own_urdf/dexterous_hand/urdf/dexterous_hand.urdf` 名义几何。",
        "> 精度定位：高精度初值，非真值。落点类残差由 jog 标定闭环吸收；",
        "> 开度类残差用 `--render` 渲染图与实机照片比对核验。",
        "",
        "## 1. 贴掌端角（远端指节最贴近掌面的关节角）",
        "",
        "| 电机 | 自由度 | 贴掌端角 (rad) | (°) | 最近距离 (mm) | 贴掌端对应 p | 备注 |",
        "|---|---|---|---|---|---|---|",
    ]
    for mid in sorted(MOTOR_JOINTS):
        d = endpoint_dist[mid]
        note = "真实接触" if d < CONTACT_THRESH_M else "未实际接触，名义贴掌"
        p_end = 2000 if MOTOR_JOINTS[mid]["dir"] > 0 else 0
        lines.append(f"| ID{mid} | {MOTOR_JOINTS[mid]['name']} | {CLOSED_ANGLE[mid]:.4f} | "
                     f"{np.degrees(CLOSED_ANGLE[mid]):.1f} | {d*1000:.1f} | {p_end} | {note} |")
    lines += [
        "",
        "## 2. 电机位置 → 关节角 线性映射 (rad)",
        "",
        "| 电机位置 | ID1 拇屈 | ID2 食 | ID3 中 | ID4 无名 | ID5 小 | ID6 拇旋 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows_map:
        cells = " | ".join(f"{row[m]:.4f}" for m in sorted(MOTOR_JOINTS))
        lines.append(f"| {row['pos']} | {cells} |")
    lines += [
        "",
        "## 3. 指尖开口距离 (mm)",
        "",
        "| 电机位置 | 拇尖-食尖 | 拇尖-中尖 | 食尖-小尖(横向) |",
        "|---|---|---|---|",
    ]
    for r in rows_open:
        lines.append(f"| {r['pos']} | {r['d_thumb_index']*1000:.1f} | "
                     f"{r['d_thumb_middle']*1000:.1f} | {r['d_index_little']*1000:.1f} |")
    lines += ["", "## 4. 抓取类型闭合位姿（close 值）包络", ""]
    for tid, e in sorted(envelopes.items()):
        c = e["centroid"]
        lines.append(f"### Type {tid} — {e['name']}")
        lines.append("")
        lines.append(f"- 闭合电机向量: `{e['motors']}`")
        lines.append(f"- 包络中心（手坐标系）: [{c[0]*1000:.1f}, {c[1]*1000:.1f}, {c[2]*1000:.1f}] mm"
                     f"（距掌原点沿 +X {e['reach_x']*1000:.1f} mm）")
        for name, d in e["openings"]:
            lines.append(f"- 闭合开口 {name}: {d*1000:.1f} mm")
        lines.append("")
    lines += [
        "## 使用说明",
        "",
        "- **ID6 方向已于 2026-09-06 实机修正**：p 增大→内旋加深（p=2000 贴掌, p=0 外展）。",
        "  hand_bridge.py 旧注释（2000=垂直掌心）方向写反，仅注释有误，PRESHAPE_LIBRARY 数值不受影响。",
        "- `hand_mount_z` 初值维持 config 现值（0.09，已含安装补偿），由 jog 标定闭环修正；",
        "  本表第 4 节的\"距掌原点沿 +X\"是手内几何量，供将来管线改造使用。",
        "- 阶段 4 若发现预塑形开度不对：`python scripts/tools/compute_hand_geometry.py --render 1000`",
        "  生成渲染图，与同电机位置实机照片比对，把偏差补进第 2 节映射表。",
        "- `hand_bridge.py` 的 PRESHAPE_LIBRARY 若有调整，重跑本工具刷新第 4 节。",
    ]
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


joint_idx_global = None

if __name__ == "__main__":
    main()
