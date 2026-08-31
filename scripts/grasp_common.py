#!/usr/bin/env python3
"""AnyDexGrasp 眼在手上 — 手安装/朝向计算共享模块。

统一封装自研手在 Z1 法兰上的安装参数与「网络抓取位姿 → 法兰目标位姿」的计算，
执行器 anydex_eih_grasp.py (演示) / anydex_eih_grasp_calib.py (标定) 与预览
run_inspire_preview.py 共用，避免逻辑漂移。

坐标约定
--------
- 网络 palm 位姿（translation / rotation_matrix）在相机坐标系，来自
  run_inspire_preview.py 的 JSON 输出（translation = 手掌中心）。
- palm 帧 +X = 接近方向（指尖方向）。
- 真机确认：观测/抓取时指尖朝**前下方**（与相机同向），即手沿法兰 **+X**
  安装（眼在手上一体化，手在法兰末端最外侧）。
  故安装旋转 R_mount = Rx(α)：palm +X 映射到法兰 +X（同轴），Rx(α) 是绕
  接近轴的滚转角（拇指/掌心朝向，右手手型 α=90°，需空载验证）。
- 关键修正：执行器下发的是**法兰位姿**，而网络给的是**手掌位姿**。手装在法兰
  末端之外 hand_mount_z 处，故：
    指尖方向(基座系) = R_flange[:, 0]  （法兰 +X）
    法兰位置 = 掌心 − hand_mount_z × 指尖方向
"""

import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

# ── 默认安装参数（config/grasp_config.json 可覆盖）──────────────────
DEFAULT_HAND_MOUNT_Z = 0.12        # 法兰→掌心，沿接近轴 (m)
DEFAULT_HAND_FINGERTIP_Z = 0.24    # 法兰→指尖 (m)，用于间隙/碰撞估计
DEFAULT_MOUNT_YAW_DEG = -90.0      # 拇指/掌心绕接近轴的方位角 α（右手手型）
DEFAULT_DEV_THRESH_DEG = 90.0      # 网络朝向与固定朝向夹角阈值，超则回退
DEFAULT_OFFSET_RPY_DEG = [0.0, 0.0, 0.0]  # 基座系绕掌心旋转调试偏移（度）
DEFAULT_Z_COMP_KNOTS = []  # 基座系 z 高度补偿锚点 [[z, comp_z]…]（米）；空 = 不补偿

# ── 与执行器共用的抓取常量 ────────────────────────────────────────
FIXED_RPY = (0.0, 0.665, 0.0)     # 固定抓取朝向（兜底）
OBSERVE_RPYXYZ = (0.0, 1.057, 0.0, 0.3, 0.0, 0.35)  # 拍照/推理位姿


def default_config_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "grasp_config.json")


def load_hand_config(path=None):
    """从 config/grasp_config.json 读取手安装参数，缺省回退默认值。"""
    cfg = {}
    if path is None:
        path = default_config_path()
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
    return {
        "hand_mount_z": float(cfg.get("hand_mount_z", DEFAULT_HAND_MOUNT_Z)),
        "hand_fingertip_z": float(cfg.get("hand_fingertip_z", DEFAULT_HAND_FINGERTIP_Z)),
        "mount_yaw_deg": float(cfg.get("mount_yaw_deg", DEFAULT_MOUNT_YAW_DEG)),
        "orient_deviation_thresh_deg": float(cfg.get(
            "orient_deviation_thresh_deg", DEFAULT_DEV_THRESH_DEG)),
    }


def load_offset_xyz(path=None):
    """从 config/grasp_config.json 读取基座系抓取位置偏移 offset_xyz，缺省 [0, 0, 0]。

    用于纯水平微调（如基座 +X 前移），不改变接近轴方向，也不改变 z。
    执行器/预览/验证共用，保证三处显示与真机一致。
    """
    if path is None:
        path = default_config_path()
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        offset = cfg.get("offset_xyz", [0.0, 0.0, 0.0])
    else:
        offset = [0.0, 0.0, 0.0]
    return [float(v) for v in offset]


def load_offset_rpy_deg(path=None):
    """从 config/grasp_config.json 读取基座系旋转调试偏移 offset_rpy_deg，缺省 [0, 0, 0]。

    语义 = offset_xyz 的旋转孪生：基座系、绕**掌心**中心旋转（掌心位置由 offset_xyz
    锚定，offset_rpy 只转朝向，法兰位置随 approach 绕掌心摆动 hand_mount_z·sinθ）。
    单位度，作用于最终执行的朝向（网络或固定回退都生效），不影响网络/固定朝向的
    偏差闸门。执行器/预览/验证共用，保证三处显示与真机一致。
    """
    if path is None:
        path = default_config_path()
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        offset = cfg.get("offset_rpy_deg", DEFAULT_OFFSET_RPY_DEG)
    else:
        offset = DEFAULT_OFFSET_RPY_DEG
    return [float(v) for v in offset]


def load_z_comp_knots(path=None):
    """从 config/grasp_config.json 读取 z 高度补偿锚点 [[z, comp_z], ...]（米）。

    每个锚点 = [命令法兰 z（补偿前）, z 方向补偿量]；分段线性、范围外 clamp。
    空/缺失 → []（不补偿）。执行器/预览/验证共用，保证三处一致。
    """
    if path is None:
        path = default_config_path()
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        knots = cfg.get("z_comp_knots", DEFAULT_Z_COMP_KNOTS)
    else:
        knots = DEFAULT_Z_COMP_KNOTS
    return [[float(a), float(b)] for a, b in knots]


def compute_z_comp(z, knots):
    """分段线性 z 补偿：键 = 命令法兰 z（补偿前），锚点间线性插值，范围外取端点值（clamp）。

    纯函数，便于离线单测。knots: [[z_i, comp_i], ...]，要求 z 列按升序（np.interp 前提，
    不静默排序——异常配置应尽早报错）。空 / None / 少于 2 个锚点 → 0.0。
    补偿是单次修正，不是不动点迭代：键始终用补偿前的 z。
    """
    if not knots or len(knots) < 2:
        return 0.0
    zs = np.asarray([float(k[0]) for k in knots], dtype=float)
    cs = np.asarray([float(k[1]) for k in knots], dtype=float)
    return float(np.interp(float(z), zs, cs))


def compute_R_mount(mount_yaw_deg):
    """palm 帧 → 法兰帧 的固定安装旋转 R_mount = Rx(α)。

    手沿法兰 +X 安装（指尖前下方，与相机同向），故 palm +X（接近/指尖）直接
    映射到法兰 +X；Rx(α) 是绕接近轴的滚转角（拇指/掌心朝向，α=90° 右手手型，
    需一次空载验证确定正负）。
    """
    a = np.deg2rad(mount_yaw_deg)
    return Rotation.from_euler("x", a).as_matrix()


def compute_grasp_target(grasp, R_c2f, t_c2f, flange_pose,
                         hand_mount_z=None, mount_yaw_deg=None,
                         dev_thresh_deg=None, offset_xyz=None,
                         offset_rpy_deg=None, z_comp_knots=None,
                         fixed_rpy=FIXED_RPY,
                         force_fixed=False):
    """网络抓取位姿（相机系）→ 法兰目标位姿（基座系）。

    纯函数，便于离线单测。返回：
        rpy         (3,) 法兰目标朝向 [roll, pitch, yaw]
        p_flange    (3,) 法兰目标位置（已补偿安装偏移 + offset_xyz + z 高度补偿，基座系）
        used_network bool 是否采用网络朝向（False = 回退固定朝向）
        deviation_deg 网络朝向与固定朝向的夹角（°）
        p_palm_base (3,) 掌心位置（基座系，已含 offset_xyz 手动偏移与 z 高度补偿）
        z_comp      float 本次应用的基座系 z 高度补偿量（米，键=补偿前命令法兰 z）

    offset_rpy_deg : (3,) 基座系旋转调试偏移（度，绕掌心中心），作用于最终执行的
        朝向（网络或固定回退都生效）；在偏差闸门（网络 vs 固定）判定**之后**应用，
        故 deviation_deg 始终是 offset 前的原始值。零偏移完全跳过，逐字节等价旧行为。

    z_comp_knots : [[z, comp_z], ...] 分段线性 z 高度补偿（键=补偿前命令法兰 z，
        范围外 clamp，只动 z）。None → 读 config；空 → 不补偿，逐字节等价旧行为。
        与 p_palm_base 同移 Δz（整手刚性平移），不触碰 rpy/approach/deviation。
    """
    if hand_mount_z is None or mount_yaw_deg is None or dev_thresh_deg is None:
        cfg = load_hand_config()
        hand_mount_z = hand_mount_z if hand_mount_z is not None else cfg["hand_mount_z"]
        mount_yaw_deg = mount_yaw_deg if mount_yaw_deg is not None else cfg["mount_yaw_deg"]
        dev_thresh_deg = dev_thresh_deg if dev_thresh_deg is not None else cfg["orient_deviation_thresh_deg"]
    if offset_xyz is None:
        offset_xyz = load_offset_xyz()
    if offset_rpy_deg is None:
        offset_rpy_deg = load_offset_rpy_deg()
    if z_comp_knots is None:
        z_comp_knots = load_z_comp_knots()

    p_palm_cam = np.asarray(grasp["translation"], dtype=float)
    R_f2b = Rotation.from_euler("xyz", flange_pose[:3]).as_matrix()
    t_f2b = np.asarray(flange_pose[3:6], dtype=float)

    # 掌心位姿：相机 → 法兰 → 基座
    p_palm_base = R_f2b @ (R_c2f @ p_palm_cam + t_c2f) + t_f2b

    # 基座系手动偏移（config offset_xyz，如纯水平 +X 微调）：整只手平移，z 不变。
    # 加在 p_palm_base 上，p_flange/指尖随动，三处（执行/预览/验证）保持一致。
    p_palm_base = p_palm_base + np.asarray(offset_xyz, dtype=float)

    R_mount = compute_R_mount(mount_yaw_deg)
    R_fixed = Rotation.from_euler("xyz", fixed_rpy).as_matrix()

    # 决定朝向（偏差闸门用 offset 前的原始 R_flange，offset_rpy 不影响"是否信任网络"）
    used_network = False
    deviation_deg = 0.0
    if not force_fixed and "rotation_matrix" in grasp:
        R_palm = np.asarray(grasp["rotation_matrix"], dtype=float).reshape(3, 3)
        R_flange = R_f2b @ R_c2f @ R_palm @ R_mount
        deviation_deg = float(
            np.abs(Rotation.from_matrix(R_flange @ R_fixed.T).magnitude()) * 180.0 / np.pi)
        used_network = deviation_deg <= dev_thresh_deg
        if used_network:
            rpy = Rotation.from_matrix(R_flange).as_euler("xyz")
            approach = R_flange[:, 0]      # 指尖方向 = 法兰 +X
            R_sel = R_flange
        else:
            rpy = np.asarray(fixed_rpy, dtype=float)
            approach = R_fixed[:, 0]
            R_sel = R_fixed
    else:
        rpy = np.asarray(fixed_rpy, dtype=float)
        approach = R_fixed[:, 0]
        R_sel = R_fixed

    # 基座系旋转调试偏移（config offset_rpy_deg，度，绕掌心中心）：基座系左乘，
    # 改朝向不改掌心位置 p_palm_base；approach 随之旋转 ⇒ 法兰位置绕掌心摆动。
    # 零偏移跳过 ⇒ 与旧行为逐字节等价（避免固定朝向 rpy 矩阵往返 ~1e-15 误差）。
    if not np.allclose(offset_rpy_deg, 0):
        R_off = Rotation.from_euler("xyz", np.deg2rad(offset_rpy_deg)).as_matrix()
        R_sel = R_off @ R_sel
        rpy = Rotation.from_matrix(R_sel).as_euler("xyz")
        approach = R_sel[:, 0]             # 指尖方向 = 法兰 +X（已含 offset）

    # 安装偏移补偿：法兰位置 = 掌心 − hand_mount_z × 接近方向
    p_flange = p_palm_base - hand_mount_z * approach

    # 基座系 z 方向高度补偿（config z_comp_knots，分段线性，键 = 命令法兰 z = 补偿前）：
    # 吸收深度/高度相关的经验误差（R_c2f 手眼旋转残差的 z 项）。p_palm_base 同步平移
    # 保证手刚性（p_flange − p_palm_base = −hand_mount_z·approach 不变），不触碰
    # rpy/approach/deviation。空锚点 → dz=0 → 逐字节等价旧行为。
    dz = compute_z_comp(p_flange[2], z_comp_knots)
    if dz != 0.0:
        p_flange[2] += dz
        p_palm_base[2] += dz

    return {
        "rpy": rpy,
        "p_flange": p_flange,
        "used_network": used_network,
        "deviation_deg": float(deviation_deg),
        "p_palm_base": p_palm_base,
        "z_comp": float(dz),
    }
