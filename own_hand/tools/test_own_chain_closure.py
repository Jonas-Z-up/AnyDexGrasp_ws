#!/usr/bin/env python3
"""B4 离线变换链闭合验证 — 自研手锚定 → JSON 契约 → 执行器 compute_grasp_target。

合成相机系场景(物体 + 竖直向下两指抓取) → own_hand_grasp 锚定 → 按 preview 的
own 分支组装 result dict → JSON 序列化往返 → 走执行器真实代码路径 → 与独立
复算的期望法兰位姿对比。

判据(Debug):
  1) 法兰位置/朝向与独立复算一致 (<1e-6, 验证字段管道与参数选择)
  2) 法兰接近轴 = 自研手 base +X 变换结果 (mount_yaw 绕 X 不改接近轴)
  3) 包络中心经执行链精确落在物体上 (<1e-6, 核心语义)
  注: mount_yaw_own=-180 与实物的符合性由预览渲染/真机验证, 本测试只保证链内自洽。

用法: python own_hand/tools/test_own_chain_closure.py
"""

import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(WS_ROOT, "own_hand"))         # own_hand_grasp
sys.path.insert(0, os.path.join(WS_ROOT, "scripts"))          # grasp_common

import grasp_common as gc          # noqa: E402
import own_hand_grasp as ohg       # noqa: E402


class _G:
    pass


def main():
    # ── 1. 相机系合成场景 ──
    # 约定(graspnetAPI, 已用真实JSON核实): 相机+Z=视线方向(朝下),
    # 抓取X=接近方向 指向物体(沿视线), 掌心/base 在抓取点 -X 侧(机器人一侧)。
    p_obj_cam = np.array([0.02, -0.05, 0.42])
    x_g = np.array([0.0, 0.0, 1.0])           # 接近: 沿视线指向物体
    y_g = np.array([1.0, 0.0, 0.0])           # 开合: 相机 +X
    z_g = np.cross(x_g, y_g)
    R_g = np.column_stack([x_g, y_g, z_g])
    g = _G()
    g.translation = p_obj_cam
    g.rotation_matrix = R_g
    grasp_type = 5

    t_base_cam, R_base_cam = ohg.base_pose_from_grasp(g, grasp_type)
    center, u = ohg.envelope_data(grasp_type)

    # ── 2. 组 result dict (同 preview own 分支) + JSON 往返 ──
    result = {
        "score": 1.0,
        "translation": t_base_cam.tolist(),
        "rotation_matrix": R_base_cam.tolist(),
        "depth": 0.02,
        "width": 0.06,
        "grasp_type": grasp_type,
        "hand_model": "own",
        "two_finger_translation": p_obj_cam.tolist(),
        "two_finger_rotation_matrix": R_g.tolist(),
    }
    result = json.loads(json.dumps(result))   # 序列化往返: 抓 list/shape 问题

    # ── 3. 执行器真实代码路径 (手参数 None → config → own 组) ──
    with open(os.path.join(WS_ROOT, "calibration", "cam2flange_current.json")) as f:
        calib = json.load(f)
    R_c2f = np.array(calib["R_cam2flange"], dtype=float)
    t_c2f = np.array(calib["t_cam2flange"], dtype=float)
    flange = np.array(gc.OBSERVE_RPYXYZ, dtype=float)

    hand_cfg = gc.load_hand_config(model="own")
    assert hand_cfg["hand_model"] == "own", "config 未切到 own?"
    target = gc.compute_grasp_target(result, R_c2f, t_c2f, flange,
                                     fixed_rpy=gc.FIXED_RPY)

    # ── 4. 独立复算期望值 (不经 compute_grasp_target) ──
    R_f2b = Rotation.from_euler("xyz", flange[:3]).as_matrix()
    t_f2b = flange[3:6]
    R_base_b = R_f2b @ R_c2f @ R_base_cam
    p_base_b = R_f2b @ (R_c2f @ t_base_cam + t_c2f) + t_f2b
    hz = hand_cfg["hand_mount_z"]
    p_flange_exp = p_base_b - hz * R_base_b[:, 0]
    R_flange_exp = R_base_b @ gc.compute_R_mount(hand_cfg["mount_yaw_deg"])

    # ── 5. 三项判据 ──
    ok = True

    err_p = float(np.linalg.norm(np.asarray(target["p_flange"]) - p_flange_exp))
    R_got = Rotation.from_euler("xyz", target["rpy"]).as_matrix()
    err_R_deg = float(np.degrees(
        Rotation.from_matrix(R_got @ R_flange_exp.T).magnitude()))
    c1 = err_p < 1e-6 and err_R_deg < 1e-4
    ok &= c1
    print(f"[判据1] 法兰位姿一致性: Δp={err_p*1000:.6f}mm (<0.001)  "
          f"ΔR={err_R_deg:.6f}° (<0.0001)  {'✅' if c1 else '❌'}")

    approach_got = R_got[:, 0]
    approach_exp = R_base_b[:, 0]
    ang = float(np.degrees(np.arccos(np.clip(
        np.dot(approach_got, approach_exp), -1, 1))))
    c2 = ang < 1e-4
    ok &= c2
    print(f"[判据2] 接近轴一致(mount_yaw绕X不改接近轴): {ang:.6f}°  {'✅' if c2 else '❌'}")

    # 包络中心语义: p_obj_base = p_palm_base + R_palm_base @ center
    p_obj_b = R_f2b @ (R_c2f @ p_obj_cam + t_c2f) + t_f2b
    R_palm_rec = R_got @ gc.compute_R_mount(-hand_cfg["mount_yaw_deg"])
    center_land = np.asarray(target["p_palm_base"]) + R_palm_rec @ center
    err_c = float(np.linalg.norm(center_land - p_obj_b))
    c3 = err_c < 1e-6
    ok &= c3
    print(f"[判据3] 包络中心落点: 误差={err_c*1000:.6f}mm (<0.001)  "
          f"{'✅' if c3 else '❌'}")

    print(f"\n[链路] used_network={target['used_network']} "
          f"deviation={target['deviation_deg']:.1f}° "
          f"(与固定朝向的闸门关系, 仅参考)")
    print(f"[数值] p_flange_base={np.round(target['p_flange'],4).tolist()}  "
          f"hz_own={hz}  mount_yaw_own={hand_cfg['mount_yaw_deg']}")
    print(f"\n{'✅ B4 全部判据通过' if ok else '❌ B4 存在失败项, 检查上述输出'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
