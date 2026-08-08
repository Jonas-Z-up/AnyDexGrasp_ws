#!/usr/bin/env python3
"""自研手抓取朝向/安装补偿 — 真机验证脚本。

安全优先：默认不移动机械臂。只有显式加 --move 才会把手臂移到固定朝向
检查位姿（安全高度 z=0.40），用于目视核对指尖/掌心/拇指方向。

用法（conda activate anydex 后）:
  python validate_grasp_setup.py                 # 只打印参数与方向（不移动）
  python validate_grasp_setup.py --target        # + 计算最新 grasp 的目标位姿
  python validate_grasp_setup.py --move          # + 移臂到检查位姿（需回车确认）
  python validate_grasp_setup.py --move --target # 两个都做
"""

import json
import os
import sys
import time

# 让 anydex_eih_grasp 导入时跳过 hand_bridge（避免连手串口）
_user_args = list(sys.argv[1:])
sys.argv = ["validate_grasp_setup.py", "--no-hand"] + _user_args

import numpy as np
from scipy.spatial.transform import Rotation

import grasp_common as gc
from anydex_eih_grasp import Z1Server

sys.argv = ["validate_grasp_setup.py"] + _user_args
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.join(SCRIPT_DIR, "..")
DEFAULT_GRASP_JSON = os.path.join(WS_ROOT, "logs", "preview", "latest_inspire_grasp.json")
DEFAULT_CALIB = os.path.join(WS_ROOT, "calibration", "cam2flange_current.json")

# 检查位姿：固定朝向 + 安全高度（不执行抓取）
INSPECT_POSE = (0.0, 0.665, 0.0, 0.30, 0.0, 0.40)

# Z1 安全范围（与 MoveZ1.py 一致）
SAFE_Z = (0.12, 0.70)


def print_params():
    cfg = gc.load_hand_config()
    R_mount = gc.compute_R_mount(cfg["mount_yaw_deg"])
    print("=" * 64)
    print(" 自研手抓取设置 — 参数与方向")
    print("=" * 64)
    print("手安装参数 (config/grasp_config.json):")
    for k, v in cfg.items():
        print(f"  {k:32s} = {v}")
    print()
    print("坐标系约定（已用 SDK 探针 + 你的确认校准）:")
    print("  · 手沿法兰 +X 安装：指尖朝前下方，与相机拍摄方向一致")
    print("  · R_mount = Rx(α)：palm +X(接近/指尖) → 法兰 +X")
    print(f"  · R_mount(α={cfg['mount_yaw_deg']}°) 列映射:")
    print(f"      palm +X → 法兰 {R_mount[:, 0].round(3)}  (应≈[1,0,0])")
    print(f"      palm +Y → 法兰 {R_mount[:, 1].round(3)}")
    print(f"      palm +Z → 法兰 {R_mount[:, 2].round(3)}")
    print("  · 指尖方向(基座系) = 法兰 +X 首列")
    print("  · 法兰位置 = 掌心 − hand_mount_z × 指尖方向")
    print("  · 固定朝向(兜底) FIXED_RPY =", gc.FIXED_RPY)
    print("  · offset_xyz =", gc.load_offset_xyz(), "  (基座系平移, m)")
    print("  · offset_rpy_deg =", gc.load_offset_rpy_deg(), "  (基座系绕掌心旋转, °)")
    print()
    print("静态检查清单（移到检查位姿后目视核对）:")
    print("  [1] 指尖朝前下方（朝向桌面）")
    print("  [2] 手掌平面大致朝下（对着桌面）")
    print("  [3] 手无倒装/反装感：掌心朝下+右手手型时，手指弯曲方向应能自然包住桌面物体")
    print("=" * 64)


def compute_and_print_target():
    if not os.path.exists(DEFAULT_GRASP_JSON):
        print(f"[跳过] 未找到 grasp JSON: {DEFAULT_GRASP_JSON}")
        print("       先运行 run_inspire_preview.py 生成抓取结果。")
        return
    if not os.path.exists(DEFAULT_CALIB):
        print(f"[跳过] 未找到标定: {DEFAULT_CALIB}")
        return

    with open(DEFAULT_GRASP_JSON) as f:
        grasp = json.load(f)
    with open(DEFAULT_CALIB) as f:
        calib = json.load(f)
    R_c2f = np.array(calib["R_cam2flange"], dtype=float)
    t_c2f = np.array(calib["t_cam2flange"], dtype=float)

    # 执行器在 Phase 2 时手臂停在观测位姿，故用观测位姿作为法兰（与执行器一致）
    flange = np.array(gc.OBSERVE_RPYXYZ, dtype=float)

    cfg = gc.load_hand_config()
    t = gc.compute_grasp_target(grasp, R_c2f, t_c2f, flange,
                                cfg["hand_mount_z"], cfg["mount_yaw_deg"],
                                cfg["orient_deviation_thresh_deg"])

    Rf = Rotation.from_euler("xyz", t["rpy"]).as_matrix()
    approach = Rf[:, 0]                                # 指尖方向 = 法兰 +X
    tip = t["p_palm_base"] + (cfg["hand_fingertip_z"] - cfg["hand_mount_z"]) * approach

    print("=" * 64)
    print(" 目标位姿（最新 grasp JSON）— 对照执行器 Phase 2 会用到的值")
    print("=" * 64)
    print(f" 采用网络朝向: {t['used_network']}"
          f"   (偏差 {t['deviation_deg']:.1f}°, 阈值 {cfg['orient_deviation_thresh_deg']}°)")
    print(f" 目标 rpy(deg)   = {np.rad2deg(t['rpy']).round(1)}")
    print(f" 掌心(基座系)    = {t['p_palm_base'].round(3)}")
    print(f" 法兰(基座系)    = {t['p_flange'].round(3)}")
    print(f" 指尖(基座系)    = {tip.round(3)}")
    print(f" 指尖方向(基座系) = {approach.round(3)}")
    z = t["p_flange"][2]
    ok = SAFE_Z[0] <= z <= SAFE_Z[1]
    print(f" 法兰 z={z:.3f} ∈ {SAFE_Z} ?  {'✅ 安全' if ok else '❌ 超出安全范围（MoveZ1 会拒绝）'}")
    print("=" * 64)


def move_to_inspect():
    print(f"[提示] 将把机械臂移到检查位姿: {INSPECT_POSE}")
    print("       (固定朝向 + 安全高度 z=0.40，不执行抓取)")
    try:
        input(">>> 确认手臂处于安全状态，按回车移动 (Ctrl+C 取消): ")
    except KeyboardInterrupt:
        print("\n已取消，不移动。")
        return

    try:
        z1 = Z1Server(speed=0.15)
    except Exception as e:
        print(f"❌ 无法连接 Z1 子进程: {e}")
        print("   请确认终端0的 ./z1_ctrl 已在运行。")
        return

    try:
        print("  MoveJ → 检查位姿...")
        z1.movej(INSPECT_POSE)
        time.sleep(1.0)   # 等机械臂完全停稳后再读实际位姿，避免伺服收敛误差
        flange = z1.getpose()
        print(f"  ✅ 已到达检查位姿")
        print(f"  目标  rpy=({INSPECT_POSE[0]},{INSPECT_POSE[1]},{INSPECT_POSE[2]}) "
              f"xyz=({INSPECT_POSE[3]},{INSPECT_POSE[4]},{INSPECT_POSE[5]})")
        print(f"  实际  rpy={flange.round(3)}")
        # 掌心/指尖偏移从 config 读取（避免与 hand_mount_z/hand_fingertip_z 改动脱节）
        cfg = gc.load_hand_config()
        mz, fz = cfg["hand_mount_z"], cfg["hand_fingertip_z"]
        Rf = Rotation.from_euler("xyz", flange[:3]).as_matrix()
        tip_dir = Rf[:, 0]
        palm = flange[3:6] + mz * tip_dir
        fingertip = flange[3:6] + fz * tip_dir
        # 预期值：检查位姿 + config 偏移，动态计算（不写死）
        R_exp = Rotation.from_euler("xyz", INSPECT_POSE[:3]).as_matrix()
        tip_dir_exp = R_exp[:, 0]
        palm_exp = np.asarray(INSPECT_POSE[3:6]) + mz * tip_dir_exp
        fingertip_exp = np.asarray(INSPECT_POSE[3:6]) + fz * tip_dir_exp
        print(f"  指尖方向(基座系) = {tip_dir.round(3)}  ← 应≈{tip_dir_exp.round(3)}")
        print(f"  掌心位置(基座系) = {palm.round(3)}   ← 应≈{palm_exp.round(3)}")
        print(f"  指尖位置(基座系) = {fingertip.round(3)}  ← 应≈{fingertip_exp.round(3)}")
        print()
        print("  目视检查清单：")
        print("    [1] 指尖朝前下方（朝向桌面）")
        print("    [2] 手掌平面大致朝下（对着桌面）")
        print("    [3] 手无倒装/反装感")
        print()
        print("  检查完按 Ctrl+C 退出（先回零再断开，避免机械臂停半空下垂）。")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n结束检查，先回零再断开...")
    finally:
        try:
            z1.home(timeout=30)   # 先回零位，再下电
        except BaseException as e:   # 含 KeyboardInterrupt：回零被中断也要继续断开
            print(f"[提示] 回零失败/被中断，跳过: {e}")
        z1.quit()


def main():
    ap = argparse.ArgumentParser(description="自研手抓取朝向/补偿 真机验证")
    ap.add_argument("--params", action="store_true", help="只打印参数与方向（默认行为）")
    ap.add_argument("--target", action="store_true", help="同时计算并打印最新 grasp 的目标位姿")
    ap.add_argument("--move", action="store_true", help="同时把手臂移到检查位姿（需回车确认）")
    args = ap.parse_args()

    do_params = args.params or not (args.target or args.move)
    if args.move:
        move_to_inspect()
    if do_params:
        print_params()
    if args.target:
        compute_and_print_target()


if __name__ == "__main__":
    main()
