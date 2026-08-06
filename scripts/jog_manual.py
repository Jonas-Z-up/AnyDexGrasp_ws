#!/usr/bin/env python3
"""抓取偏差标定 — 人工微调 + 记录。

集成进 anydex_eih_grasp.py (--calib-jog):到达目标抓取点后进入键盘微调循环,
操作员用 w/s/a/d/q/e 在基座系 XYZ 微调手臂对准物体,按空格记录调整前后的
位姿与偏移(JSONL 历史),供 analyze_jog_history.py 分析系统性偏差。

本模块不依赖手控制,可单独以 --dry-run 跑键盘自测(不动机械臂、不写文件)。
"""

import argparse
import json
import os
import sys
import termios
import time
import tty
from datetime import datetime

import numpy as np
from scipy.spatial.transform import Rotation

# ── 按键 → 基座系轴 (+方向) 映射 ─────────────────────────────
# 方向反了改这里即可:key → (xyz 下标, 符号)。微调只平移,保持 rpy 不变。
KEYMAP = {
    "w": (0, +1),   # +X (前)
    "s": (0, -1),   # -X (后)
    "a": (1, +1),   # +Y (左)
    "d": (1, -1),   # -Y (右)
    "q": (2, +1),   # +Z (上)
    "e": (2, -1),   # -Z (下)
}

STEP_SIZES_MM = [2.0, 5.0, 10.0]


def _step_index(step):
    """把任意步长映射到 STEP_SIZES_MM 里最近的下标(防 --jog-step-mm 传非标准值)。"""
    return min(range(len(STEP_SIZES_MM)), key=lambda i: abs(STEP_SIZES_MM[i] - step))

HELP = """\
  人工微调 (基座系平移, 保持朝向):
    w/s  前/后 (+X/−X)    a/d  左/右 (+Y/−Y)    q/e  上/下 (+Z/−Z)
    空格 记录本次调整并继续抓取    回车 不记录继续抓取
    [+]/[-] 步长切换 (2/5/10 mm)    [?]/[h] 本帮助
    Esc / Ctrl+C 中止 (有调整会补记一条)
"""


def getch():
    """读取单个按键(无需回车)。raw mode 下 Ctrl+C 是字节 \\x03。"""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def _arr(pose):
    return np.asarray(pose, dtype=float)


def _pose_dict(pose):
    p = _arr(pose)
    return {"rpy": [float(v) for v in p[:3]], "xyz": [float(v) for v in p[3:6]]}


def _approach_from_rpy(rpy):
    R = Rotation.from_euler("xyz", _arr(rpy)).as_matrix()
    return [float(v) for v in R[:, 0]]


def build_record(target_pose, before, after, grasp_meta, cfg):
    """组装一条标定样本 (JSON 可序列化 dict)。"""
    before = _arr(before)
    after = _arr(after)
    return {
        "ts": datetime.now().isoformat(),
        "grasp_json": grasp_meta.get("grasp_json"),
        "grasp_type": grasp_meta.get("grasp_type"),
        "score": grasp_meta.get("score"),
        "p_cam": [float(v) for v in _arr(grasp_meta.get("p_cam", [0.0, 0.0, 0.0]))],
        "depth_cam": float(grasp_meta.get("depth_cam", 0.0)),
        "used_network": bool(grasp_meta.get("used_network", True)),
        "deviation_deg": float(grasp_meta.get("deviation_deg", 0.0)),
        "target": _pose_dict(target_pose),
        "before": _pose_dict(before),
        "after": _pose_dict(after),
        "offset": {
            "rpy": [float(v) for v in (after[:3] - before[:3])],
            "xyz": [float(v) for v in (after[3:6] - before[3:6])],
        },
        "approach_base": _approach_from_rpy(after[:3]),
        "cfg": {
            "hand_mount_z": cfg.get("hand_mount_z"),
            "hand_fingertip_z": cfg.get("hand_fingertip_z"),
            "mount_yaw_deg": cfg.get("mount_yaw_deg"),
            "offset_xyz": [float(v) for v in cfg.get("offset_xyz", [0.0, 0.0, 0.0])],
            "orient_deviation_thresh_deg": cfg.get("orient_deviation_thresh_deg"),
        },
    }


def write_record(history_path, rec):
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    off = rec["offset"]["xyz"]
    print(f"  [记录] 已写入 {history_path}  offset.xyz=[{', '.join(f'{v*1000:+.0f}mm' for v in off)}]",
          flush=True)


def jog_manual_stage(z1, target_pose, grasp_meta, cfg,
                     history_path, step_mm=5.0, dry_run=False):
    """到达目标抓取点后的人工微调循环。

    参数
    ----
    z1 : Z1Server 或 None(dry_run 时 None)
    target_pose : (6,) 命令抓取位姿 [r, p, y, x, y, z](基座系)
    grasp_meta : dict,含 grasp_json/grasp_type/score/p_cam/used_network 等
    cfg : dict,含 hand_mount_z/mount_yaw_deg/offset_xyz 等(写入记录供分析)

    返回 "continue"(空格/回车:继续正常抓取)或 "abort"(Esc/Ctrl+C)。
    dry_run=True 只打印按键→命令,不移动机械臂、不写文件。
    """
    # 微调基准 = 命令抓取位姿 (MoveJ 同坐标系), 绝不拿 getpose 当基准:
    # 真机诊断显示 getpose 与实际存在 ~30-42mm 系统偏移(沿接近轴), 若拿它当基准,
    # 2mm 微调会被放大成数十 mm 的大幅运动。用命令目标当基准, 恒定偏移互相抵消,
    # 相邻两次 MoveJ 目标差 2mm → 手臂实际每步正好动 2mm。
    base = _arr(target_pose).copy()
    before = base.copy()                 # 记录用"调整前"(命令系)
    if not dry_run:
        time.sleep(0.5)                  # 等机械臂完全停稳
        actual0 = z1.getpose()
        prev_actual = actual0
        off0 = (actual0[3:6] - base[3:6]) * 1000.0
        print(f"  [诊断] getpose 实际 vs 命令抓取位姿: 差 {off0.round(1)} mm"
              f" (≠0 说明存在系统偏移; 微调以命令目标为基准, 不受影响)", flush=True)

    step = float(step_mm)
    jogged = False
    recorded = False
    print("\n" + "=" * 64)
    print(" 人工微调 — 对准物体后按空格记录 (Esc/Ctrl+C 中止)")
    print("=" * 64)
    print(HELP)
    print(f"  初始目标: {_pose_dict(target_pose)['xyz']}  步长={step:.0f}mm\n", flush=True)

    try:
        while True:
            ch = getch()

            if ch in KEYMAP:
                axis, sign = KEYMAP[ch]
                delta = sign * step / 1000.0   # step 单位 mm → 位姿数组单位 m
                cmd_mm = [0.0, 0.0, 0.0]
                cmd_mm[axis] = sign * step
                base[3 + axis] += delta        # 在命令系累加(基准始终是命令抓取位姿)
                if dry_run:
                    xyz = base[3:6]
                else:
                    z1.movej(base)             # MoveJ 绝对目标: 终点 IK 可靠(抓取管线验证过)
                    time.sleep(0.15)           # 等伺服收敛
                    actual = z1.getpose()
                    act_mm = (actual[3:6] - prev_actual[3:6]) * 1000.0   # 相邻两次实际位移
                    prev_actual = actual
                jogged = True
                if dry_run:
                    print(f"  [jog] {'+X' if axis == 0 else '+Y' if axis == 1 else '+Z'}"
                          f"{'+' if sign > 0 else '-'} {step:.0f}mm  →  "
                          f"xyz=[{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}]", flush=True)
                else:
                    print(f"  [jog] {'+X' if axis == 0 else '+Y' if axis == 1 else '+Z'}"
                          f"{'+' if sign > 0 else '-'} {step:.0f}mm  →  命令位移"
                          f"[{cmd_mm[0]:+.1f},{cmd_mm[1]:+.1f},{cmd_mm[2]:+.1f}]mm  实际位移"
                          f"[{act_mm[0]:+.1f},{act_mm[1]:+.1f},{act_mm[2]:+.1f}]mm", flush=True)

            elif ch == " ":
                after = base.copy()            # 记录命令系最终位姿(offset=操作员调整量)
                rec = build_record(target_pose, before, after, grasp_meta, cfg)
                if dry_run:
                    print("  [dry-run] 将写入记录(此处仅预览):")
                    print(json.dumps(rec, ensure_ascii=False, indent=2), flush=True)
                else:
                    write_record(history_path, rec)
                recorded = True
                print("  [记录] 已记录,继续抓取。", flush=True)
                return "continue"

            elif ch in ("\r", "\n"):
                print("  [跳过] 不记录,继续抓取。", flush=True)
                return "continue"

            elif ch in ("+", "]"):
                idx = min(_step_index(step) + 1, len(STEP_SIZES_MM) - 1)
                step = STEP_SIZES_MM[idx]
                print(f"  [步长] → {step:.0f}mm", flush=True)

            elif ch in ("-", "["):
                idx = max(_step_index(step) - 1, 0)
                step = STEP_SIZES_MM[idx]
                print(f"  [步长] → {step:.0f}mm", flush=True)

            elif ch in ("?", "h"):
                print(HELP, flush=True)

            elif ch in ("\x1b", "\x03", "\x04"):
                if jogged and not recorded and not dry_run:
                    after = base.copy()
                    rec = build_record(target_pose, before, after, grasp_meta, cfg)
                    write_record(history_path, rec)
                print("  [中止] 已取消(有调整已补记)。", flush=True)
                return "abort"

            else:
                # 忽略其它键;显示帮助提示一次
                print(f"  [未绑定] '{ch!r}'  (? 帮助)", flush=True)

    except KeyboardInterrupt:
        # raw mode 下 Ctrl+C 一般以 \x03 字节到达;这里兜底。
        print("\n  [中止] 已取消。", flush=True)
        return "abort"


def main():
    ap = argparse.ArgumentParser(
        description="抓取偏差标定 — 键盘微调自测(--dry-run, 不动机械臂)")
    ap.add_argument("--dry-run", action="store_true",
                    help="纯键盘自测:打印按键→命令,不移动机械臂、不写文件")
    ap.add_argument("--step-mm", type=float, default=5.0, help="初始步长 (mm)")
    args = ap.parse_args()

    if not args.dry_run:
        ap.error("本脚本仅支持 --dry-run;真机微调请通过 anydex_eih_grasp.py --calib-jog 使用。")

    print("键盘自测(dry-run, 不动机械臂)。")
    jog_manual_stage(z1=None, target_pose=np.array([0.0, 0.665, 0.0, 0.30, 0.0, 0.40]),
                     grasp_meta={"grasp_type": 2, "score": 1.0,
                                 "p_cam": [0.07, 0.04, 0.12], "depth_cam": 0.12,
                                 "used_network": True, "deviation_deg": 10.0},
                     cfg={"hand_mount_z": 0.09, "hand_fingertip_z": 0.24,
                          "mount_yaw_deg": -90.0, "offset_xyz": [0.1, 0.0, 0.0],
                          "orient_deviation_thresh_deg": 90.0},
                     history_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "..", "logs", "jog_calib_history.jsonl"),
                     step_mm=args.step_mm, dry_run=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
