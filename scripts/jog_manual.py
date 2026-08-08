#!/usr/bin/env python3
"""抓取偏差标定 — 人工微调 + 记录。

集成进 anydex_eih_grasp_calib.py (--calib-jog):到达目标抓取点后进入键盘微调循环,
操作员用 w/s/a/d/q/e 在基座系 XYZ 微调手臂对准物体,按空格记录调整前后的
位姿与偏移(JSONL 历史),供 analyze_jog_history.py 分析系统性偏差。

本模块不依赖手控制,可单独以 --dry-run 跑键盘自测(不动机械臂、不写文件)。
"""

import argparse
import json
import os
import select
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

# R/F → 绕法兰中心俯仰:直接改命令 rpy 的 pitch 分量(下标1),xyz 不动。
# +pitch = 指尖下压(接近方向下转), −pitch = 指尖上抬。方向反了改这里即可。
PITCH_KEYMAP = {"r": +1, "f": -1}
PITCH_STEP_DEG_DEFAULT = 1.0   # 俯仰步长 (°),--jog-pitch-step-deg / --pitch-step-deg 可覆盖

STEP_SIZES_MM = [2.0, 5.0, 10.0]

# 可打断下降:步进 MoveJ 下降的默认步长 (mm) 与按键轮询间隔 (s)
DESCEND_STEP_MM_DEFAULT = 10.0
DESCEND_POLL_S = 0.05
# 空格/Esc 暂停下降;Ctrl+C 不在此列——cbreak 保留 ISIG,会以 SIGINT 中断程序(编排器
# except KeyboardInterrupt → 回零下电)。轮询分支对 \x03 字节另有兜底 raise。
DESCEND_INTERRUPT_KEYS = ("\x1b", " ")


def _step_index(step):
    """把任意步长映射到 STEP_SIZES_MM 里最近的下标(防 --jog-step-mm 传非标准值)。"""
    return min(range(len(STEP_SIZES_MM)), key=lambda i: abs(STEP_SIZES_MM[i] - step))

HELP = """\
  人工微调 (基座系平移, 保持朝向):
    w/s  前/后 (+X/−X)    a/d  左/右 (+Y/−Y)    q/e  上/下 (+Z/−Z)
    r/f  俯仰 +/− (绕法兰中心改 pitch: r=指尖下压, f=指尖上抬)
    空格 记录本次调整并继续抓取    回车 有微调会补记/没微调则跳过
    [+]/[-] 步长切换 (2/5/10 mm)    [?]/[h] 本帮助
    Esc / Ctrl+C 中止 (有调整会补记一条)
  注: config offset_rpy_deg 绕**掌心**中心旋转(朝向调试), 与这里 r/f 绕**法兰**中心
     不同: 小角度下角度效果相同, 区别在法兰位置摆动(offset 摆, jog 不摆)。
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
            "offset_rpy_deg": [float(v) for v in cfg.get("offset_rpy_deg", [0.0, 0.0, 0.0])],
            "orient_deviation_thresh_deg": cfg.get("orient_deviation_thresh_deg"),
        },
    }


def write_record(history_path, rec):
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    off = rec["offset"]["xyz"]
    msg = f"  [记录] 已写入 {history_path}  offset.xyz=" \
          f"[{', '.join(f'{v*1000:+.0f}mm' for v in off)}]"
    off_rpy = rec["offset"]["rpy"]
    if any(abs(v) > 1e-6 for v in off_rpy):
        msg += f"  offset.rpy=[{', '.join(f'{v*180.0/np.pi:+.1f}°' for v in off_rpy)}]"
    print(msg, flush=True)


def descend_interruptible(z1, rpy, xyz_start, xyz_end,
                          step_mm=DESCEND_STEP_MM_DEFAULT,
                          interrupt_keys=DESCEND_INTERRUPT_KEYS,
                          poll_s=DESCEND_POLL_S, dry_run=False):
    """从 xyz_start 沿直线步进 MoveJ 下降到 xyz_end (保持 rpy),期间轮询键盘。

    参数
    ----
    z1 : Z1Server 或 None(dry_run 时 None)
    rpy : (3,) 保持的朝向
    xyz_start / xyz_end : (3,) 起止位置(一般只差 z)
    step_mm : 每步下降高度 (mm)
    interrupt_keys : 命中即中断下降的按键集合 (默认 Esc/Space/Ctrl+C)

    返回 (interrupted, pose):
      interrupted=True  → 人工按下中断键, pose = 当前已到达的命令位姿(可作接管基准)
      interrupted=False → 走完下降, pose = end_pose
    dry_run=True 只打印,不移动机械臂。
    """
    start = np.concatenate([_arr(rpy), _arr(xyz_start)])
    end = np.concatenate([_arr(rpy), _arr(xyz_end)])
    dist = float(np.linalg.norm(end[3:6] - start[3:6]))
    step = float(step_mm) / 1000.0
    n = max(int(np.ceil(dist / step)), 1)

    fd = sys.stdin.fileno()
    is_tty = sys.stdin.isatty()
    old = None
    if is_tty:
        # cbreak 式: 关 ICANON|ECHO,保留 OPOST → 后台 Z1 打印仍正常换行不错乱
        old = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        new[3] &= ~(termios.ICANON | termios.ECHO)
        termios.tcsetattr(fd, termios.TCSADRAIN, new)
    try:
        for i in range(1, n + 1):
            frac = i / n
            pose = start + (end - start) * frac
            if not dry_run:
                z1.movej(pose)
            else:
                print(f"  [descend] z={pose[5]:.3f}", flush=True)
            if is_tty:
                ready, _, _ = select.select([sys.stdin], [], [], poll_s)
                if ready:
                    ch = sys.stdin.read(1)
                    if ch == "\x03":
                        # 正常情况 cbreak 保留 ISIG,Ctrl+C 以 SIGINT 到达并被编排器
                        # except KeyboardInterrupt 捕获→回零;这里兜底:若终端模式被改
                        # 导致 \x03 以字节到达,仍按"中断程序"处理,绝不当作暂停。
                        raise KeyboardInterrupt
                    if ch in interrupt_keys:
                        print(f"  [下降] 已中断(键 {ch!r}),停在 z={pose[5]:.3f}",
                              flush=True)
                        return True, pose
    finally:
        if old is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    print(f"  [下降] 完成,z={end[5]:.3f}", flush=True)
    return False, end


def jog_manual_stage(z1, target_pose, grasp_meta, cfg,
                     history_path, step_mm=5.0,
                     pitch_step_deg=PITCH_STEP_DEG_DEFAULT,
                     record=True, dry_run=False):
    """到达目标抓取点后的人工微调循环。

    参数
    ----
    z1 : Z1Server 或 None(dry_run 时 None)
    target_pose : (6,) 命令抓取位姿 [r, p, y, x, y, z](基座系)
    grasp_meta : dict,含 grasp_json/grasp_type/score/p_cam/used_network 等
    cfg : dict,含 hand_mount_z/mount_yaw_deg/offset_xyz/offset_rpy_deg 等(写入记录供分析)
    pitch_step_deg : 俯仰步长 (°),r/f 每按一次改 base[1] 这么多度
    record : False 时不写标定历史(供"下降被打断→人工接管"用,避免污染标定数据)

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
    print(f"  初始目标: {_pose_dict(target_pose)['xyz']}  步长={step:.0f}mm  "
          f"俯仰步长={pitch_step_deg:.1f}°\n", flush=True)

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

            elif ch in PITCH_KEYMAP:
                sign = PITCH_KEYMAP[ch]
                delta = sign * pitch_step_deg * np.pi / 180.0   # 度 → 弧度
                base[1] += delta                                # 绕法兰中心改 pitch,xyz 不动
                jogged = True
                if dry_run:
                    print(f"  [jog] pitch {'+' if sign > 0 else '-'}{pitch_step_deg:.1f}°  →  "
                          f"rpy=[{base[0]:.3f}, {base[1]:.3f}, {base[2]:.3f}]", flush=True)
                else:
                    z1.movej(base)
                    time.sleep(0.15)           # 等伺服收敛
                    actual = z1.getpose()
                    act_deg = (actual[1] - prev_actual[1]) * 180.0 / np.pi
                    prev_actual = actual
                    print(f"  [jog] pitch {'+' if sign > 0 else '-'}{pitch_step_deg:.1f}°  →  "
                          f"命令pitch={base[1] * 180.0 / np.pi:+.2f}°  "
                          f"实际Δ={act_deg:+.2f}°", flush=True)

            elif ch == " ":
                after = base.copy()            # 记录命令系最终位姿(offset=操作员调整量)
                if record:
                    rec = build_record(target_pose, before, after, grasp_meta, cfg)
                    if dry_run:
                        print("  [dry-run] 将写入记录(此处仅预览):")
                        print(json.dumps(rec, ensure_ascii=False, indent=2), flush=True)
                    else:
                        write_record(history_path, rec)
                    recorded = True
                    print("  [记录] 已记录,继续抓取。", flush=True)
                else:
                    print("  [接管] 位置已确认,继续。", flush=True)
                return "continue"

            elif ch in ("\r", "\n"):
                # 回车在有微调时也补记:抓取流程到处都是"回车确认",操作员在标定
                # 提示下容易顺手按回车,导致微调数据被静默丢弃(实测发生过)。
                # 只有"完全没动过"的回车才跳过,避免产生全零偏移的脏样本。
                if record and jogged and not recorded:
                    after = base.copy()
                    rec = build_record(target_pose, before, after, grasp_meta, cfg)
                    if dry_run:
                        print("  [dry-run] 回车(有微调)将补记,此处仅预览:")
                        print(json.dumps(rec, ensure_ascii=False, indent=2), flush=True)
                    else:
                        write_record(history_path, rec)
                    recorded = True
                    print("  [记录] 回车确认(检测到微调),已补记写入历史。", flush=True)
                else:
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
                if record and jogged and not recorded and not dry_run:
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
    ap.add_argument("--step-mm", type=float, default=5.0, help="初始平移步长 (mm)")
    ap.add_argument("--pitch-step-deg", type=float, default=PITCH_STEP_DEG_DEFAULT,
                    help="俯仰步长 (度, r/f 每按一次)")
    ap.add_argument("--descend-step-mm", type=float, default=DESCEND_STEP_MM_DEFAULT,
                    help="可打断下降的每步高度 (mm)")
    ap.add_argument("--descend-dry-run", action="store_true",
                    help="下降自测(不动机械臂):打印每步 z,按 Esc/Space 可验证中断")
    args = ap.parse_args()

    if args.descend_dry_run:
        print("下降自测(dry-run, 不动机械臂)。按 Esc/Space 验证中断。")
        interrupted, pose = descend_interruptible(
            z1=None,
            rpy=np.array([0.0, 0.665, 0.0]),
            xyz_start=np.array([0.30, 0.0, 0.52]),
            xyz_end=np.array([0.30, 0.0, 0.40]),
            step_mm=args.descend_step_mm, dry_run=True)
        print(f"  结果: interrupted={interrupted} pose={pose.round(3).tolist()}")
        return 0

    if not args.dry_run:
        ap.error("本脚本仅支持 --dry-run / --descend-dry-run;真机微调请通过 anydex_eih_grasp_calib.py --calib-jog 使用。")

    print("键盘自测(dry-run, 不动机械臂)。")
    jog_manual_stage(z1=None, target_pose=np.array([0.0, 0.665, 0.0, 0.30, 0.0, 0.40]),
                     grasp_meta={"grasp_type": 2, "score": 1.0,
                                 "p_cam": [0.07, 0.04, 0.12], "depth_cam": 0.12,
                                 "used_network": True, "deviation_deg": 10.0},
                     cfg={"hand_mount_z": 0.09, "hand_fingertip_z": 0.24,
                          "mount_yaw_deg": -90.0, "offset_xyz": [0.1, 0.0, 0.0],
                          "offset_rpy_deg": [0.0, 0.0, 0.0],
                          "orient_deviation_thresh_deg": 90.0},
                     history_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "..", "logs", "jog_calib_history.jsonl"),
                     step_mm=args.step_mm,
                     pitch_step_deg=args.pitch_step_deg, dry_run=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
