#!/usr/bin/env python3
"""AnyDexGrasp 眼在手上抓取编排脚本 (终端A)。

保持 Z1 常驻连接，分两阶段执行：
  Phase 1: 移动到观测位姿 → 等待拍照推理
  Phase 2: (用户确认后) 查询法兰 → 变换 → 移动到抓取位姿

用法:
  python anydex_eih_grasp.py
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np

import grasp_common
import jog_manual

# ── 手控制 ──────────────────────────────────────────────────
USE_HAND = "--no-hand" not in sys.argv
if USE_HAND:
    from hand_bridge import GraspHand


# ── 路径常量 ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.join(SCRIPT_DIR, "..")

MOVEZ1_SCRIPT = os.path.join(SCRIPT_DIR, "MoveZ1.py")
Z1_PYTHON = "/usr/bin/python3.8"

DEFAULT_GRASP_JSON = os.path.join(WS_ROOT, "logs", "preview", "latest_inspire_grasp.json")
DEFAULT_CALIB = os.path.join(WS_ROOT, "calibration", "cam2flange_current.json")

# ── 抓取偏差标定:人工微调 + 记录 ─────────────────────────────
def _argv_value(name, default):
    """从 sys.argv 取 `--name value`,缺省返回 default。"""
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def _argv_float(name, default):
    try:
        return float(_argv_value(name, default))
    except (TypeError, ValueError):
        return default


# --calib-jog 默认开启：标定版下降到位后必进人工微调停留点(测试 pitch/平移, 空格记录,
# Esc 退出不抓取)。加 --no-calib-jog 可关闭(回到直接抓取流程)。--calib-jog 仍兼容保留。
CALIB_JOG = "--no-calib-jog" not in sys.argv
JOG_HISTORY = _argv_value("--jog-history",
                          os.path.join(WS_ROOT, "logs", "jog_calib_history.jsonl"))
JOG_STEP_MM = _argv_float("--jog-step-mm", 5.0)
JOG_PITCH_STEP_DEG = _argv_float("--jog-pitch-step-deg", 1.0)
JOG_DESCEND_STEP_MM = _argv_float("--jog-descend-step-mm", 10.0)  # 可打断下降每步高度 (mm)

# 丢弃后松手 → 自动回零下电的等待秒数 (松手后给操作者看物体掉落的时间)
DROP_SETTLE_S = _argv_float("--drop-settle-s", 3.0)

# 准备点 — 安全中间位姿，MoveL 的可靠起点
PREP_RPYXYZ = (0.0, -0.094, 0.0, 0.2, 0.0, 0.3)

# 观测位姿 [roll, pitch, yaw, x, y, z]
OBSERVE_RPYXYZ = (0.0, 1.057, 0.0, 0.3, 0.0, 0.35)

# 抓取参数
APPROACH_Z = 0.12   # 抓取点上方安全距离 (m)
LIFT_Z = 0.08       # 抓取后上提高度 (m)
FIXED_RPY = grasp_common.FIXED_RPY  # 固定抓取朝向（兜底）
SPEED = 0.15

# 丢弃位姿 [roll, pitch, yaw, x, y, z]  — 抓取后移到这里松手
DROP_RPYXYZ = (0.0, -0.094, 0.0, 0.35, 0.0, 0.3)

# 安全范围
X_RANGE = (-0.05, 0.65)
Y_RANGE = (-0.50, 0.50)
Z_RANGE = (0.12, 0.70)


# ── Z1 Server 客户端 ─────────────────────────────────────

class Z1Server:
    """MoveZ1.py --server 子进程管理。"""

    def __init__(self, speed=SPEED):
        cmd = [
            Z1_PYTHON, MOVEZ1_SCRIPT, "--server",
            "--no-start-home",
            "--speed", f"{speed:.6f}",
            "--x-range", f"{X_RANGE[0]:.6f}", f"{X_RANGE[1]:.6f}",
            "--y-range", f"{Y_RANGE[0]:.6f}", f"{Y_RANGE[1]:.6f}",
            "--z-range", f"{Z_RANGE[0]:.6f}", f"{Z_RANGE[1]:.6f}",
        ]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
            start_new_session=True,  # 独立进程组，Ctrl+C 不传播到子进程
        )
        # 后台线程持续排空 stdout，_expect 从队列读取以实现真正的超时
        self._lines = queue.Queue()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._expect("READY", timeout=30)

    def _cmd(self, s):
        self.proc.stdin.write(s + "\n")
        self.proc.stdin.flush()

    def _read_loop(self):
        """后台排空子进程 stdout：进度行即时透传终端，结果行进队列。

        子进程退出时（stdout 关闭）放入 None 哨兵，_expect 据此抛 EOFError，
        避免在死掉的子进程上无限阻塞。
        """
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            # 透传非结果行到终端，方便调试
            if line.startswith("[") or line.startswith("✅") or line.startswith("⚠"):
                print("  [Z1]", line, flush=True)
            self._lines.put(line)
        self._lines.put(None)

    def _expect(self, prefix, timeout=60):
        """等待子进程输出以 prefix 开头的结果行。

        timeout 秒内未收到 → 抛 TimeoutError；子进程已退出 → 抛 EOFError。
        通过队列读取，超时真实生效（不再是阻塞 readline 的空转超时）。
        """
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"等待 Z1 响应超时: {prefix}")
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(f"等待 Z1 响应超时: {prefix}")
            if line is None:
                raise EOFError("Z1 子进程已退出")
            if line.startswith("ERROR"):
                raise RuntimeError(f"Z1 错误: {line}")
            if line.startswith(prefix):
                return line[len(prefix):].strip()

    def pose(self, rpyxyz):
        """MoveJ 到准备点 → MoveL 到目标位姿（可靠，不会 IK 失败）。"""
        self._cmd("pose " + " ".join(f"{v:.6f}" for v in rpyxyz))
        self._expect("DONE pose")

    def jogpose(self, rpyxyz):
        """直接 MoveL 到目标位姿（仅适合小范围微调）。"""
        self._cmd("jogpose " + " ".join(f"{v:.6f}" for v in rpyxyz))
        self._expect("DONE jogpose")

    def movej(self, rpyxyz):
        """纯 MoveJ 关节空间运动 — 只算终点 IK，不会失败。"""
        self._cmd("movej " + " ".join(f"{v:.6f}" for v in rpyxyz))
        self._expect("DONE movej")

    def getpose(self):
        """查询当前法兰位姿 → [r, p, y, x, y, z]."""
        self._cmd("getpose")
        resp = self._expect("POS ")
        return np.array([float(v) for v in resp.split()], dtype=float)

    def home(self, timeout=60):
        """MoveJ 回机械零位；timeout 秒内无响应抛超时。"""
        self._cmd("home")
        self._expect("DONE home", timeout=timeout)

    def quit(self, timeout=5):
        """优雅退出子进程；超时或异常时强制结束。"""
        try:
            self._cmd("quit")
            self._expect("BYE", timeout=timeout)
            self.proc.wait(timeout=5)
        except Exception:
            print("[Z1] quit 超时或异常，强制结束子进程", flush=True)
            self._kill()

    def _kill(self):
        """强制结束子进程（幂等，已退出时无副作用）。"""
        try:
            self.proc.kill()
        except (ProcessLookupError, OSError):
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


# ── 主流程 ────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  AnyDexGrasp 眼在手上抓取编排器")
    print("=" * 55)
    print(f"  [标定] 模式: {'已启用(默认, --no-calib-jog 关闭)' if CALIB_JOG else '未启用(--no-calib-jog)'}")
    print(f"  [标定] 历史文件: {JOG_HISTORY}")
    print(f"  [标定] 空格=记录 / 回车=有微调会补记、没微调跳过 / Esc=中止")
    print("=" * 55)

    # ── Phase 1: 移动到观测位姿 ──────────────────────────
    print("\n── Phase 1: 移动到观测位姿 ──")
    z1 = Z1Server()

    # 初始化手控制
    hand = None
    if USE_HAND:
        hand = GraspHand()
        if not hand.connect():
            print("[手] ⚠ 手控制不可用，将跳过手动作")
            hand = None

    try:
        # MoveJ 到准备点（关节空间，只需终点 IK，不受起始位姿影响）
        print("  MoveJ → 准备点...")
        z1.movej(PREP_RPYXYZ)
        print("  MoveJ → 观测位姿...")
        z1.movej(OBSERVE_RPYXYZ)
        r, p, y, x, y_pos, z = OBSERVE_RPYXYZ
        print(f"\n✅ Z1 已到达观测位姿: r={r:.3f} p={p:.3f} y={y:.3f} "
              f"x={x:.3f} y={y_pos:.3f} z={z:.3f}")
        print(f"   Z1 连接保持中，机械臂不会掉力")

        # ── 等待用户拍照推理 ──────────────────────────────
        print(f"\n{'─' * 55}")
        print("[拍照推理] 请在另一个终端运行:")
        print(f"     conda activate anydex")
        print(f"     cd {SCRIPT_DIR}")
        print(f"     python run_inspire_preview.py --visualize --collision \\")
        print(f"         --min_depth 0.2 --max_depth 0.6")
        print(f"  检查 Open3D 窗口中的抓取姿态，确认无误后回到此处。")
        print(f"{'─' * 55}")

        input("\n>>> 确认抓取位姿无误，按回车执行抓取 (Ctrl+C 取消): ")

        # ── Phase 2: 执行抓取 ─────────────────────────────
        print("\n── Phase 2: 执行抓取 ──")

        # 加载标定
        with open(DEFAULT_CALIB) as f:
            calib = json.load(f)
        R_c2f = np.array(calib["R_cam2flange"], dtype=float)
        t_c2f = np.array(calib["t_cam2flange"], dtype=float)
        print(f"[标定] cam2flange 已加载")

        # 加载抓取
        if not os.path.exists(DEFAULT_GRASP_JSON):
            print(f"❌ 抓取文件不存在: {DEFAULT_GRASP_JSON}")
            print(f"   请先运行 run_inspire_preview.py 生成抓取位姿。")
            return 1

        with open(DEFAULT_GRASP_JSON) as f:
            grasp = json.load(f)
        p_cam = np.array(grasp["translation"], dtype=float)
        print(f"[抓取] score={grasp['score']:.3f}  "
              f"type={grasp['grasp_type']}  "
              f"p_cam=[{p_cam[0]:.3f}, {p_cam[1]:.3f}, {p_cam[2]:.3f}]")

        # 查询法兰: 用命令观测位姿替代 getpose()
        # verify_z1_pose.py 实测 getpose/FK 有 ~39.4±1.0mm 沿手前向的系统误差,
        # 用它做 cam→base 会把 ~40mm 注入每个抓取(即 CLAUDE.md 的 42mm 偏移)。
        # MoveJ 到位稳定(Test A 重复性 0mm),命令观测位姿更可靠;
        # 预览 run_inspire_preview.py 也一直按 OBSERVE_RPYXYZ 画法兰框。
        flange = np.array(OBSERVE_RPYXYZ, dtype=float)
        print(f"[Z1] 法兰(命令观测位姿): r={flange[0]:.3f} p={flange[1]:.3f} y={flange[2]:.3f} "
              f"x={flange[3]:.3f} y={flange[4]:.3f} z={flange[5]:.3f}")

        # 计算目标：网络朝向 + 安装偏移补偿 + 偏差保护
        hand_cfg = grasp_common.load_hand_config()
        target = grasp_common.compute_grasp_target(
            grasp, R_c2f, t_c2f, flange,
            hand_cfg["hand_mount_z"], hand_cfg["mount_yaw_deg"],
            hand_cfg["orient_deviation_thresh_deg"],
            fixed_rpy=FIXED_RPY,
        )
        rpy = target["rpy"]
        p_flange = target["p_flange"]

        if target["used_network"]:
            off_note = "  (rpy 已含 offset_rpy，deviation 为 offset 前值)" \
                if not np.allclose(grasp_common.load_offset_rpy_deg(), 0) else ""
            print(f"[朝向] 使用网络朝向 rpy=[{rpy[0]:.3f},{rpy[1]:.3f},{rpy[2]:.3f}]  "
                  f"(与固定朝向夹角 {target['deviation_deg']:.1f}°){off_note}")
        else:
            print(f"[朝向] 网络朝向夹角 {target['deviation_deg']:.1f}° 超阈值，回退固定朝向 "
                  f"[{FIXED_RPY[0]:.3f},{FIXED_RPY[1]:.3f},{FIXED_RPY[2]:.3f}]")

        # 构建位姿（法兰位姿 = 补偿后的位置 + 当前朝向）
        def _pose(rpy, xyz):
            return np.array([rpy[0], rpy[1], rpy[2], xyz[0], xyz[1], xyz[2]])

        approach = _pose(rpy, p_flange + np.array([0, 0, APPROACH_Z]))
        lift_target = _pose(rpy, p_flange + np.array([0, 0, LIFT_Z]))
        drop = np.array(DROP_RPYXYZ, dtype=float)
        grasp_type = int(grasp["grasp_type"])

        off_rpy = grasp_common.load_offset_rpy_deg()
        off_rpy_txt = ""
        if not np.allclose(off_rpy, 0):
            off_rpy_txt = (f"  +offset_rpy=[{', '.join(f'{v:+.1f}°' for v in off_rpy)}]"
                           f"(基座系绕掌心)")
        print(f"[目标] rpy=[{rpy[0]:.3f},{rpy[1]:.3f},{rpy[2]:.3f}] "
              f"xyz=[{p_flange[0]:.3f},{p_flange[1]:.3f},{p_flange[2]:.3f}] "
              f"(掌心=[{target['p_palm_base'][0]:.3f},{target['p_palm_base'][1]:.3f},"
              f"{target['p_palm_base'][2]:.3f}]){off_rpy_txt}")

        # ── 自动抓取流程 ─────────────────────────────────
        # Phase 2a: 移动到抓取点上方。Z1 真实 IK 工作空间比粗过滤窄,网络朝向不可达
        # 时回退固定朝向;两者都失败则逐步降低接近高度(12→8→6cm)重试——很多情况
        # 只是接近点(竖直 +APPROACH_Z)抬太高超出 IK 可达区。全失败 = 位姿真不可达,
        # 优雅中止(回零退出,不抛 traceback)。
        print(f"\n[执行] Phase 2a: MoveJ → 抓取点上方 {APPROACH_Z*100:.0f}cm")
        approach_heights = [APPROACH_Z, 0.08, 0.06]
        # 候选姿态:先网络朝向(若本轮用了网络),再固定朝向兜底
        cands = []
        if target["used_network"]:
            cands.append((target, "网络朝向"))
        t_fixed = grasp_common.compute_grasp_target(
            grasp, R_c2f, t_c2f, flange,
            hand_cfg["hand_mount_z"], hand_cfg["mount_yaw_deg"],
            hand_cfg["orient_deviation_thresh_deg"],
            fixed_rpy=FIXED_RPY, force_fixed=True,
        )
        cands.append((t_fixed, "固定朝向"))

        moved = False
        for az in approach_heights:
            for t, label in cands:
                rpy = t["rpy"]
                p_flange = t["p_flange"]
                approach = _pose(rpy, p_flange + np.array([0, 0, az]))
                lift_target = _pose(rpy, p_flange + np.array([0, 0, LIFT_Z]))
                try:
                    z1.movej(approach)
                    moved = True
                    target = t
                    break
                except Exception as e:
                    print(f"[执行] ⚠ {label} · 接近高 {az*100:.0f}cm MoveJ 失败 ({e})", flush=True)
            if moved:
                break

        if not moved:
            print("\n" + "=" * 55)
            print("❌ 该抓取位姿超出 Z1 可达范围:网络/固定朝向在接近高度 "
                  f"{'/'.join(f'{h*100:.0f}' for h in approach_heights)}cm 均 IK 无解。")
            print("   请换个物体位置、或重新拍照推理、或调小 config 的抓取范围后重试。")
            print("=" * 55)
            return 1

        if hand is not None:
            hand.open()                        # 张开手指
            hand.preshape(grasp_type)          # 空中预塑形(含拇指内旋),避免低空旋转撞桌

        print(f"[执行] Phase 2b: 下降 {APPROACH_Z*100:.0f}cm → 抓取点 "
              f"(Space/Esc 暂停接管人工微调)")
        interrupted, cur_pose = jog_manual.descend_interruptible(
            z1, rpy, p_flange + np.array([0, 0, APPROACH_Z]), p_flange,
            step_mm=JOG_DESCEND_STEP_MM)
        if interrupted:
            print("\n[接管] 已暂停自动下降。请用 w/s/a/d/q/e + r/f 人工微调,"
                  "回车确认后继续闭合。")
            res = jog_manual.jog_manual_stage(
                z1, cur_pose, {}, {}, JOG_HISTORY,
                step_mm=JOG_STEP_MM, pitch_step_deg=JOG_PITCH_STEP_DEG,
                record=False)
            if res == "abort":
                print("\n[接管] 已中止,回零后退出...")
                raise KeyboardInterrupt

        # ── 抓取偏差标定:人工微调 + 记录 (--calib-jog) ────────────
        if CALIB_JOG:
            print("\n[标定] 已到达抓取点。请用 w/s/a/d/q/e 微调对准物体。")
            print(f"[标定] 对准后按【空格】记录(写入 {JOG_HISTORY});回车=有微调补记/没微调跳过;Esc=中止。")
            grasp_meta = {
                "grasp_json": DEFAULT_GRASP_JSON,
                "grasp_type": grasp_type,
                "score": grasp.get("score"),
                "p_cam": p_cam,
                "depth_cam": float(np.linalg.norm(p_cam)),
                "used_network": target["used_network"],
                "deviation_deg": target["deviation_deg"],
            }
            calib_cfg = {**hand_cfg,
                         "offset_xyz": grasp_common.load_offset_xyz(),
                         "offset_rpy_deg": grasp_common.load_offset_rpy_deg()}
            res = jog_manual.jog_manual_stage(
                z1, np.concatenate([rpy, p_flange]), grasp_meta,
                calib_cfg, JOG_HISTORY, step_mm=JOG_STEP_MM,
                pitch_step_deg=JOG_PITCH_STEP_DEG)
            if res == "abort":
                print("\n[标定] 已中止,回零后退出...")
                raise KeyboardInterrupt

        if hand is not None:
            hand.close(grasp_type)             # 闭合抓取
        else:
            input(f"\n>>> [手动] 请用手 GUI 完成抓取，按回车继续: ")

        # ── 抓取确认闸门：人眼观察抓取是否成功，确认后才上提 ──
        print(f"\n{'─' * 55}")
        print("[观察] 手已闭合。请目视确认抓取是否成功")
        print("      (物体是否被稳稳握住、手型是否正确)。")
        print(f"{'─' * 55}")
        input("\n>>> 确认抓取成功，按回车上提+移动到丢弃位姿 (Ctrl+C 取消): ")

        print(f"[执行] Phase 2c: MoveJ → 上提 {LIFT_Z*100:.0f}cm")
        z1.movej(lift_target)

        print(f"[执行] Phase 2d: MoveJ → 丢弃位姿")
        z1.movej(drop)

        # ── 丢弃确认闸门：确认丢弃位置后，才打开手执行丢弃 ──
        print(f"\n{'─' * 55}")
        print("[观察] 已到达丢弃位姿。请确认丢弃位置无遮挡")
        print("      (垃圾桶/容器在正下方)。")
        print(f"{'─' * 55}")
        input("\n>>> 确认丢弃位置，按回车打开手执行丢弃 (Ctrl+C 取消): ")

        if hand is not None:
            hand.release()                     # 松手

        print(f"\n✅ 抓取完成，物体已丢弃。")

        # 丢弃后自动回零:松手后等 DROP_SETTLE_S 秒(看物体掉落),然后落入 finally
        # 自动回零下电,无需再按回车。
        print(f"[下电] {DROP_SETTLE_S:.0f} 秒后自动回零下电 (Ctrl+C 可随时取消)...")
        time.sleep(DROP_SETTLE_S)

    except KeyboardInterrupt:
        print("\n\n⚠️  用户取消，先回零再松手...")
        # 不设 SIG_IGN：清理期间再按一次 Ctrl+C 会立即抛 KeyboardInterrupt 强退
        # 不在 except 里松手: 低空(下降中/抓取点)直接张开手指易扫到桌面损坏手,
        # 统一放到 finally 在机械臂回零之后执行。
    finally:
        print("\n[关闭] 安全退出...")
        # 顺序: 先机械臂回零(手随臂离开桌面到安全高度), 再松手复位灵巧手。
        # 方案 A：任何退出路径（正常完成 / Ctrl+C / 异常）下电前先回零。
        # 编排器已不再传 --no-return-home，MoveZ1 shutdown 默认也会 backToStart 兜底；
        # 这里先 home 让子进程下电前的 backToStart 成为空操作，
        # 避免 quit 等待超时把正在回零的子进程强杀。
        try:
            z1.home(timeout=30)
        except Exception:
            print("[Z1] 回零失败或超时，跳过回零", flush=True)
        # 机械臂回零后, 手已在安全高度, 再复位灵巧手(全开)并关串口
        if hand is not None:
            try:
                hand.release()
                hand.close_connection()
            except Exception:
                pass
        # 给子进程留足时间完成 shutdown 内的 backToStart + PASSIVE，超时/异常再强杀
        try:
            z1.quit(timeout=30)
        except Exception:
            pass
        print("✅ 完成。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
