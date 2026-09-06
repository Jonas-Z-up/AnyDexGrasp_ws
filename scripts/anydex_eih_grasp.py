#!/usr/bin/env python3
"""AnyDexGrasp 眼在手上 — 演示循环抓取编排脚本 (终端A)。

恒为自动循环: 每轮拍照推理后在终端A按一次回车确认新抓取位姿, 之后 Phase 2
(接近→张手+预塑形→可打断下降→闭合→上提→丢弃→松手) 全自动执行, 手部步骤用
固定等待 (HAND_WAIT_S) 替代回车闸门; 松手后回到观测位姿循环下一轮。
进程仅在 Ctrl+C 或硬错误时退出(回零下电)。

用法:
  python anydex_eih_grasp.py              # 演示循环
  python anydex_eih_grasp.py --no-hand    # 无手桥: 手部步骤用固定等待模拟

所有可调参数在文件顶部「参数设置」区, 非命令行 flags。
标定(人工微调+记录)请用 anydex_eih_grasp_calib.py --calib-jog。
"""

import json
import os
import queue
import select
import subprocess
import sys
import termios
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

# ═══ 参数设置（demo 调参区）═══
# 所有可调参数集中在此；非命令行 flags。
DESCEND_STEP_MM = 10.0          # 可打断下降每步高度 (mm)
GRASP_WAIT_S = 1.0              # 到达抓取点后、闭合前等待秒数 (让手稳定对准)
HAND_WAIT_S = 3.0               # 手部步骤固定等待秒数
DROP_SETTLE_S = 1.0             # 松手后等待秒数 (看物体掉落)
TAKEOVER_STEP_MM = 5.0          # 接管微调初始平移步长 (mm)
TAKEOVER_PITCH_STEP_DEG = 1.0   # 接管微调俯仰步长 (°)

# 准备点 — 安全中间位姿, MoveL 的可靠起点
PREP_RPYXYZ = (0.0, -0.094, 0.0, 0.2, 0.0, 0.3)
# 观测位姿 [roll, pitch, yaw, x, y, z]
OBSERVE_RPYXYZ = (0.0, 1.057, 0.0, 0.3, 0.0, 0.35)
# 抓取参数
APPROACH_Z = 0.12               # 抓取点上方安全距离 (m)
LIFT_Z = 0.08                   # 抓取后上提高度 (m)
FIXED_RPY = grasp_common.FIXED_RPY   # 固定抓取朝向（兜底）
SPEED = 0.15
# 中转节点 [roll, pitch, yaw, x, y, z] — 避障
TRANSIT_RPYXYZ = (0.0, -0.094, 0.0, 0.5, 0.0, 0.4)
# 丢弃位姿 [roll, pitch, yaw, x, y, z] — 抓取后移到这里松手
DROP_RPYXYZ = (0.0, -0.094, 0.5, 0.5, 0.2, 0.4)
# 安全范围
X_RANGE = (-0.05, 0.65)
Y_RANGE = (-0.50, 0.50)
Z_RANGE = (0.12, 0.70)

# 手安装/朝向参数 —— 统一从 config/grasp_config.json 读取（与标定脚本/预览同源）。
# 旧版此处为硬编码常量，会与 config 漂移导致"标定结果不迁移到演示循环"，已废除。
_hand_cfg = grasp_common.load_hand_config()
HAND_MOUNT_Z = _hand_cfg["hand_mount_z"]                          # 法兰→掌心沿接近轴 (m)
HAND_FINGERTIP_Z = _hand_cfg["hand_fingertip_z"]                  # 法兰→指尖 (m), 参考值
MOUNT_YAW_DEG = _hand_cfg["mount_yaw_deg"]                        # 拇指/掌心绕接近轴方位角 α
ORIENT_DEV_THRESH_DEG = _hand_cfg["orient_deviation_thresh_deg"]  # 网络朝向偏差阈值, 超则回退
OFFSET_XYZ = grasp_common.load_offset_xyz()                       # 基座系抓取位置偏移 (m)
OFFSET_RPY_DEG = grasp_common.load_offset_rpy_deg()               # 基座系绕掌心旋转偏移 (度)
Z_COMP_KNOTS = grasp_common.load_z_comp_knots()                   # z 高度补偿锚点 [命令z, 补偿量] (m)


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


# ── 手动接管（下降被打断后）─────────────────────────────

# 接管微调帮助 (紧凑版)
TAKEOVER_HELP = """\
  接管微调 (命令位姿基准, 绝对 MoveJ):
    w/s  前/后 (+X/−X)    a/d  左/右 (+Y/−Y)    q/e  上/下 (+Z/−Z)
    r/f  俯仰 +/− (绕法兰中心改 pitch)
    + / -  步长切换 (2/5/10 mm)      ? / h  本帮助
    Enter / Space  确认此位姿并继续闭合   Esc  中止(回零退出)
  (基准=命令位姿, 绝不用 getpose: 真机 FK 有 ~39.4mm 系统偏移)"""


def _manual_takeover(z1, cur_pose,
                     step_mm=TAKEOVER_STEP_MM,
                     pitch_step_deg=TAKEOVER_PITCH_STEP_DEG):
    """下降被打断后的纯人工接管: 平移 w/s/a/d/q/e + 俯仰 r/f。

    基准 = 命令位姿 cur_pose (MoveJ 同坐标系), 绝不用 getpose 当基准:
    真机 getpose/FK 有 ~39.4mm 沿接近轴系统偏移, 拿它当基准会把 2mm 微调
    放大成数十 mm。相邻两次 MoveJ 命令差 2mm → 手臂实际正好动 2mm。
    返回 "continue"(Enter/Space) 或 "abort"(Esc); Ctrl+C 直接抛 KeyboardInterrupt。
    无记录/无历史/无 getpose 诊断。
    """
    if not sys.stdin.isatty():
        print("[接管] ⚠ 非终端输入，无法键盘微调，按当前位姿继续。")
        return "continue"

    base = np.asarray(cur_pose, dtype=float).copy()
    step = float(step_mm)
    keymap = jog_manual.KEYMAP            # 复用 jog_manual 的键→轴映射
    pitch_map = jog_manual.PITCH_KEYMAP
    steps = list(jog_manual.STEP_SIZES_MM)

    print("\n[接管] 已暂停自动下降，进入人工微调 (命令位姿基准):")
    print(TAKEOVER_HELP)
    print(f"  初始 xyz=[{base[3]:.3f},{base[4]:.3f},{base[5]:.3f}]  步长={step:.0f}mm")

    fd = sys.stdin.fileno()
    # cbreak 式: 关 ICANON|ECHO, 保留 ISIG → Ctrl+C 以 SIGINT 到达(编排器
    # except KeyboardInterrupt → 回零下电), 不必特判 \x03 字节。
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    new[3] &= ~(termios.ICANON | termios.ECHO)
    termios.tcsetattr(fd, termios.TCSADRAIN, new)
    try:
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            if not ready:
                continue
            ch = sys.stdin.read(1)
            if ch in keymap:
                axis, sign = keymap[ch]
                base[3 + axis] += sign * step / 1000.0
                z1.movej(base)
                print(f"  [jog] {'XYZ'[axis]}{'+' if sign > 0 else '-'}{step:.0f}mm "
                      f"→ xyz=[{base[3]:.3f},{base[4]:.3f},{base[5]:.3f}]", flush=True)
            elif ch in pitch_map:
                sign = pitch_map[ch]
                base[1] += sign * np.deg2rad(pitch_step_deg)
                z1.movej(base)
                print(f"  [jog] pitch {'+' if sign > 0 else '-'}{pitch_step_deg:.1f}° "
                      f"→ rpy=[{base[0]:.3f},{base[1]:.3f},{base[2]:.3f}]", flush=True)
            elif ch in ("+", "-"):
                idx = min(range(len(steps)), key=lambda i: abs(steps[i] - step))
                idx = max(0, min(len(steps) - 1, idx + (1 if ch == "+" else -1)))
                step = steps[idx]
                print(f"  [步长] → {step:.0f}mm", flush=True)
            elif ch in ("?", "h"):
                print(TAKEOVER_HELP, flush=True)
            elif ch in ("\r", "\n", " "):
                print("  [接管] 位置已确认，继续抓取。", flush=True)
                return "continue"
            elif ch == "\x1b":
                print("  [接管] 已中止，回零后退出。", flush=True)
                return "abort"
            else:
                print(f"  [未绑定] {ch!r}  (? 帮助)", flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ── 主流程 ────────────────────────────────────────────────

def grasp_once(z1, hand, round_no):
    """执行第 round_no 轮完整抓取 (Phase 2), 返回状态码 0=成功, 1=失败。

    Phase 2a 接近 → 空中预塑形 → Phase 2b 下降(可打断接管) → 闭合 → 上提 → 丢弃 → 松手。
    手部步骤用固定等待 (HAND_WAIT_S) 替代回车闸门。
    失败 (抓取文件缺失 / 超出可达范围) 直接 return 1, 由 main() 续轮 (机械臂不断电)。
    """
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
    # 手参数显式传文件顶部常量, 不读 config json (compute_grasp_target 参数为
    # None 时会回退读 config, 此处显式传值保证 demo 零 config 依赖)。
    target = grasp_common.compute_grasp_target(
        grasp, R_c2f, t_c2f, flange,
        HAND_MOUNT_Z, MOUNT_YAW_DEG, ORIENT_DEV_THRESH_DEG,
        offset_xyz=OFFSET_XYZ, offset_rpy_deg=OFFSET_RPY_DEG,
        z_comp_knots=Z_COMP_KNOTS,
        fixed_rpy=FIXED_RPY,
    )
    rpy = target["rpy"]
    p_flange = target["p_flange"]

    if target["used_network"]:
        off_note = "  (rpy 已含 offset_rpy，deviation 为 offset 前值)" \
            if not np.allclose(OFFSET_RPY_DEG, 0) else ""
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
    transit = np.array(TRANSIT_RPYXYZ, dtype=float)
    grasp_type = int(grasp["grasp_type"])

    off_rpy = OFFSET_RPY_DEG
    off_rpy_txt = ""
    if not np.allclose(off_rpy, 0):
        off_rpy_txt = (f"  +offset_rpy=[{', '.join(f'{v:+.1f}°' for v in off_rpy)}]"
                       f"(基座系绕掌心)")
    z_comp_txt = f"  +z_comp={target['z_comp']:.3f}" if target["z_comp"] != 0.0 else ""
    print(f"[目标] rpy=[{rpy[0]:.3f},{rpy[1]:.3f},{rpy[2]:.3f}] "
          f"xyz=[{p_flange[0]:.3f},{p_flange[1]:.3f},{p_flange[2]:.3f}] "
          f"(掌心=[{target['p_palm_base'][0]:.3f},{target['p_palm_base'][1]:.3f},"
          f"{target['p_palm_base'][2]:.3f}]){off_rpy_txt}{z_comp_txt}")

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
        HAND_MOUNT_Z, MOUNT_YAW_DEG, ORIENT_DEV_THRESH_DEG,
        offset_xyz=OFFSET_XYZ, offset_rpy_deg=OFFSET_RPY_DEG,
        z_comp_knots=Z_COMP_KNOTS,
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
        print("   请换个物体位置、或重新拍照推理、或调大文件顶部 Z_RANGE 后重试。")
        print("=" * 55)
        return 1

    if hand is not None:
        hand.open()                        # 张开手指
        hand.preshape(grasp_type)          # 空中预塑形(含拇指内旋),避免低空旋转撞桌

    print(f"[执行] Phase 2b: 下降 {APPROACH_Z*100:.0f}cm → 抓取点 "
          f"(Space/Esc 暂停接管人工微调)")
    interrupted, cur_pose = jog_manual.descend_interruptible(
        z1, rpy, p_flange + np.array([0, 0, APPROACH_Z]), p_flange,
        step_mm=DESCEND_STEP_MM)
    if interrupted:
        res = _manual_takeover(z1, cur_pose)
        if res == "abort":
            raise KeyboardInterrupt

    if hand is not None:
        print(f"[确认] 已到达抓取点，{GRASP_WAIT_S:.0f}s 后闭合抓取...")
        time.sleep(GRASP_WAIT_S)
        hand.close(grasp_type)             # 闭合抓取
    else:
        print(f"[手] --no-hand：等待 {HAND_WAIT_S:.0f}s 模拟人工抓取...")
        time.sleep(HAND_WAIT_S)

    print(f"[确认] 手已闭合，{HAND_WAIT_S:.0f}s 后自动上提 (Ctrl+C 可取消)...")
    time.sleep(HAND_WAIT_S)

    print(f"[执行] Phase 2c: MoveJ → 上提 {LIFT_Z*100:.0f}cm")
    z1.movej(lift_target)

    print(f"[执行] Phase 2d: MoveJ → 中转节点 (丢弃位姿上方)")
    z1.movej(transit)

    print(f"[执行] Phase 2e: MoveJ → 丢弃位姿")
    z1.movej(drop)

    print(f"[确认] 已到达丢弃位姿，{HAND_WAIT_S:.0f}s 后自动松手 (Ctrl+C 可取消)...")
    time.sleep(HAND_WAIT_S)

    if hand is not None:
        hand.release()                     # 松手

    print(f"\n[第 {round_no} 轮] ✅ 抓取完成，物体已丢弃。")
    print(f"[循环] {DROP_SETTLE_S:.0f}s 后自动返回观测位姿，等待下一轮 (Ctrl+C 取消)...")
    time.sleep(DROP_SETTLE_S)
    return 0


def main():
    print("=" * 55)
    print("  AnyDexGrasp 眼在手上 — 演示循环模式")
    print("    每轮: 拍照推理 → 回车自动抓取 → 松手后回观测位姿循环")
    print("    Ctrl+C 回零下电退出(唯一退出方式); 调参改文件顶部「参数设置」区")
    print("=" * 55)
    print("  (保持本终端: 关闭会掉力)")

    # ── 初始化 Z1 与手控制 ─────────────────────────────
    z1 = Z1Server()
    hand = None
    if USE_HAND:
        hand = GraspHand()
        if not hand.connect():
            print("[手] ⚠ 手控制不可用，将跳过手动作")
            hand = None

    try:
        round_no = 0
        while True:
            round_no += 1
            print(f"\n{'=' * 55}\n  第 {round_no} 轮\n{'=' * 55}")
            print("  MoveJ → 准备点...")
            z1.movej(PREP_RPYXYZ)
            print("  MoveJ → 观测位姿...")
            z1.movej(OBSERVE_RPYXYZ)
            r, p, y, x, y_pos, z = OBSERVE_RPYXYZ
            print(f"  已到达观测位姿 r={r:.3f} p={p:.3f} y={y:.3f} "
                  f"x={x:.3f} y={y_pos:.3f} z={z:.3f}")
            print(f"  [拍照推理] 终端B: python run_inspire_preview.py --visualize --collision "
                  f"--preprocess --min_depth 0.2 --max_depth 0.6")

            input("\n>>> 确认抓取位姿无误，按回车执行抓取 (Ctrl+C 取消): ")

            print(f"── 第 {round_no} 轮 Phase 2: 执行抓取 ──")
            status = grasp_once(z1, hand, round_no)
            if status != 0:
                print(f"\n[第 {round_no} 轮] 本轮失败，返回观测位姿。重新拍照推理后按回车重试。")
                continue                      # 回循环顶 PREP→OBSERVE，机械臂不断电
            # 成功 → 自然进入下一轮: 循环顶重跑 PREP→OBSERVE 即"回到观测位姿"

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
