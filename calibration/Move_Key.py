#!/usr/bin/python3.8
"""
Z1 混合控制器: 键盘点动 + 命令输入 + 位姿服务

功能:
  - 键盘点动模式 (WASD/IJKLUO 微调, 默认)
  - 命令输入模式 (空格键 → pose/jogpose/home/zero/quit)
  - Unix socket 位姿服务 (默认 /tmp/z1_pose_server.sock, 供 verify/data_collect 获取实时位姿)

启动时自动 MoveJ 到准备点。
pose 命令直接 MoveJ 到目标 (关节空间插值, 终点 IK 始终可解)。

位姿帧说明 (重要, 关系到手眼标定的正确性)
----------------------------------------
本文件用 ArmInterface(hasGripper=False) 连接机械臂, 因此 get_end_posture()
(FK 读回) 与 MoveJ/MoveL 的**命令帧**是同一个帧 —— 法兰安装面。

若误用 hasGripper=True, FK 末端帧会前移 38.2mm 当作"夹爪 TCP", 两者脱节:
点动基准跳变, 且按 P 打印的位姿会让标定外参沿接近轴整体偏 ~38mm。
启动时会自动跑 check_frame_consistency() 拦截这种情况。
"""

import argparse
import os
import select
import sys
import termios
import time
import tty

import numpy as np

# ========== Z1 位姿 IPC (socket 服务) ==========
from z1_pose_client import PoseServer, DEFAULT_SOCKET_PATH

# ========== Z1 SDK 路径 ==========
SDK_LIB_DIR = "/home/jonas/Projects/unitree_z1/Unitree-Z1_SDK/z1_sdk/lib"
if SDK_LIB_DIR not in sys.path:
    sys.path.insert(0, SDK_LIB_DIR)
os.environ["LD_LIBRARY_PATH"] = SDK_LIB_DIR + ":" + os.environ.get("LD_LIBRARY_PATH", "")
import unitree_arm_interface

# ========== 键盘点动映射 ==========
# 平移: 末端坐标系
KEY_DELTAS = {
    "w": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),   # +X 前
    "s": np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0]),  # -X 后
    "a": np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0]),  # -Y 左
    "d": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),   # +Y 右
    "q": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),   # +Z 上
    "e": np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),  # -Z 下
}
# 旋转: 末端坐标系
RPY_KEY_DELTAS = {
    "i": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),   # roll +
    "k": np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # roll -
    "j": np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0]),  # pitch -
    "l": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),   # pitch +
    "u": np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0]),  # yaw -
    "o": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),   # yaw +
}

# 默认准备点 [roll, pitch, yaw, x, y, z]
# DEFAULT_START_POSE = np.array([0.0, -0.094, 0.0, 0.2, 0.0, 0.30], dtype=float)
DEFAULT_START_POSE = np.array([0.075, 1.057, 0.073, 0.327, -0.08, 0.35], dtype=float)


# ══════════════════════════════════════════════════════════
# 终端管理
# ══════════════════════════════════════════════════════════

class RawTerminal:
    """cbreak 终端管理器, 支持临时恢复 cooked 模式以读取命令行."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        self._set_cbreak()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._restore()

    def _set_cbreak(self):
        tty.setcbreak(self.fd)

    def _restore(self):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def read_line(self, prompt="CMD> "):
        """临时恢复 cooked 终端读取一行, 然后切回 cbreak."""
        self._restore()
        try:
            line = input(prompt)
        finally:
            self._set_cbreak()
        return line


def read_key(timeout=0.05):
    """非阻塞读取单个按键."""
    readable, _, _ = select.select([sys.stdin], [], [], timeout)
    if readable:
        return sys.stdin.read(1)
    return None


# ══════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════

def clamp_workspace(posture, x_range, y_range, z_range):
    x, y, z = posture[3], posture[4], posture[5]
    if not (x_range[0] <= x <= x_range[1]):
        raise ValueError(f"X={x:.3f} 超出 [{x_range[0]:.3f}, {x_range[1]:.3f}]")
    if not (y_range[0] <= y <= y_range[1]):
        raise ValueError(f"Y={y:.3f} 超出 [{y_range[0]:.3f}, {y_range[1]:.3f}]")
    if not (z_range[0] <= z <= z_range[1]):
        raise ValueError(f"Z={z:.3f} 超出 [{z_range[0]:.3f}, {z_range[1]:.3f}]")


def format_status(posture, q, step, rpy_step, target_pose=None):
    target_text = ""
    if target_pose is not None:
        target_text = (
            f"  TGT R{target_pose[0]: .3f} P{target_pose[1]: .3f} Yaw{target_pose[2]: .3f}"
            f" X{target_pose[3]: .3f} Y{target_pose[4]: .3f} Z{target_pose[5]: .3f}"
        )
    return (
        f"RPYXYZ [{posture[0]: .3f} {posture[1]: .3f} {posture[2]: .3f} | "
        f"X{posture[3]: .3f} Y{posture[4]: .3f} Z{posture[5]: .3f}]  "
        f"J [{q[0]: .2f} {q[1]: .2f} {q[2]: .2f} {q[3]: .2f} {q[4]: .2f} {q[5]: .2f}]  "
        f"xyz={step*1000:.1f}mm rpy={np.degrees(rpy_step):.1f}deg"
        f"{target_text}"
    )


def print_help():
    print()
    print("=" * 60)
    print("  Z1 混合控制器: 键盘点动 + 命令输入")
    print("=" * 60)
    print("键盘点动:")
    print("  W/S: +X/-X   A/D: -Y/+Y   Q/E: +Z/-Z")
    print("  I/K: Roll+/-   J/L: Pitch-/+   U/O: Yaw-/+")
    print("  [ / ]: XYZ步长减半/加倍    - / =: RPY步长减半/加倍")
    print("  P: 打印当前法兰位姿(可直接粘贴到采集脚本)   F: 回准备点")
    print("  H: 回零位   X: 退出")
    print()
    print("命令模式 (按 空格键 进入):")
    print("  pose r p y x y z     直接MoveJ到目标   [推荐, 关节空间始终可达]")
    print("  jogpose r p y x y z  当前位置直接MoveL  [直线, 仅微调]")
    print("  r / home             回到准备点")
    print("  zero                 回机械零位")
    print("  q / hold             保持姿态, Ctrl+C退出")
    print("  quit                 退出")
    print("=" * 60)
    print()


# ══════════════════════════════════════════════════════════
# 混合控制器
# ══════════════════════════════════════════════════════════

class Z1HybridController:
    """键盘点动 + 命令输入 混合控制."""

    def __init__(self, step=0.005, rpy_step=0.035, speed=0.15,
                 no_return_home=False, no_safe_check=False, no_start_move=False,
                 start_pose=None,
                 x_range=(0.05, 0.65), y_range=(-0.35, 0.35), z_range=(0.10, 0.60),
                 pose_server_path=None):
        self.step = step
        self.rpy_step = rpy_step
        self.speed = speed
        self.no_return_home = no_return_home
        self.no_safe_check = no_safe_check
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.running = True

        # 准备点
        if start_pose is None:
            self.start_pose = DEFAULT_START_POSE.copy()
        else:
            self.start_pose = np.array(start_pose, dtype=float)

        # 位姿服务器 (Unix domain socket, 供 verify/data_collect 获取实时位姿)
        self.pose_server = None
        if pose_server_path is not None:
            self.pose_server = PoseServer(pose_server_path)
            self.pose_server.start()
            print(f"[初始化] Pose 服务器已启动: {pose_server_path}")

        # 连接机械臂
        # ⚠️ hasGripper 必须为 False: 末端装的是 D435i + 自研手, 没有 Unitree 夹爪。
        # 传 True 会让 ArmModel 把末端帧(index=6)从法兰安装面前移 38.2mm 当作"夹爪 TCP",
        # 于是 get_end_posture() 与 MoveJ/MoveL 的**命令帧**不再重合:
        #   - 首次点动会以错误的基准跳 38.2mm
        #   - 按 P 打印的位姿存进 poses.txt → 标定出的外参沿接近轴整体偏 38.2mm
        #     (这正是 CLAUDE.md 里"getpose 与命令位姿差 39.4±1.0mm"的根因, 不是 FK 虚报)
        # 注: 下方 MoveJ/MoveL 仍传 gripperPos=0.0 (三参数重载) —— hasGripper=False 时
        #     夹爪通道不存在, 该参数被忽略; 保持原调用形式以最小化改动。
        print("\n[初始化] 连接 Z1 机械臂...")
        self.arm = unitree_arm_interface.ArmInterface(hasGripper=False)
        self.arm_model = self.arm._ctrlComp.armModel
        self.arm.loopOn()
        time.sleep(1.0)

        # 启动: setFsm → MoveJ 到准备点
        if not no_start_move:
            print(f"[初始化] MoveJ 到准备点: {self.start_pose.tolist()}")
            self.arm.setFsm(unitree_arm_interface.ArmFSMState.JOINTCTRL)
            time.sleep(0.2)
            ok = self.arm.MoveJ(self.start_pose, 0.0, self.speed)
            if not ok:
                print("[警告] 初始 MoveJ 到准备点失败, 尝试继续...")
            time.sleep(0.5)
            # 帧一致性自检: 必须在 reset_target_from_current() 之前, 否则基准被 FK 覆盖
            self.check_frame_consistency(self.start_pose)
        else:
            print("[初始化] 跳过初始移动, 仅初始化 FSM...")
            self.arm.setFsm(unitree_arm_interface.ArmFSMState.JOINTCTRL)
            time.sleep(0.2)
            print("[初始化] 跳过帧自检 (无命令参考位姿)")

        self.reset_target_from_current()
        print("[初始化] ✅ Z1 就绪\n")

    # ── 状态查询 ──────────────────────────────────────────

    def get_joint_q(self):
        return np.asarray(self.arm.lowstate.getQ(), dtype=float)

    def get_end_posture(self):
        """从关节角 FK 计算实际末端位姿 [r,p,yaw,x,y,z].

        hasGripper=False 时该位姿即**法兰安装面**帧, 与 MoveJ/MoveL 的命令帧一致,
        可直接作为标定用的法兰位姿 (按 P 打印 → 粘到采集脚本)。
        """
        q = self.get_joint_q()
        homo = self.arm_model.forwardKinematics(q, 6)
        posture = unitree_arm_interface.homoToPosture(homo)
        return np.asarray(posture, dtype=float)

    def check_frame_consistency(self, pose_cmd=None, tol_mm=3.0, tol_deg=1.0):
        """校验 FK 读回帧与命令帧是否重合。

        ArmInterface(hasGripper=True) 会把 FK 末端帧前移 38.2mm (夹爪 TCP), 使
        get_end_posture() 与命令帧脱节; 此时若把读回值当作法兰位姿存进 poses.txt,
        手眼外参会沿接近轴整体偏 ~38mm, 且标定报告的一致性 RMS 依旧很好看
        (系统性偏差在各组间互相抵消), 极难事后发现。故启动时强制自检。

        pose_cmd: 参考命令位姿, 默认 self.target_pose。
        返回 (ok, pos_err_mm, rpy_err_deg)。
        """
        if pose_cmd is None:
            if not hasattr(self, "target_pose"):
                print("[自检] 跳过: 无命令参考位姿")
                return True, 0.0, 0.0
            pose_cmd = self.target_pose
        pose_cmd = np.asarray(pose_cmd, dtype=float)

        pose_fk = self.get_end_posture()
        d = pose_fk - pose_cmd
        pos_err_mm = float(np.linalg.norm(d[3:6]) * 1000.0)
        rpy_err_deg = float(np.degrees(np.abs(d[:3])).max())
        ok = pos_err_mm <= tol_mm and rpy_err_deg <= tol_deg

        if ok:
            print(f"[自检] 帧一致 ✅  读回−命令: {pos_err_mm:.2f} mm / {rpy_err_deg:.2f} deg")
            return ok, pos_err_mm, rpy_err_deg

        print("\n" + "!" * 64)
        print(f"[自检] 帧不一致 ❌  读回−命令: {pos_err_mm:.2f} mm / {rpy_err_deg:.2f} deg")
        print("  FK 读回帧与命令帧不重合, 位姿不可用于标定。")
        print("  最常见原因: ArmInterface(hasGripper=True) —— 末端装的是相机/自研手,")
        print("  没有 Unitree 夹爪, 该参数会让 FK 末端帧沿法兰 +X 前移 38.2mm。")
        print("  请检查本文件构造 ArmInterface 处是否为 hasGripper=False。")
        print("!" * 64 + "\n")
        return ok, pos_err_mm, rpy_err_deg

    def reset_target_from_current(self):
        """用 FK 读回值重建点动基准。

        仅在 hasGripper=False (读回帧 == 命令帧) 时安全; 否则会把 38.2mm 的
        帧偏差写进 target_pose, 第一次 MoveL 点动直接跳出去。
        """
        self.target_pose = self.get_end_posture()

    # ── 安全检查 ──────────────────────────────────────────

    def _check_pose_safe(self, pose):
        x, y, z = pose[3], pose[4], pose[5]
        if not (self.x_range[0] <= x <= self.x_range[1]):
            raise ValueError(f"X={x:.3f} 超出安全范围 [{self.x_range[0]:.3f}, {self.x_range[1]:.3f}]")
        if not (self.y_range[0] <= y <= self.y_range[1]):
            raise ValueError(f"Y={y:.3f} 超出安全范围 [{self.y_range[0]:.3f}, {self.y_range[1]:.3f}]")
        if not (self.z_range[0] <= z <= self.z_range[1]):
            raise ValueError(f"Z={z:.3f} 超出安全范围 [{self.z_range[0]:.3f}, {self.z_range[1]:.3f}]")

    def _make_pose(self, values):
        if len(values) == 6:
            pose = np.array(values, dtype=float)
        else:
            raise ValueError("需要 6 个数 (r p y x y z)")
        if not self.no_safe_check:
            self._check_pose_safe(pose)
        return pose

    # ── 运动方法 ──────────────────────────────────────────

    def move_delta(self, delta):
        """相对 MoveL 从 self.target_pose (键盘点动用)."""
        target = self.target_pose + delta
        if not self.no_safe_check:
            clamp_workspace(target, self.x_range, self.y_range, self.z_range)
        ok = self.arm.MoveL(target, 0.0, self.speed)
        if not ok:
            raise RuntimeError("MoveL IK 失败")
        self.target_pose = target

    def go_to_pose_safe(self, target):
        """直接 MoveJ 到目标 (关节空间插值, 只需终点 IK, 始终可达)."""
        ok = self.arm.MoveJ(target, 0.0, self.speed)
        if not ok:
            raise RuntimeError("MoveJ 到目标失败 (目标不可达)")
        self.target_pose = target.copy()

    def go_to_pose_direct(self, target):
        """当前位置直接 MoveL (微调, 可能 IK 失败)."""
        ok = self.arm.MoveL(target, 0.0, self.speed)
        if not ok:
            raise RuntimeError("MoveL IK 失败 (请用 pose 代替)")
        self.target_pose = target.copy()

    def go_to_start(self):
        """MoveJ 返回准备点."""
        print("\n[动作] MoveJ 到准备点...")
        ok = self.arm.MoveJ(self.start_pose, 0.0, self.speed)
        if not ok:
            raise RuntimeError("MoveJ 到准备点失败")
        time.sleep(0.3)
        self.reset_target_from_current()

    def go_to_zero(self):
        """backToStart 回机械零位."""
        print("\n[动作] backToStart 回零位...")
        self.arm.backToStart()
        time.sleep(1.0)
        self.reset_target_from_current()

    # ── 命令解析 (空格键进入) ─────────────────────────────

    def execute_command(self, cmd_line):
        """解析并执行命令, 返回结果字符串."""
        cmd_line = cmd_line.strip()
        if not cmd_line:
            return ""

        parts = cmd_line.replace(",", " ").split()
        cmd = parts[0].lower()

        try:
            # 退出
            if cmd in ("quit", "exit"):
                self.running = False
                return "退出中..."

            # 保持姿态
            if cmd in ("q", "hold"):
                print("[保持] 固定当前姿态, 电机使能保持. Ctrl+C 退出.")
                try:
                    while self.running:
                        time.sleep(0.5)
                except KeyboardInterrupt:
                    pass
                return "保持结束"

            # 回准备点
            if cmd in ("r", "home"):
                self.go_to_start()
                return "✅ 已到准备点"

            # 回零位
            if cmd == "zero":
                self.go_to_zero()
                return "✅ 已到机械零位"

            # 安全 pose
            if cmd == "pose":
                values = [float(p) for p in parts[1:]]
                if len(values) != 6:
                    return "ERROR: pose 需要 6 个数: pose r p y x y z"
                target = self._make_pose(values)
                print(f"  MoveJ 直接移动到 → {target.tolist()}", flush=True)
                self.go_to_pose_safe(target)
                return f"✅ Pose 到位: {target.tolist()}"

            # 直接 jogpose
            if cmd in ("jogpose", "movejpose"):
                values = [float(p) for p in parts[1:]]
                if len(values) != 6:
                    return "ERROR: jogpose 需要 6 个数: jogpose r p y x y z"
                target = self._make_pose(values)
                print(f"  MoveL → {target.tolist()}", flush=True)
                self.go_to_pose_direct(target)
                return f"✅ JogPose 到位: {target.tolist()}"

            # 设置步长
            if cmd == "step":
                if len(parts) >= 2:
                    self.step = float(parts[1])
                return f"XYZ 步长 = {self.step * 1000:.1f} mm"
            if cmd == "rpystep":
                if len(parts) >= 2:
                    self.rpy_step = float(parts[1])
                return f"RPY 步长 = {np.degrees(self.rpy_step):.1f} deg"
            if cmd == "speed":
                if len(parts) >= 2:
                    self.speed = float(parts[1])
                return f"速度 = {self.speed:.3f} m/s"

            # 帮助
            if cmd in ("help", "?"):
                return (
                    "命令:\n"
                    "  pose r p y x y z     直接MoveJ到目标\n"
                    "  jogpose r p y x y z  当前位置直接MoveL\n"
                    "  r / home             回到准备点\n"
                    "  zero                 回机械零位\n"
                    "  q / hold             保持, Ctrl+C退出\n"
                    "  step [mm]            设置XYZ步长\n"
                    "  rpystep [deg]        设置RPY步长\n"
                    "  speed [m/s]          设置速度\n"
                    "  quit                 退出"
                )

            raise ValueError(f"未知命令: '{cmd}' (输入 'help' 查看命令)")

        except Exception as exc:
            return f"ERROR: {exc}"

    # ── 关闭 ──────────────────────────────────────────────

    def shutdown(self):
        print("\n[关闭] 正在安全退出...")
        # 先关 socket 服务器, 避免残留文件
        if self.pose_server is not None:
            self.pose_server.stop()
            print("[关闭] Pose 服务器已停止")
        try:
            if not self.no_return_home:
                print("[关闭] backToStart 回零位...")
                self.arm.backToStart()
                time.sleep(1.5)
                self.arm.setFsm(unitree_arm_interface.ArmFSMState.PASSIVE)
            else:
                print("[关闭] 不自动回零, 保持电机使能。")
        except Exception as e:
            print(f"[关闭] 警告: {e}")
        finally:
            self.arm.loopOff()
            print("✅ 连接已安全关闭。")

    # ── 主循环 ────────────────────────────────────────────

    def run(self):
        print_help()
        last_status = ""
        last_error = ""

        try:
            with RawTerminal() as raw:
                while self.running:
                    # ── 状态显示 (原地刷新) ──
                    try:
                        posture = self.get_end_posture()
                        q = self.get_joint_q()
                        # 更新位姿服务器缓存 (供 verify/data_collect 脚本读取)
                        if self.pose_server is not None:
                            self.pose_server.update_posture(posture)
                        status = format_status(posture, q, self.step, self.rpy_step, self.target_pose)
                        if status != last_status:
                            print("\r" + status + " " * 12, end="", flush=True)
                            last_status = status
                    except Exception as exc:
                        msg = str(exc)
                        if msg != last_error:
                            print(f"\n[状态错误] {msg}")
                            last_error = msg

                    # ── 读取按键 ──
                    key = read_key(timeout=0.05)
                    if key is None:
                        continue

                    # ── 空格键 → 命令输入模式 ──
                    if key == ' ':
                        print()  # 换行, 断开状态行
                        cmd_line = raw.read_line("CMD> ")
                        if cmd_line.strip():
                            result = self.execute_command(cmd_line.strip())
                            if result:
                                print(result, flush=True)
                        last_status = ""  # 强制重新绘制状态行
                        continue

                    # ── 键盘点动 / 功能键 ──
                    key_lower = key.lower()
                    try:
                        if key_lower == 'x':
                            print("\n[退出] 收到退出命令。")
                            self.running = False
                            break

                        if key_lower == 'h':
                            self.go_to_zero()
                            last_status = ""
                            continue

                        if key_lower == 'f':
                            self.go_to_start()
                            last_status = ""
                            continue

                        if key_lower == 'p':
                            posture = self.get_end_posture()
                            q = self.get_joint_q()
                            print("\n" + format_status(posture, q, self.step, self.rpy_step, self.target_pose))
                            # 输出纯数字行, 可直接复制粘贴到终端3
                            print(f">>> {posture[0]:.6f} {posture[1]:.6f} {posture[2]:.6f} "
                                  f"{posture[3]:.6f} {posture[4]:.6f} {posture[5]:.6f}")
                            last_status = ""
                            continue

                        if key == '[':
                            self.step = max(0.001, self.step * 0.5)
                            print(f"\n[步长] XYZ={self.step * 1000:.1f}mm")
                            continue

                        if key == ']':
                            self.step = min(0.05, self.step * 2.0)
                            print(f"\n[步长] XYZ={self.step * 1000:.1f}mm")
                            continue

                        if key == '-':
                            self.rpy_step = max(0.005, self.rpy_step * 0.5)
                            print(f"\n[姿态步长] {np.degrees(self.rpy_step):.1f}deg")
                            continue

                        if key == '=':
                            self.rpy_step = min(0.35, self.rpy_step * 2.0)
                            print(f"\n[姿态步长] {np.degrees(self.rpy_step):.1f}deg")
                            continue

                        if key_lower in KEY_DELTAS:
                            delta = KEY_DELTAS[key_lower] * self.step
                            self.move_delta(delta)
                            last_status = ""
                            continue

                        if key_lower in RPY_KEY_DELTAS:
                            delta = RPY_KEY_DELTAS[key_lower] * self.rpy_step
                            self.move_delta(delta)
                            last_status = ""
                            continue

                    except Exception as exc:
                        print(f"\n[拒绝/失败] {exc}")

        except KeyboardInterrupt:
            print("\n[退出] Ctrl+C")
        finally:
            self.shutdown()


# ══════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Z1 混合控制器: 键盘点动 + 命令输入")
    parser.add_argument("--step", type=float, default=0.005, help="XYZ 步长 (米), 默认 0.005")
    parser.add_argument("--rpy-step", type=float, default=0.035, help="RPY 步长 (弧度), 默认 0.035 (~2°)")
    parser.add_argument("--speed", type=float, default=0.15, help="运动速度 (m/s), 默认 0.15")
    parser.add_argument("--start-pose", nargs=6, type=float,
                        metavar=("R", "P", "Y", "X", "Y", "Z"),
                        help="自定义准备点")
    parser.add_argument("--no-start-move", action="store_true", help="启动时不 MoveJ 到准备点")
    parser.add_argument("--no-safe-check", action="store_true", help="关闭工作空间安全检查")
    parser.add_argument("--no-return-home", action="store_true", help="退出时不回零位, 保持电机使能")
    parser.add_argument("--x-range", nargs=2, type=float, default=(0.05, 0.65), metavar=("MIN", "MAX"))
    parser.add_argument("--y-range", nargs=2, type=float, default=(-0.35, 0.35), metavar=("MIN", "MAX"))
    parser.add_argument("--z-range", nargs=2, type=float, default=(0.10, 0.60), metavar=("MIN", "MAX"))
    parser.add_argument("--pose-socket", type=str, nargs="?", const=DEFAULT_SOCKET_PATH,
                        default=DEFAULT_SOCKET_PATH, metavar="PATH",
                        help=f"Unix socket 位姿服务路径 (默认: {DEFAULT_SOCKET_PATH})")
    parser.add_argument("--no-pose-server", action="store_true",
                        help="禁用位姿 socket 服务器")
    return parser.parse_args()


def main():
    args = parse_args()
    # 位姿服务器路径: --no-pose-server 显式禁用, 否则用 --pose-socket 的值
    pose_server_path = None if args.no_pose_server else args.pose_socket
    ctrl = Z1HybridController(
        step=args.step,
        rpy_step=args.rpy_step,
        speed=args.speed,
        no_return_home=args.no_return_home,
        no_safe_check=args.no_safe_check,
        no_start_move=args.no_start_move,
        start_pose=args.start_pose,
        x_range=tuple(args.x_range),
        y_range=tuple(args.y_range),
        z_range=tuple(args.z_range),
        pose_server_path=pose_server_path,
    )
    ctrl.run()


if __name__ == "__main__":
    main()
