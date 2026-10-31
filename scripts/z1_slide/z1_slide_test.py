#!/usr/bin/python3.8
"""Z1 水平平移往复滑移测试 —— 指尖传感器测试的运动激励程序。

用法（三终端，仿真用 sim_ctrl、真机用 z1_ctrl，本脚本不变）:
    # 终端0 仿真: roslaunch unitree_gazebo z1.launch UnitreeGripperYN:=false \
    #              TipProbeYN:=true wname:=earth_table
    # 终端0 真机: ./z1_ctrl                (unitree_z1 SDK build 目录)
    # 终端1:     ./sim_ctrl 或 ./z1_ctrl   (z1_controller/build 目录)
    # 终端2:     python z1_slide_test.py --axis x --stroke 0.20 --speed 0.05 --cycles 5
    详见本文件夹 README.md。

流程: 启动 MoveZ1.py --server 子进程 → backToStart → MoveJ 到滑移起点 A
      → jogpose(MoveL) 在 A/B 之间往复 cycles 次 → 回零下电。
      姿态全程锁定不变（纯平移），每段行程记录 起止时间戳 + 命令/读回位姿 到 CSV。
      Ctrl+C 任意时刻中断 → 回零下电。

CSV 列: t, phase, stroke_dir, dur_s, cmd_r p y x y z, rb_r p y x y z (rb=读回)
说明: MoveZ1.py 协议是阻塞式的（运动完成才处理下一条 stdin），因此只能在行程
      端点采样，无法记录行程中的连续轨迹；对称性分析用各段耗时 dur_s。
"""

import argparse
import csv
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
Z1_PYTHON = "/usr/bin/python3.8"
MOVEZ1_SCRIPT = SCRIPT_DIR.parent / "MoveZ1.py"   # 本脚本在 scripts/z1_slide/ 下

# 与 MoveZ1.py 默认安全箱一致，用于本地预检（报错信息更直接）
SAFE_X = (0.15, 0.65)
SAFE_Y = (-0.50, 0.50)
SAFE_Z = (0.12, 0.60)

# 指尖探针锚点: URDF link06 系 [0.230, 0.0125, 0.003]（自研手中指尖伸直位, 手 base
# 前移 0.05 + URDF FK, compute_hand_geometry.py）。实测(2026-09-27, 4 位姿一致) SDK
# 命令帧在 link06 前方 0.0613 m(沿法兰+X), 故相对命令帧的锚点为 [0.169, 0.0125, 0.003]。
TIP_OFFSET = np.array([0.169, 0.0125, 0.003])
TABLE_Z = 0.002   # earth_table.world 桌面板顶面高度（基座系，对应真机桌面 z≈0）


def rpy_to_R(r, p, y):
    """Z1 位姿 [roll,pitch,yaw] → 旋转矩阵（Rz(y)@Ry(p)@Rx(r)，正 pitch=指尖下压）。"""
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class Z1Server:
    """MoveZ1.py --server 子进程管理（精简自 anydex_eih_grasp.py 的 Z1Server）。"""

    def __init__(self, speed):
        cmd = [
            Z1_PYTHON, str(MOVEZ1_SCRIPT), "--server",
            "--speed", f"{speed:.6f}",
        ]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
            start_new_session=True,  # 独立进程组，Ctrl+C 不传播到子进程
        )
        self._lines = queue.Queue()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._expect("READY", timeout=60)

    def _cmd(self, s):
        self.proc.stdin.write(s + "\n")
        self.proc.stdin.flush()

    def _read_loop(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            if line.startswith("[") or line.startswith("✅") or line.startswith("⚠"):
                print("  [Z1]", line, flush=True)
            self._lines.put(line)
        self._lines.put(None)

    def _expect(self, prefix, timeout=120):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
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
        self._cmd("pose " + " ".join(f"{v:.6f}" for v in rpyxyz))
        self._expect("DONE pose")

    def jogpose(self, rpyxyz):
        self._cmd("jogpose " + " ".join(f"{v:.6f}" for v in rpyxyz))
        self._expect("DONE jogpose")

    def getpose(self):
        self._cmd("getpose")
        resp = self._expect("POS ")
        return np.array([float(v) for v in resp.split()], dtype=float)

    def home(self, timeout=60):
        self._cmd("home")
        self._expect("DONE home", timeout=timeout)

    def quit(self, timeout=10):
        try:
            self._cmd("quit")
            self._expect("BYE", timeout=timeout)
            self.proc.wait(timeout=5)
        except Exception:
            self._kill()

    def _kill(self):
        try:
            self.proc.kill()
        except (ProcessLookupError, OSError):
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def parse_args():
    p = argparse.ArgumentParser(description="Z1 水平平移往复滑移测试")
    p.add_argument("--axis", default="x",
                   help="滑移方向：x、y，或逗号组合依次执行（如 x,y = 先X后Y，"
                        "每段换轴时自动经基准准备点过渡），默认 x")
    p.add_argument("--stroke", type=float, default=0.20,
                   help="总行程 A→B 距离，单位 m，默认 0.20")
    p.add_argument("--x", type=float, default=0.40, help="行程中点 x，默认 0.40")
    p.add_argument("--y", type=float, default=0.0, help="行程中点 y，默认 0.0")
    p.add_argument("--z", type=float, default=0.35, help="法兰固定高度 z，默认 0.35")
    p.add_argument("--rpy", nargs=3, type=float, default=[0.0, -0.094, 0.0],
                   metavar=("R", "P", "Y"),
                   help="全程锁定姿态 [roll pitch yaw] rad，默认 0 -0.094 0（掌心朝下）")
    p.add_argument("--speed", type=float, default=0.05,
                   help="MoveL 速度 m/s，默认 0.05")
    p.add_argument("--cycles", type=int, default=5,
                   help="往复循环次数（1 次 = A→B→A），默认 5")
    p.add_argument("--dwell", type=float, default=0.3,
                   help="端点停顿秒数，默认 0.3")
    p.add_argument("--log", default="",
                   help="CSV 输出路径，默认 logs/slide_{axis}_{时间戳}.csv")
    p.add_argument("--tip-print", action="store_true",
                   help="打印指尖探针（自研手中指尖伸直位）在端点处的估算高度")
    p.add_argument("--direct-switch", action="store_true",
                   help="多轴换轴时不回基准准备点，从上一轴终点直接 MoveL 斜线到"
                        "下一轴起点（贴面高度下指尖会保持接触斜向滑过桌面）；"
                        "首段仍经准备点进入")
    return p.parse_args()


def check_endpoints(args, pa, pb):
    for name, pose in (("A", pa), ("B", pb)):
        x, y, z = pose[3], pose[4], pose[5]
        if not (SAFE_X[0] <= x <= SAFE_X[1]):
            raise SystemExit(f"端点 {name} x={x:.3f} 超出安全范围 {SAFE_X}")
        if not (SAFE_Y[0] <= y <= SAFE_Y[1]):
            raise SystemExit(f"端点 {name} y={y:.3f} 超出安全范围 {SAFE_Y}")
        if not (SAFE_Z[0] <= z <= SAFE_Z[1]):
            raise SystemExit(f"端点 {name} z={z:.3f} 超出安全范围 {SAFE_Z}")


def main():
    args = parse_args()
    if args.stroke <= 0 or args.stroke > 0.4:
        raise SystemExit("--stroke 需在 (0, 0.4] m 内")
    if not (0.005 <= args.speed <= 0.3):
        raise SystemExit("--speed 需在 [0.005, 0.3] m/s 内")
    if args.cycles < 1:
        raise SystemExit("--cycles 至少 1")

    axes = [a.strip().lower() for a in args.axis.split(",") if a.strip()]
    for ax in axes:
        if ax not in ("x", "y"):
            raise SystemExit(f"--axis 仅支持 x、y 或逗号组合（如 x,y），收到: {ax}")

    segs = []   # (轴名, pose_a, pose_b, pa, pb)
    for ax in axes:
        half = args.stroke / 2.0
        if ax == "x":
            pa = [args.x - half, args.y]
            pb = [args.x + half, args.y]
        else:
            pa = [args.x, args.y - half]
            pb = [args.x, args.y + half]
        pose_a = list(args.rpy) + [pa[0], pa[1], args.z]
        pose_b = list(args.rpy) + [pb[0], pb[1], args.z]
        check_endpoints(args, pose_a, pose_b)
        segs.append((ax, pose_a, pose_b, pa, pb))

    log_path = Path(args.log) if args.log else (
        SCRIPT_DIR / "logs"
        / f"slide_{'_'.join(axes)}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[配置] 轴序={'→'.join(axes)}（每轴各 {args.cycles} 次） "
          f"行程={args.stroke:.3f}m z={args.z:.3f} rpy={args.rpy} "
          f"速度={args.speed:.3f}m/s 端点停顿={args.dwell}s")
    print(f"[日志] {log_path}")

    if args.tip_print:
        R = rpy_to_R(*args.rpy)
        tip_xyz = R @ TIP_OFFSET
        for ax, _, _, pa, pb in segs:
            for name, pxy in (("A", pa), ("B", pb)):
                base = np.array([pxy[0], pxy[1], args.z])
                tip = base + tip_xyz
                print(f"[指尖] [{ax.upper()}-{name}]: "
                      f"tip=({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f})"
                      f"  高于桌面 {1000 * (tip[2] - TABLE_Z):+.1f} mm"
                      f"  (触桌法兰 z≈{args.z - (tip[2] - TABLE_Z):.3f})")
        if abs(args.rpy[1]) < 0.3:
            print("[提示] 正 pitch=指尖下压；贴面滑移建议 pitch≈+0.9~1.1 rad、"
                  "法兰 z≈0.14~0.16（--tip-print 可复核）")
        p = args.rpy[1]
        print(f"[提示] 该 pitch({p:.2f}) 下探针触桌的法兰 z≈"
              f"{TABLE_Z + TIP_OFFSET[0] * np.sin(p) - TIP_OFFSET[2] * np.cos(p):.3f}")

    z1 = None
    t0 = time.monotonic()
    rows = []
    try:
        z1 = Z1Server(speed=args.speed)
        for si, (ax, pose_a, pose_b, pa, pb) in enumerate(segs):
            if si == 0 or not args.direct_switch:
                print(f"[执行] [{ax.upper()}] 移动到滑移起点 A（经基准准备点）...",
                      flush=True)
                # 'pose' 自带 MoveJ→基准准备点→MoveL：换轴时即“回到准备位置”
                z1.pose(pose_a)
            else:
                print(f"[执行] [{ax.upper()}] 不回准备点，直接斜线过渡到起点 A...",
                      flush=True)
                z1.jogpose(pose_a)
            rb = z1.getpose()
            if si == 0:
                t0 = time.monotonic()
            rows.append([time.monotonic() - t0, f"{ax.upper()}_START", 0, 0.0]
                        + list(args.rpy) + pa + [args.z] + rb.tolist())
            print(f"[执行] [{ax.upper()}] 到达 A: 读回 {np.round(rb, 4).tolist()}",
                  flush=True)

            for cyc in range(1, args.cycles + 1):
                for phase, target, pdir in ((f"{ax.upper()}_A_TO_B", pose_b, 1),
                                            (f"{ax.upper()}_B_TO_A", pose_a, -1)):
                    tb = time.monotonic()
                    z1.jogpose(target)
                    time.sleep(args.dwell)
                    rb = z1.getpose()
                    dur = time.monotonic() - tb - args.dwell
                    rows.append([time.monotonic() - t0, phase, pdir, dur]
                                + list(target) + rb.tolist())
                    print(f"[往复 {cyc}/{args.cycles}] {phase}: "
                          f"{dur:.2f}s, 读回 x={rb[3]:.4f} y={rb[4]:.4f}",
                          flush=True)
        print("[完成] 往复结束。")
    except KeyboardInterrupt:
        print("\n[中断] Ctrl+C，回零退出 ...")
    finally:
        if z1 is not None:
            try:
                rb = z1.getpose()
                rows.append([time.monotonic() - t0, "END", 0, 0.0]
                            + [np.nan] * 6 + rb.tolist())
            except Exception:
                pass
            with open(log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "phase", "stroke_dir", "dur_s",
                            "cmd_r", "cmd_p", "cmd_yaw", "cmd_x", "cmd_y", "cmd_z",
                            "rb_r", "rb_p", "rb_yaw", "rb_x", "rb_y", "rb_z"])
                w.writerows(rows)
            print(f"[日志] 已写入 {log_path} ({len(rows)} 行)")
            try:
                z1.home()
            except Exception:
                pass
            z1.quit()
    print("[退出] 安全关闭完成。")


if __name__ == "__main__":
    main()
