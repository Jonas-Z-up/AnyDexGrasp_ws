#!/usr/bin/env python3
"""眼在手上 (Eye-in-Hand) 方案：将 AnyDexGrasp 抓取位姿转换到 Z1 基座坐标系并执行。

与眼在手外方案的关键区别：
  眼在手外: p_base = R_cam2base · p_cam + t_cam2base           (一步固定变换)
  眼在手上: p_flange = R_cam2flange · p_cam + t_cam2flange       (手眼标定 X 矩阵)
           p_base   = R_flange2base · p_flange + t_flange2base   (实时查询 Z1)

用法:
  # 仅预览目标位姿（不移动）
  python z1_from_anydex_preview_eih.py

  # 执行抓取动作
  python z1_from_anydez_preview_eih.py --execute

  # 先移动到观测位姿，再查询、计算、执行
  python z1_from_anydex_preview_eih.py --execute --move-to-observe
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation


# ── 路径常量 ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.join(SCRIPT_DIR, "..")

Z1_CONTROL_SCRIPT = os.path.join(SCRIPT_DIR, "MoveZ1.py")
Z1_PYTHON = "/usr/bin/python3.8"

DEFAULT_GRASP_INPUT = os.path.join(WS_ROOT, "logs", "preview", "latest_inspire_grasp.json")
DEFAULT_CALIB = os.path.join(WS_ROOT, "calibration", "cam2flange_current.json")

# 观测位姿: Z1 移动到该位姿后，相机可以拍摄桌面场景
# 格式: [roll, pitch, yaw, x, y, z]  (rad, m)
OBSERVE_POSE = [0.0, 1.057, 0.0, 0.3, 0.0, 0.35]


def parse_args():
    parser = argparse.ArgumentParser(
        description="AnyDexGrasp → Z1 眼在手上目标位姿转换与执行"
    )
    parser.add_argument("--input", default=DEFAULT_GRASP_INPUT,
                        help="run_inspire_preview.py 输出的 JSON 文件")
    parser.add_argument("--calibration", default=DEFAULT_CALIB,
                        help="眼在手上标定 JSON (cam2flange)")
    parser.add_argument("--execute", action="store_true",
                        help="实际移动机械臂，不带则仅预览")
    parser.add_argument("--move-to-observe", action="store_true",
                        help="执行前先移动到观测位姿")
    parser.add_argument("--observe-pose", nargs=6, type=float,
                        default=OBSERVE_POSE,
                        metavar=("R", "P", "Y", "X", "Y", "Z"),
                        help="观测位姿 roll pitch yaw x y z (默认: 0 1.057 0 0.3 0 0.35)")
    parser.add_argument("--use-grasp-orientation", action="store_true",
                        help="使用网络预测的抓取朝向（默认使用固定朝向）")
    parser.add_argument("--fixed-rpy", nargs=3, type=float,
                        default=(0.0, 0.665, 0.0),
                        metavar=("ROLL", "PITCH", "YAW"),
                        help="固定抓取朝向 (默认: 0 0.665 0)")
    parser.add_argument("--xyz-offset", nargs=3, type=float,
                        default=(0.0, 0.0, 0.0),
                        metavar=("DX", "DY", "DZ"),
                        help="抓取位置的 xyz 偏移 (m)")
    parser.add_argument("--approach-z", type=float, default=0.12,
                        help="抓取点的上方安全距离 (m)")
    parser.add_argument("--speed", type=float, default=0.08,
                        help="Z1 运动速度")
    parser.add_argument("--x-range", nargs=2, type=float,
                        default=(-0.05, 0.65))
    parser.add_argument("--y-range", nargs=2, type=float,
                        default=(-0.50, 0.50))
    parser.add_argument("--z-range", nargs=2, type=float,
                        default=(0.12, 0.70))
    return parser.parse_args()


# ── 加载 ──────────────────────────────────────────────────

def load_grasp(path):
    with open(path, "r") as f:
        return json.load(f)


def load_calibration(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"标定文件不存在: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    R = np.asarray(data["R_cam2flange"], dtype=float)
    t = np.asarray(data["t_cam2flange"], dtype=float)
    return R, t


# ── Z1 通信 ──────────────────────────────────────────────

class Z1PoseClient:
    """通过 MoveZ1.py --server 子进程查询位姿并发送运动指令."""

    def __init__(self, speed=0.08, x_range=(-0.05, 0.65),
                 y_range=(-0.50, 0.50), z_range=(0.12, 0.70)):
        self.speed = speed
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self._proc = None

    def _ensure_server(self):
        """启动 MoveZ1.py --server 子进程（如尚未启动）。"""
        if self._proc is not None and self._proc.poll() is None:
            return
        cmd = [
            Z1_PYTHON, Z1_CONTROL_SCRIPT, "--server",
            "--no-start-home", "--no-return-home",
            "--speed", f"{self.speed:.6f}",
            "--x-range", f"{self.x_range[0]:.6f}", f"{self.x_range[1]:.6f}",
            "--y-range", f"{self.y_range[0]:.6f}", f"{self.y_range[1]:.6f}",
            "--z-range", f"{self.z_range[0]:.6f}", f"{self.z_range[1]:.6f}",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # 等待服务端就绪
        for line in self._proc.stdout:
            if "READY" in line:
                break

    def _send_cmd(self, cmd_str):
        self._proc.stdin.write(cmd_str + "\n")
        self._proc.stdin.flush()

    def _read_response(self, prefix, timeout=10):
        """读取以 prefix 开头的行，返回该行去掉前缀后的内容。"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = self._proc.stdout.readline()
            if not line:
                continue
            line = line.strip()
            if line.startswith("ERROR"):
                raise RuntimeError(f"Z1 错误: {line}")
            if line.startswith(prefix):
                return line[len(prefix):].strip()
        raise TimeoutError(f"等待 Z1 响应 '{prefix}' 超时 ({timeout}s)")

    def get_flange_pose(self):
        """查询当前法兰末端位姿，返回 [roll, pitch, yaw, x, y, z] (rad, m)。"""
        self._ensure_server()
        self._send_cmd("getpose")
        resp = self._read_response("POS ")
        return np.array([float(v) for v in resp.split()], dtype=float)

    def move_to_pose(self, rpyxyz, approach_z=0.0):
        """先移动到上方安全点再移动到目标位姿 (jogpose = 直接 MoveL)。

        参数:
          rpyxyz: [roll, pitch, yaw, x, y, z] 目标位姿
          approach_z: 在目标上方额外的高度 (m)，先到这个上方点
        """
        self._ensure_server()
        r, p, y, x, y_pos, z = rpyxyz

        if approach_z > 0:
            # Phase 1: 移动到目标上方
            above = [r, p, y, x, y_pos, z + approach_z]
            self._send_cmd(
                "jogpose " + " ".join(f"{v:.6f}" for v in above)
            )
            self._read_response("DONE jogpose")

        # Phase 2: 移动到目标位姿
        self._send_cmd(
            "jogpose " + " ".join(f"{v:.6f}" for v in rpyxyz)
        )
        self._read_response("DONE jogpose")

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            self._send_cmd("quit")
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
            self._proc = None


# ── 坐标变换 ──────────────────────────────────────────────

def cam_to_base(pose_cam, R_cam2flange, t_cam2flange, flange_pose):
    """两步串联变换：相机坐标系 → 法兰坐标系 → Z1 基座坐标系。

    参数:
      pose_cam: (3,) ndarray — 抓取位置 (相机坐标系)
      R_cam2flange: (3,3) — 手眼标定旋转矩阵 X
      t_cam2flange: (3,) — 手眼标定平移向量 X
      flange_pose: (6,) — Z1 法兰 [roll, pitch, yaw, x, y, z]

    返回:
      pose_base: (3,) ndarray — 抓取位置 (基座坐标系)
      R_flange2base: (3,3) — 法兰到基座的旋转 (可用于朝向变换)
      t_flange2base: (3,) — 法兰到基座的平移
    """
    # 法兰位姿 6D → 旋转矩阵 + 平移
    r, p, y = flange_pose[0], flange_pose[1], flange_pose[2]
    R_flange2base = Rotation.from_euler("xyz", [r, p, y]).as_matrix()
    t_flange2base = flange_pose[3:6].copy()

    # Step 1: 相机 → 法兰
    p_flange = R_cam2flange @ pose_cam + t_cam2flange

    # Step 2: 法兰 → 基座
    p_base = R_flange2base @ p_flange + t_flange2base

    return p_base, R_flange2base, t_flange2base


def check_range(name, value, limits):
    if not (limits[0] <= value <= limits[1]):
        raise ValueError(
            f"{name}={value:.4f} 超出安全范围 [{limits[0]:.3f}, {limits[1]:.3f}]"
        )


# ── 主流程 ────────────────────────────────────────────────

def main():
    args = parse_args()

    # 1. 加载标定
    R_cam2flange, t_cam2flange = load_calibration(args.calibration)
    print(f"[标定] 加载: {args.calibration}")
    print(f"  R_cam2flange:\n{R_cam2flange}")
    print(f"  t_cam2flange: {t_cam2flange}")

    # 2. 加载抓取结果
    grasp = load_grasp(args.input)
    p_cam = np.asarray(grasp["translation"], dtype=float)
    R_cam = np.asarray(grasp["rotation_matrix"], dtype=float)

    print(f"\n[抓取] 加载: {args.input}")
    print(f"  score: {grasp.get('score')}")
    print(f"  grasp_type: {grasp.get('grasp_type')}")
    print(f"  p_cam: {p_cam}")
    print(f"  num_output_grasps: {grasp.get('num_output_grasps')}")

    if not args.execute:
        # 仅预览模式 — 需要有法兰位姿才能计算，但不需要实际连接
        print("\n⚠️  dry-run 模式: 无法查询 Z1 当前法兰位姿。")
        print("   加上 --execute 执行实际运动，或手动提供法兰位姿。")
        print(f"\n   观测位姿参考: {args.observe_pose}")
        print(f"   抓取位姿 (相机坐标系): p={p_cam}")
        return 0

    # 3. 连接 Z1
    client = Z1PoseClient(
        speed=args.speed,
        x_range=tuple(args.x_range),
        y_range=tuple(args.y_range),
        z_range=tuple(args.z_range),
    )

    try:
        # 4. 移动到观测位姿（可选）
        if args.move_to_observe:
            observe = np.array(args.observe_pose, dtype=float)
            print(f"\n[观测] 移动到观测位姿: {observe}")
            client.move_to_pose(observe)
            time.sleep(0.5)  # 等待机械臂稳定

        # 5. 查询当前法兰位姿
        flange_pose = client.get_flange_pose()
        print(f"\n[Z1] 当前法兰位姿: roll={flange_pose[0]:.4f} "
              f"pitch={flange_pose[1]:.4f} yaw={flange_pose[2]:.4f} "
              f"x={flange_pose[3]:.4f} y={flange_pose[4]:.4f} z={flange_pose[5]:.4f}")

        # 6. 坐标变换: 相机 → 基座
        p_base_raw, R_f2b, t_f2b = cam_to_base(
            p_cam, R_cam2flange, t_cam2flange, flange_pose
        )
        p_base = p_base_raw + np.array(args.xyz_offset, dtype=float)
        p_base[2] += args.approach_z

        print(f"\n[变换] 抓取位姿 (基座坐标系): x={p_base[0]:.4f} "
              f"y={p_base[1]:.4f} z={p_base[2]:.4f}")

        # 7. 确定朝向
        if args.use_grasp_orientation:
            # 链式变换: R_base = R_flange2base · R_cam2flange · R_cam
            R_base = R_f2b @ R_cam2flange @ R_cam
            rpy = Rotation.from_matrix(R_base).as_euler("xyz", degrees=False)
            print(f"  [朝向] 使用网络预测朝向 → rpy={rpy}")
        else:
            rpy = np.array(args.fixed_rpy, dtype=float)
            print(f"  [朝向] 使用固定朝向 → rpy={rpy}")

        target_pose = np.array([
            rpy[0], rpy[1], rpy[2],
            p_base[0], p_base[1], p_base[2],
        ], dtype=float)

        # 8. 安全检查
        check_range("x", target_pose[3], args.x_range)
        check_range("y", target_pose[4], args.y_range)
        check_range("z", target_pose[5], args.z_range)

        print(f"\n[目标] Z1 抓取位姿 (rpy xyz): "
              f"r={target_pose[0]:.4f} p={target_pose[1]:.4f} y={target_pose[2]:.4f} "
              f"x={target_pose[3]:.4f} y={target_pose[4]:.4f} z={target_pose[5]:.4f}")

        # 9. 执行运动
        print("\n[执行] 发送 Z1 运动指令...")
        client.move_to_pose(target_pose, approach_z=0.0)

        print("\n✅ AnyDexGrasp EIH 执行完成")
        return 0

    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
