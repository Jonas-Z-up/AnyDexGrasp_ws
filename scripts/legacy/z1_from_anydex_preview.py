#!/usr/bin/env python3
import argparse
import json
import os
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation


Z1_WRAPPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yprxyz_to_z1.py")
Z1_PYTHON = "/usr/bin/python3.8"
DEFAULT_CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "calibration", "cam2base_current.json")

R_CAM2BASE = np.array([
    [-0.04563462, 0.99836490, 0.03442389],
    [0.99894416, 0.04578957, -0.00372603],
    [-0.00529620, 0.03421751, -0.99940038],
], dtype=float)
T_CAM2BASE = np.array([0.37268625, 0.00069477, 0.85300561], dtype=float)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert run_inspire_preview.py output into a conservative Z1 target pose."
    )
    parser.add_argument("--input", default="../logs/preview/latest_inspire_grasp.json")
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION_PATH,
                        help="JSON/NPZ file containing R_cam2base and t_cam2base.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--use-grasp-orientation", action="store_true")
    parser.add_argument("--fixed-rpy", nargs=3, type=float, default=(0.0, 0.665, 0.0),
                        metavar=("ROLL", "PITCH", "YAW"))
    parser.add_argument("--xyz-offset", nargs=3, type=float, default=(0.0, 0.0, 0.0),
                        metavar=("DX", "DY", "DZ"))
    parser.add_argument("--approach-z", type=float, default=0.12)
    parser.add_argument("--speed", type=float, default=0.08)
    parser.add_argument("--x-range", nargs=2, type=float, default=(-0.05, 0.65))
    parser.add_argument("--y-range", nargs=2, type=float, default=(-0.50, 0.50))
    parser.add_argument("--z-range", nargs=2, type=float, default=(0.12, 0.70))
    parser.add_argument("--movej", action="store_true")
    parser.add_argument("--start-home", action="store_true")
    parser.add_argument("--no-return-home", action="store_true", help="退出时不回零，保持当前位置（默认回零）")
    parser.add_argument("--no-safe-check", action="store_true")
    return parser.parse_args()


def check_range(name, value, limits):
    if not (limits[0] <= value <= limits[1]):
        raise ValueError(f"{name}={value:.4f} outside safe range [{limits[0]:.3f}, {limits[1]:.3f}]")


def load_grasp(path):
    with open(path, "r") as f:
        return json.load(f)


def load_calibration(path):
    if not path:
        return R_CAM2BASE, T_CAM2BASE
    if not os.path.exists(path):
        print(f"[WARNING] calibration file not found: {path}, using hardcoded defaults")
        return R_CAM2BASE, T_CAM2BASE
    if path.endswith(".npz"):
        data = np.load(path)
        return np.asarray(data["R_cam2base"], dtype=float), np.asarray(data["t_cam2base"], dtype=float)
    with open(path, "r") as f:
        data = json.load(f)
    return np.asarray(data["R_cam2base"], dtype=float), np.asarray(data["t_cam2base"], dtype=float)


def convert_pose(grasp, args):
    r_cam2base, t_cam2base = load_calibration(args.calibration)
    p_cam = np.asarray(grasp["translation"], dtype=float)
    r_cam = np.asarray(grasp["rotation_matrix"], dtype=float)
    p_base_raw = r_cam2base @ p_cam + t_cam2base
    p_base = p_base_raw + np.asarray(args.xyz_offset, dtype=float)
    p_base[2] += args.approach_z

    if args.use_grasp_orientation:
        rpy = Rotation.from_matrix(r_cam2base @ r_cam).as_euler("xyz", degrees=False)
    else:
        rpy = np.asarray(args.fixed_rpy, dtype=float)

    pose = np.array([rpy[0], rpy[1], rpy[2], p_base[0], p_base[1], p_base[2]], dtype=float)
    check_range("x", pose[3], args.x_range)
    check_range("y", pose[4], args.y_range)
    check_range("z", pose[5], args.z_range)
    return pose, p_base_raw


def execute_pose(pose, args):
    roll, pitch, yaw, x, y, z = pose
    cmd = [
        Z1_PYTHON,
        Z1_WRAPPER,
        f"{yaw:.6f}",
        f"{pitch:.6f}",
        f"{roll:.6f}",
        f"{x:.6f}",
        f"{y:.6f}",
        f"{z:.6f}",
        "--speed",
        f"{args.speed:.6f}",
        "--x-range",
        f"{args.x_range[0]:.6f}",
        f"{args.x_range[1]:.6f}",
        "--y-range",
        f"{args.y_range[0]:.6f}",
        f"{args.y_range[1]:.6f}",
        "--z-range",
        f"{args.z_range[0]:.6f}",
        f"{args.z_range[1]:.6f}",
    ]
    if args.movej:
        cmd.append("--movej")
    if args.start_home:
        cmd.append("--start-home")
    if args.no_return_home:
        cmd.append("--no-return-home")
    if args.no_safe_check:
        cmd.append("--no-safe-check")
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    grasp = load_grasp(args.input)
    pose, raw_xyz = convert_pose(grasp, args)

    print("AnyDexGrasp -> Z1 preview")
    print(f"input: {args.input}")
    print(f"calibration: {args.calibration}")
    print(f"grasp score: {grasp.get('score')}")
    print(f"grasp type: {grasp.get('grasp_type')}")
    print(f"raw grasp xyz in Z1 base: [{raw_xyz[0]:.4f}, {raw_xyz[1]:.4f}, {raw_xyz[2]:.4f}]")
    print(
        "target pose rpyxyz: "
        f"[{pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f}, "
        f"{pose[3]:.4f}, {pose[4]:.4f}, {pose[5]:.4f}]"
    )
    print(
        "wrapper yprxyz: "
        f"[{pose[2]:.4f}, {pose[1]:.4f}, {pose[0]:.4f}, "
        f"{pose[3]:.4f}, {pose[4]:.4f}, {pose[5]:.4f}]"
    )

    if not args.execute:
        print("dry-run only. Add --execute to send this approach pose to Z1.")
        return 0

    execute_pose(pose, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())