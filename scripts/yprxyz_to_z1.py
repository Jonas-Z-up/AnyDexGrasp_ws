#!/usr/bin/env python3
import argparse
import os
import subprocess


CONTROL_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MoveZ1.py")
ROBOT_PYTHON = "/usr/bin/python3.8"


def parse_args():
    parser = argparse.ArgumentParser(
        description="输入 yaw pitch roll x y z，让 Z1 运动到指定末端位姿。"
    )
    parser.add_argument("yaw", type=float, help="yaw，单位 rad")
    parser.add_argument("pitch", type=float, help="pitch，单位 rad")
    parser.add_argument("roll", type=float, help="roll，单位 rad")
    parser.add_argument("x", type=float, help="x，单位 m")
    parser.add_argument("y", type=float, help="y，单位 m")
    parser.add_argument("z", type=float, help="z，单位 m")
    parser.add_argument("--speed", type=float, default=0.12, help="运动速度")
    parser.add_argument("--movej", action="store_true", help="使用关节路径 MoveJ；默认使用 MoveL")
    parser.add_argument("--start-home", action="store_true", help="启动时先回零；默认保持当前位置接管")
    parser.add_argument("--no-return-home", action="store_true", help="退出时不回零，保持当前位置（默认回零）")
    parser.add_argument("--no-safe-check", action="store_true", help="关闭安全范围检查")
    parser.add_argument("--no-wait", action="store_true", help="移动完成后立即退出，不等待 Ctrl+C")
    parser.add_argument("--x-range", nargs=2, type=float, default=(-0.05, 0.65), metavar=("MIN", "MAX"))
    parser.add_argument("--y-range", nargs=2, type=float, default=(-0.50, 0.50), metavar=("MIN", "MAX"))
    parser.add_argument("--z-range", nargs=2, type=float, default=(0.12, 0.70), metavar=("MIN", "MAX"))
    return parser.parse_args()


def main():
    args = parse_args()
    rpyxyz = [args.roll, args.pitch, args.yaw, args.x, args.y, args.z]
    cmd = [
        ROBOT_PYTHON,
        CONTROL_SCRIPT,
        "--once",
        *[f"{v:.6f}" for v in rpyxyz],
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
    if not args.start_home:
        cmd.append("--no-start-home")
    if args.no_return_home:
        cmd.append("--no-return-home")
    if args.no_safe_check:
        cmd.append("--no-safe-check")
    if args.no_wait:
        cmd.append("--no-wait")

    print(
        "[YPRXYZ] 输入 yaw pitch roll xyz = "
        f"{args.yaw:.6f} {args.pitch:.6f} {args.roll:.6f} "
        f"{args.x:.6f} {args.y:.6f} {args.z:.6f}"
    )
    print(
        "[YPRXYZ] 发送给 Z1 的 roll pitch yaw xyz = "
        f"{rpyxyz[0]:.6f} {rpyxyz[1]:.6f} {rpyxyz[2]:.6f} "
        f"{rpyxyz[3]:.6f} {rpyxyz[4]:.6f} {rpyxyz[5]:.6f}"
    )

    if args.movej:
        server_cmd = [
            ROBOT_PYTHON,
            CONTROL_SCRIPT,
            "--server",
            "--clearance",
            "0.000000",
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
        if not args.start_home:
            server_cmd.append("--no-start-home")
        if args.no_return_home:
            server_cmd.append("--no-return-home")
        if args.no_safe_check:
            server_cmd.append("--no-safe-check")
        server = subprocess.Popen(
            server_cmd,
            cwd=os.path.dirname(CONTROL_SCRIPT),
            stdin=subprocess.PIPE,
            text=True,
        )
        try:
            server.stdin.write("movejpose " + " ".join(f"{v:.6f}" for v in rpyxyz) + "\n")
            server.stdin.write("quit\n")
            server.stdin.flush()
            return server.wait()
        finally:
            if server.poll() is None:
                server.terminate()

    return subprocess.call(cmd, cwd=os.path.dirname(CONTROL_SCRIPT))


if __name__ == "__main__":
    raise SystemExit(main())