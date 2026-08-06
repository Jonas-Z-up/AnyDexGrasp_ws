#!/usr/bin/python3.8
# 该程序是机械臂的“大脑运动中枢”：
# 1) 默认监听 /z1_arm/target_pose 话题
# 2) 使用 --manual 时可以在终端手动输入 x y z
# 3) 使用 --once x y z 时只执行一次移动
# 4) 使用 --server 时从 stdin 接收 move/home/quit 命令并保持连接
import argparse
import os
import sys
import time

import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray

# ==========================================
# Unitree Z1 SDK 库路径配置 - Python接口编译后在此目录
# 编译步骤：
# 1. 安装pybind11: sudo apt-get install python3-pybind11
# 2. 编译SDK: cd z1_sdk/build && cmake .. && make
# 3. 编译后会在lib目录生成unitree_arm_interface模块
# ==========================================
SDK_LIB_DIR = "/home/jonas/Projects/unitree_z1/Unitree-Z1_SDK/z1_sdk/lib"  # ← Python接口模块所在目录
if SDK_LIB_DIR not in sys.path:
    sys.path.insert(0, SDK_LIB_DIR)
os.environ["LD_LIBRARY_PATH"] = SDK_LIB_DIR + ":" + os.environ.get("LD_LIBRARY_PATH", "")
import unitree_arm_interface  # ← 编译后才能导入此模块


class Z1RosController:
    def __init__(self, use_ros=True, clearance=0.15, move_speed=0.25, safe_check=True,
                 return_home_on_shutdown=True, start_home=True,
                 x_range=(0.15, 0.65), y_range=(-0.50, 0.50), z_range=(0.12, 0.60)):
        self.clearance = clearance
        self.move_speed = move_speed
        self.safe_check = safe_check
        self.return_home_on_shutdown = return_home_on_shutdown
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

        print("\n[初始化] 启动 Z1 ROS 控制节点...")
        if use_ros:
            rospy.init_node('z1_pose_controller', anonymous=True)

        # 初始化机械臂接口
        self.arm = unitree_arm_interface.ArmInterface(hasGripper=True)
        self.arm_model = self.arm._ctrlComp.armModel  # FK 模型，用于查询当前末端位姿
        self.arm.loopOn()
        time.sleep(1)

        # 机械臂回零位（可通过 --no-start-home 跳过）
        if start_home:
            print("[状态] 正在返回初始零位...")
            self.arm.backToStart()
            time.sleep(2)
        else:
            print("[状态] 跳过启动回零，保持当前姿态。")
            # backToStart 不仅是回零，更是与 z1_ctrl 的首次 FSM 握手。
            # 跳过它时必须显式初始化 FSM，否则后续 MoveJ/MoveL 会永久阻塞。
            self.arm.setFsm(unitree_arm_interface.ArmFSMState.JOINTCTRL)
        
        # 定义基准准备点 [roll, pitch, yaw, x, y, z]。
        # 不在启动时自动移动到这里；需要时可手动发送完整 pose。
        self.z1_init_pose = np.array([0.0, -0.094, 0.0, 0.2, 0.0, 0.30], dtype=float)
        
        if use_ros:
            # 订阅目标位置话题。
            # 支持两种消息：
            #   3 个元素: [x, y, z]，自动加 clearance 到 z，上方点测试用
            #   6 个元素: [roll, pitch, yaw, x, y, z]，直接执行完整末端位姿
            self.pose_sub = rospy.Subscriber(
                "/z1_arm/target_pose",
                Float64MultiArray,
                self.pose_callback,
                queue_size=1,
            )
            print("[Z1] ✅ 初始化完成，正在监听话题: /z1_arm/target_pose")

    def pose_callback(self, msg):
        """
        ROS 回调函数：当收到新的目标点时触发
        """
        if len(msg.data) not in (3, 6):
            rospy.logwarn(f"⚠️ 收到的数据长度错误！期望为 3 或 6，实际为 {len(msg.data)}")
            return

        try:
            target_pose = self.make_target_pose(list(msg.data))
            rospy.loginfo(f"👉 [指令] 收到新目标位置，正在执行: {target_pose.tolist()}")
            self.move_to_pose(target_pose)
        except Exception as e:
            rospy.logerr(f"❌ [错误] 运动执行失败: {e}")

    def make_target_pose(self, values):
        if len(values) == 3:
            x, y, z = values
            pose = np.array([0.0, 0.0, 0.0, x, y, z + self.clearance], dtype=float)
        elif len(values) == 6:
            pose = np.array(values, dtype=float)
        else:
            raise ValueError("目标必须是 x y z 或 roll pitch yaw x y z")

        if self.safe_check:
            self.check_pose_safe(pose)
        return pose

    def check_pose_safe(self, pose):
        x, y, z = pose[3], pose[4], pose[5]
        if not (self.x_range[0] <= x <= self.x_range[1]):
            raise ValueError(f"x={x:.3f} 超出安全范围 [{self.x_range[0]:.2f}, {self.x_range[1]:.2f}]")
        if not (self.y_range[0] <= y <= self.y_range[1]):
            raise ValueError(f"y={y:.3f} 超出安全范围 [{self.y_range[0]:.2f}, {self.y_range[1]:.2f}]")
        if not (self.z_range[0] <= z <= self.z_range[1]):
            raise ValueError(f"z={z:.3f} 超出安全范围 [{self.z_range[0]:.2f}, {self.z_range[1]:.2f}]")

    def move_to_pose(self, target_pose, use_prepare=True):
        try:
            if use_prepare:
                print(f"[路径] 先移动至基准准备点: {self.z1_init_pose}", flush=True)
                ok_prepare = self.arm.MoveJ(self.z1_init_pose, 0.0, self.move_speed)
                if not ok_prepare:
                    raise RuntimeError("移动到基准准备点失败")
                time.sleep(0.3)

            # MoveL 是笛卡尔直线运动；如果 IK 失败，自动回退到 MoveJ。
            if use_prepare:
                print(f"[路径] 从基准准备点移动至目标: {target_pose}", flush=True)
            else:
                print(f"[路径] 从当前位置点动至目标: {target_pose}", flush=True)
            ok = self.arm.MoveL(target_pose, 0.0, self.move_speed)
            if not ok:
                print("[路径] MoveL 失败，尝试回退到 MoveJ 关节空间运动...", flush=True)
                ok = self.arm.MoveJ(target_pose, 0.0, self.move_speed)
                if not ok:
                    raise RuntimeError("MoveL 和 MoveJ 均失败，目标不可达")
                print("[路径] MoveJ 回退成功。", flush=True)
        except Exception as e:
            print(f"❌ [错误] 运动执行失败: {e}")
            raise

    def move_j(self, target_pose):
        """纯 MoveJ 关节空间运动 — 只需终点 IK，不会失败。"""
        print(f"[路径] MoveJ 关节空间运动至目标: {target_pose}", flush=True)
        ok = self.arm.MoveJ(target_pose, 0.0, self.move_speed)
        if not ok:
            raise RuntimeError("MoveJ 失败，目标不可达")
        print("[路径] MoveJ 完成。", flush=True)

    def manual_loop(self):
        print("\n[手动模式] 输入 x y z，程序会自动移动到 z + clearance 的上方点。")
        print("也可以输入 6 个数 roll pitch yaw x y z。输入 q 退出。")
        while True:
            text = input("target> ").strip()
            if text.lower() in ("q", "quit", "exit"):
                break
            parts = text.replace(",", " ").split()
            try:
                values = [float(p) for p in parts]
                target_pose = self.make_target_pose(values)
                print(f"[执行] {target_pose.tolist()}")
                self.move_to_pose(target_pose)
            except Exception as exc:
                print(f"❌ {exc}")

    def server_loop(self):
        print("\n[常驻模式] 机械臂保持连接，等待 stdin 命令。", flush=True)
        print("命令:", flush=True)
        print("  move x y z              移动到基座坐标上方点", flush=True)
        print("  pose r p y x y z         移动到完整末端位姿", flush=True)
        print("  jogpose r p y x y z      标定点动到完整末端位姿，不经过基准准备点", flush=True)
        print("  movejpose r p y x y z    同 jogpose，兼容别名", flush=True)
        print("  getpose                  查询当前法兰末端位姿 (r p y x y z)", flush=True)
        print("  home                     快速回零/准备点", flush=True)
        print("  quit                     退出", flush=True)
        print("READY", flush=True)
        while True:
            line = sys.stdin.readline()
            if line == "":
                break
            text = line.strip()
            if not text:
                continue

            parts = text.replace(",", " ").split()
            command = parts[0].lower()
            try:
                if command in ("q", "quit", "exit"):
                    print("BYE", flush=True)
                    break
                if command == "getpose":
                    pose = self.get_current_flange_pose()
                    print(f"POS {pose[0]:.6f} {pose[1]:.6f} {pose[2]:.6f} {pose[3]:.6f} {pose[4]:.6f} {pose[5]:.6f}", flush=True)
                    continue
                if command == "home":
                    self.go_home()
                    print("DONE home", flush=True)
                    continue
                if command == "move":
                    values = [float(p) for p in parts[1:]]
                    if len(values) != 3:
                        raise ValueError("move 需要 3 个数: move x y z")
                    target_pose = self.make_target_pose(values)
                    print(f"[执行] {target_pose.tolist()}", flush=True)
                    self.move_to_pose(target_pose)
                    print("DONE move", flush=True)
                    continue
                if command == "pose":
                    values = [float(p) for p in parts[1:]]
                    if len(values) != 6:
                        raise ValueError("pose 需要 6 个数: pose roll pitch yaw x y z")
                    target_pose = self.make_target_pose(values)
                    print(f"[执行] {target_pose.tolist()}", flush=True)
                    self.move_to_pose(target_pose)
                    print("DONE pose", flush=True)
                    continue
                if command in ("jogpose", "movejpose"):
                    values = [float(p) for p in parts[1:]]
                    if len(values) != 6:
                        raise ValueError("jogpose 需要 6 个数: jogpose roll pitch yaw x y z")
                    target_pose = self.make_target_pose(values)
                    print(f"[点动执行] {target_pose.tolist()}", flush=True)
                    self.move_to_pose(target_pose, use_prepare=False)
                    print("DONE jogpose", flush=True)
                    continue
                if command == "movej":
                    values = [float(p) for p in parts[1:]]
                    if len(values) != 6:
                        raise ValueError("movej 需要 6 个数: movej roll pitch yaw x y z")
                    target_pose = self.make_target_pose(values)
                    self.move_j(target_pose)
                    print("DONE movej", flush=True)
                    continue
                raise ValueError("未知命令，可用 move/home/pose/quit")
            except Exception as exc:
                print(f"ERROR {exc}", flush=True)

    def get_current_flange_pose(self):
        """查询 Z1 当前法兰末端位姿，返回 [roll, pitch, yaw, x, y, z] (rad, m)。

        通过读取关节角 → 正运动学 → homoToPosture 获得，与 Move_Key.py 中
        get_end_posture() 方法一致。
        """
        q = np.asarray(self.arm.lowstate.getQ(), dtype=float)
        homo = self.arm_model.forwardKinematics(q, 6)
        posture = unitree_arm_interface.homoToPosture(homo)
        return np.asarray(posture, dtype=float)

    def go_home(self):
        print("[状态] 返回初始零位/0点...", flush=True)
        self.arm.backToStart()
        time.sleep(1)

    def shutdown(self):
        """安全关闭机械臂连接"""
        print("\n\n[关闭] 节点终止。")
        if self.return_home_on_shutdown:
            print("[关闭] 安全复位中...")
            self.arm.backToStart()
            time.sleep(2)
        else:
            print("[关闭] 保持当前末端位置，不自动回零。")
        # 必须设置 PASSIVE 后再 loopOff，否则电机失能使能，机械臂掉力
        self.arm.setFsm(unitree_arm_interface.ArmFSMState.PASSIVE)
        self.arm.loopOff()
        print("✅ 连接已安全关闭。")


def parse_args():
    parser = argparse.ArgumentParser(description="Z1 输入坐标控制")
    parser.add_argument("--manual", action="store_true", help="终端手动输入坐标，不监听 ROS")
    parser.add_argument("--server", action="store_true", help="stdin 常驻命令模式，保持机械臂连接")
    parser.add_argument("--once", nargs="+", type=float, help="执行一次坐标。支持 x y z 或 roll pitch yaw x y z")
    parser.add_argument("--clearance", type=float, default=0.15, help="输入 x y z 时自动添加的上方高度，单位 m")
    parser.add_argument("--speed", type=float, default=0.25, help="MoveL 速度")
    parser.add_argument("--no-safe-check", action="store_true", help="关闭保守工作空间检查")
    parser.add_argument("--x-range", nargs=2, type=float, default=(0.15, 0.65), metavar=("MIN", "MAX"))
    parser.add_argument("--y-range", nargs=2, type=float, default=(-0.50, 0.50), metavar=("MIN", "MAX"))
    parser.add_argument("--z-range", nargs=2, type=float, default=(0.12, 0.60), metavar=("MIN", "MAX"))
    parser.add_argument("--no-return-home", action="store_true", help="退出时不自动回零，保持当前位置")
    parser.add_argument("--no-start-home", action="store_true", help="启动时不自动回零，保持当前姿态")
    parser.add_argument("--no-wait", action="store_true", help="--once 模式：移动完成后立即退出，不等待 Ctrl+C")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    use_ros = not args.manual and not args.server and args.once is None
    controller = Z1RosController(
        use_ros=use_ros,
        clearance=args.clearance,
        move_speed=args.speed,
        safe_check=not args.no_safe_check,
        return_home_on_shutdown=not args.no_return_home,
        start_home=not args.no_start_home,
        x_range=tuple(args.x_range),
        y_range=tuple(args.y_range),
        z_range=tuple(args.z_range),
    )
    try:
        if args.once is not None:
            target = controller.make_target_pose(args.once)
            print(f"[执行一次] {target.tolist()}")
            controller.move_to_pose(target)
            print("[完成] 已到达目标位置，机械臂保持当前姿态。", flush=True)
            if args.no_wait:
                print("[退出] --no-wait 模式，移动完成后立即退出。", flush=True)
            else:
                print("按 Ctrl+C 安全退出。", flush=True)
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
        elif args.server:
            controller.server_loop()
        elif args.manual:
            controller.manual_loop()
        else:
            # rospy.spin() 会让节点保持运行状态，等待回调函数被触发
            rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        controller.shutdown()
