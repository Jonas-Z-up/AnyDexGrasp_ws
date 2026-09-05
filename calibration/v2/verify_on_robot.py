# coding=utf-8
"""
Z1 + D435i 手眼标定 真机在线验证脚本 (ChArUco / OpenCV 4.10)

原理 (离线复测的"活体版"):
  标定板固定不动。机械臂摆到任意位姿, 实时拍一帧 → ChArUco 检测 + PnP 得到
  板在相机系位姿 → 经 T_base_flange(实时命令位姿) @ T_flange_cam(标定结果) 链路
  推算标定板特征点在【基座系】的坐标。
  - 多个位姿测同一板: 推算的板坐标应聚在一起 (散布 = 手眼链路精度)
  - 推算的板坐标 vs 卷尺实测位置: 差值 = 绝对精度 (能暴露尺度/系统性偏差)

前置:
  终端1: z1_ctrl
  终端2: Move_Key.py          (提供 /tmp/z1_pose_server.sock 位姿服务, 需看到 帧一致✅)
  终端3: 本脚本 (object_locator 环境)

操作:
  机械臂移到一个新位姿 (板子在画面内) → 按空格采集推算
  换 3-5 个差异大的位姿重复 → 按 Q 退出看汇总

按 Q 退出后, 用汇总里"板中心"的基座坐标做到达测试:
  在 Move_Key 命令模式 pose r p y <板中心x> <板中心y> <板中心z+0.05>
  (保持当前 rpy, 先抬高 5cm 接近), 到位后目视/尺量法兰中心与板中心的偏差。

运行环境: object_locator (cv2 4.10 + pyrealsense2)
"""

import os
import json
import argparse

import cv2
import numpy as np
import pyrealsense2 as rs

from z1_pose_client import PoseClient

np.set_printoptions(precision=4, suppress=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def euler_angles_to_rotation_matrix(rx, ry, rz):
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def posture_to_T_base_flange(posture):
    """PoseClient 的 [r, p, yaw, x, y, z] → T_base_flange (与标定采集同一约定 Rz@Ry@Rx)"""
    r, p, yaw, x, y, z = posture
    T = np.eye(4)
    T[:3, :3] = euler_angles_to_rotation_matrix(r, p, yaw)
    T[:3, 3] = [x, y, z]
    return T


def main():
    parser = argparse.ArgumentParser(description="Z1 手眼标定真机在线验证 (ChArUco)")
    parser.add_argument("--result-json", default=None,
                        help="标定结果 JSON (默认 measured_data_in/calibration_report.json)")
    parser.add_argument("--intrinsics-json", default=None)
    parser.add_argument("--squares-x", type=int, default=9)
    parser.add_argument("--squares-y", type=int, default=9)
    parser.add_argument("--square-size", type=float, default=0.050)
    parser.add_argument("--marker-size", type=float, default=0.037)
    parser.add_argument("--show-window", action="store_true", help="显示带检测叠加的实时画面")
    args = parser.parse_args()

    # ── 标定结果 ──
    result_json = args.result_json or os.path.join(SCRIPT_DIR, 'measured_data_in', 'calibration_report.json')
    with open(result_json, "r", encoding="utf-8") as f:
        report = json.load(f)
    R_X = np.array(report["selected"]["rotation"], dtype=float)
    t_X = np.array(report["selected"]["translation"], dtype=float).reshape(3, 1)
    T_flange_cam = np.eye(4)
    T_flange_cam[:3, :3] = R_X
    T_flange_cam[:3, 3] = t_X.flatten()
    print(f"标定结果: {report['selected_method']} ({result_json})")
    print(f"t_flange_cam: {(t_X * 1000).flatten()} mm\n")

    # ── 内参 ──
    intrinsics_json = args.intrinsics_json or "/home/jonas/Projects/AnyDexGrasp_ws/config/realsense_intrinsics.json"
    with open(intrinsics_json, "r", encoding="utf-8") as f:
        intrinsics = json.load(f)
    camera_matrix = np.array([
        [intrinsics["fx"], 0.0, intrinsics["ppx"]],
        [0.0, intrinsics["fy"], intrinsics["ppy"]],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    dist_coeffs = np.asarray(intrinsics.get("distortion_coefficients", [0.0] * 5), dtype=float).reshape(1, -1)

    # ── 标定板 / 检测器 ──
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((args.squares_x, args.squares_y), args.square_size, args.marker_size, dictionary)
    detector = cv2.aruco.CharucoDetector(board)
    # 板坐标系下: 4 个极值 charuco 角点 (id 按行优先), 其均值为图案几何中心;
    # 角点 0 是最靠棋盘图案外角的内角点 (距图案外角各 1 格, 本板 50mm), 便于尺量。
    # 板坐标系原点 = 棋盘图案外角 (第一格黑格的外角), 即离线复测输出的板位置指这个点。
    inner_w, inner_h = args.squares_x - 1, args.squares_y - 1
    outer_corner_ids = [0, inner_w - 1, (inner_h - 1) * inner_w, inner_h * inner_w - 1]
    board_corner_obj = board.getChessboardCorners()  # (N,3) 每个角点的板系坐标

    # ── 相机 ──
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    print("相机已启动。")

    # ── 位姿服务 ──
    pose_client = PoseClient()
    posture = pose_client.get_posture()
    print(f"位姿服务已连接, 当前位姿 [r p y x y z]: {np.round(posture, 3).tolist()}\n")

    print("=" * 64)
    print("  真机在线验证: 换位姿 → 空格采集推算 → 多点统计散布")
    print("=" * 64)
    print("  空格: 采集一帧并推算板角点0/板中心的基座系坐标")
    print("  Q:    退出并打印汇总")
    print()

    corner0_hist, center_hist = [], []

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())

            display = color_image.copy()
            hint = [f"SPACE: 采集推算 (已 {len(center_hist)} 次)", "Q: 退出"]
            for i, line in enumerate(hint):
                cv2.putText(display, line, (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if args.show_window:
                # 实时叠加检测, 方便对位
                cc, cid, _, _ = detector.detectBoard(color_image)
                if cid is not None and len(cid) >= 4:
                    cv2.aruco.drawDetectedCornersCharuco(display, cc, cid)
            cv2.namedWindow("verify_on_robot", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.imshow("verify_on_robot", display)

            k = cv2.waitKey(1) & 0xFF

            if k == ord(' '):
                # 当前位姿 (命令帧 = FK 读回帧, hasGripper=False)
                posture = pose_client.get_posture()
                T_base_flange = posture_to_T_base_flange(posture)

                # 检测 + PnP
                cc, cid, _, _ = detector.detectBoard(color_image)
                if cid is None or len(cid) < 4:
                    print("  [X] 角点不足 4 个, 未记录。换个能看到板子的位姿再试。")
                    continue
                obj_points, img_points = board.matchImagePoints(cc, cid)
                solved, rvec, tvec = cv2.solvePnP(
                    obj_points, img_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if not solved:
                    print("  [X] solvePnP 失败, 未记录。")
                    continue

                T_cam_board = np.eye(4)
                T_cam_board[:3, :3] = cv2.Rodrigues(rvec)[0]
                T_cam_board[:3, 3] = tvec.flatten()
                T_base_board = T_base_flange @ T_flange_cam @ T_cam_board

                # 推算特征点: 板系坐标 → 基座系
                id_set = set(cid.flatten().tolist())
                c0 = None
                if 0 in id_set:
                    p0 = (T_base_board @ np.append(board_corner_obj[0], 1.0))[:3]
                    corner0_hist.append(p0)
                    c0 = p0
                outer_idx = [i for i, g in enumerate(outer_corner_ids) if g in id_set]
                if len(outer_idx) == 4:
                    center_board = np.mean(board_corner_obj[outer_corner_ids], axis=0)
                    pc = (T_base_board @ np.append(center_board, 1.0))[:3]
                    center_hist.append(pc)

                print(f"\n─ 第 {len(corner0_hist) + len(center_hist)} 次采集 ─")
                print(f"  位姿 [r p y x y z]: {np.round(posture, 3).tolist()}")
                print(f"  识别角点: {len(cid)}")
                if c0 is not None:
                    print(f"  角点0 (板左上内角) 基座系: {np.round(c0, 4).tolist()} m")
                else:
                    print("  ⚠ 本帧未检测到角点0")
                if len(center_hist) > 0:
                    print(f"  板中心 基座系:          {np.round(center_hist[-1], 4).tolist()} m")
                    # 运行中散布统计
                    for name, hist in (("角点0", corner0_hist), ("板中心", center_hist)):
                        if len(hist) >= 2:
                            arr = np.asarray(hist)
                            spread = np.linalg.norm(arr - arr.mean(axis=0), axis=1)
                            print(f"  {name} 已测 {len(hist)} 次, 散布 RMS="
                                  f"{np.sqrt((spread ** 2).mean()) * 1000:.2f} mm, "
                                  f"最大间距={np.linalg.norm(arr.max(0) - arr.min(0)) * 1000:.2f} mm")
                else:
                    print("  ⚠ 板外角未齐, 无法算板中心")

            elif k == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    # ── 汇总 ──
    print("\n" + "=" * 64)
    print("  汇总")
    print("=" * 64)
    for name, hist in (("角点0", corner0_hist), ("板中心", center_hist)):
        if not hist:
            print(f"  {name}: 无有效数据")
            continue
        arr = np.asarray(hist)
        mean = arr.mean(axis=0)
        spread = np.linalg.norm(arr - mean, axis=1)
        rms = float(np.sqrt((spread ** 2).mean()))
        print(f"  {name}: {len(hist)} 次")
        print(f"    基座系位置均值: {np.round(mean, 4).tolist()} m")
        print(f"    散布 RMS={rms * 1000:.2f} mm, 最大={spread.max() * 1000:.2f} mm")
        if len(hist) >= 3 and rms * 1000 < 5:
            print("    ✅ 多位姿推算高度聚合, 手眼链路一致性良好")
        elif rms * 1000 < 10:
            print("    ✅ 散布可接受 (<10mm)")
        else:
            print("    ⚠ 散布偏大, 检查是否采集期间板被碰动 / 位姿未停稳就采集")
    print()
    print("到达测试 (最终验收): 用上面『板中心』的 x y z, 在 Move_Key 命令模式执行")
    print("  pose <当前r> <当前p> <当前y> <板中心x> <板中心y> <板中心z + 0.05>")
    print("先抬高 5cm 接近, 到位后目视法兰正对板中心; 再逐步降 z 量法兰-板面距离。")
    print("再用卷尺量板中心相对 Z1 基座原点的实际位置, 与推算均值对比 (绝对精度)。")


if __name__ == "__main__":
    main()
