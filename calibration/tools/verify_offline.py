# coding=utf-8
"""
Z1 + RealSense D435i 眼在手上 (Eye-in-Hand) 标定精度离线复测脚本 (ChArUco / OpenCV 4.10)

原理:
  标定板固定不动, 对每一组新采集:
    T_base_board[i] = T_base_flange[i] @ T_flange_camera @ T_camera_board[i]
  理论上所有 T_base_board[i] 应完全相同, 实际离散度 (RMS) = 标定泛化精度。

  与 hand_in_eye_ChArUco.py 内置评估的区别: 本脚本用的是【未参与标定】的新位姿,
  测的是泛化精度而非自洽性。

用法:
  1. 保持标定板不动 (换了位置也不影响本脚本, 但采集期间必须静止), 用
     data_collect_manual.py 采 8-10 组新位姿数据到 sessions/ 下新目录, 例如:
       cd calibration/tools && python data_collect_manual.py --out ../sessions/20260907_verify
  2. 运行 (默认读当前采用的解 + 最近一次复测会话):
       cd calibration/tools && python verify_offline.py
     也可显式指定: --data-dir ../sessions/<会话> --result-json <解json>

运行环境: object_locator (cv2 4.10)
"""

import os
import json
import argparse

import cv2
import numpy as np

np.set_printoptions(precision=6, suppress=True)

# OpenCV 4.10 字典映射 (与 hand_in_eye_ChArUco.py 保持一致)
ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
}


def euler_angles_to_rotation_matrix(rx, ry, rz):
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def process_arm_pose(arm_pose_file):
    """读取 poses.txt (x,y,z,rx,ry,rz 每行一组, 行号 = 图片编号)"""
    arm_poses = {}
    if not os.path.exists(arm_pose_file):
        print(f"错误: 找不到位姿文件 {arm_pose_file}")
        return arm_poses

    with open(arm_pose_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    for line_num, line in enumerate(lines):
        try:
            values = [float(v.strip()) for v in line.split(',')]
            if len(values) == 6:
                R = euler_angles_to_rotation_matrix(values[3], values[4], values[5])
                t = np.array(values[:3]).reshape(3, 1)
                arm_poses[line_num] = (R, t)
        except ValueError:
            continue
    print(f"成功读取 {len(arm_poses)} 组机械臂位姿数据")
    return arm_poses


def detect_board_poses(images_path, board, detector, camera_matrix, dist_coeffs, show_corners=False):
    """对每张图做 ChArUco 检测 + PnP, 返回 {图片编号: T_camera_board}"""
    image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    cam_poses = {}
    for f in image_files:
        idx = int(os.path.splitext(f)[0])
        img = cv2.imread(os.path.join(images_path, f))
        if img is None:
            continue

        charuco_corners, charuco_ids, _, _ = detector.detectBoard(img)
        if charuco_ids is not None and len(charuco_ids) >= 4:
            obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
            solved, rvec, tvec = cv2.solvePnP(
                obj_points, img_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if solved:
                T = np.eye(4)
                T[:3, :3] = cv2.Rodrigues(rvec)[0]
                T[:3, 3] = tvec.flatten()
                cam_poses[idx] = T
                print(f"  [√] 图片 {f} 识别成功 (识别角点数: {len(charuco_ids)})")
                if show_corners:
                    vis = img.copy()
                    cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
                    cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
                    cv2.imshow("verify", vis)
                    cv2.waitKey(200)
                continue

        print(f"  [X] 图片 {f} 识别角点不足，跳过。")

    cv2.destroyAllWindows()
    return cam_poses


def mean_rotation(rotations):
    u, _, vt = np.linalg.svd(np.sum(rotations, axis=0))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def rotation_error_deg(rotation_a, rotation_b):
    relative = rotation_a.T @ rotation_b
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def parse_args():
    parser = argparse.ArgumentParser(description="Z1 眼在手上标定精度离线复测 (ChArUco - OpenCV 4.10)")
    parser.add_argument("--data-dir", default="../sessions/20260906_verify_31", help="新采集数据目录 (含 *.jpg 与 poses.txt)")
    parser.add_argument("--result-json", default="../sessions/20260906_crosscheck/solve_more/calibration_report.json",
                        help="标定结果 JSON (默认为当前采用的解: 2026-09-06 自标定内参 31 组)")
    parser.add_argument("--intrinsics-json", default=None)
    # ChArUco 标定板参数 (与 hand_in_eye_ChArUco.py 默认值保持一致)
    parser.add_argument("--squares-x", type=int, default=9, help="横向方格数")
    parser.add_argument("--squares-y", type=int, default=9, help="纵向方格数")
    parser.add_argument("--square-size", type=float, default=0.050, help="方格物理边长 (米)")
    parser.add_argument("--marker-size", type=float, default=0.037, help="ArUco 码物理边长 (米)")
    parser.add_argument("--aruco-dict", type=str, default="DICT_4X4_50", help="ArUco 字典类型")
    parser.add_argument("--show-corners", action="store_true")
    parser.add_argument("--exclude-groups", default="",
                        help="剔除的组编号列表 (逗号分隔, 如 '5' 或 '5,8'), 用于剔除离群组后重算")
    return parser.parse_args()


def main():
    args = parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))

    DATA_DIR = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(current_dir, args.data_dir)
    RESULT_JSON = args.result_json or os.path.join(current_dir, 'measured_data_in', 'calibration_report.json')
    INTRINSICS_JSON = args.intrinsics_json or "/home/jonas/Projects/AnyDexGrasp_ws/config/realsense_intrinsics.json"
    ARM_POSE_FILE = os.path.join(DATA_DIR, 'poses.txt')

    # 1. 加载手眼标定结果 (求解脚本自动选优写入的那份)
    print(f"[1/4] 加载标定结果: {RESULT_JSON}")
    with open(RESULT_JSON, "r", encoding="utf-8") as f:
        report = json.load(f)
    R_X = np.array(report["selected"]["rotation"], dtype=float)
    t_X = np.array(report["selected"]["translation"], dtype=float).reshape(3, 1)
    print(f"      方法: {report['selected_method']}, 标定样本数: {report.get('sample_count', '?')}")
    print(f"      R_flange_cam:\n{R_X}")
    print(f"      t_flange_cam: {(t_X * 1000).flatten()} mm")

    # 2. 读取相机内参
    print(f"[2/4] 加载相机内参: {INTRINSICS_JSON}")
    with open(INTRINSICS_JSON, "r", encoding="utf-8") as f:
        intrinsics = json.load(f)
    camera_matrix = np.array([
        [intrinsics["fx"], 0.0, intrinsics["ppx"]],
        [0.0, intrinsics["fy"], intrinsics["ppy"]],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    dist_coeffs = np.asarray(intrinsics.get("distortion_coefficients", [0.0] * 5), dtype=float).reshape(1, -1)

    # 3. 检测新数据
    print(f"[3/4] 检测新采集数据: {DATA_DIR}")
    dict_enum = ARUCO_DICT_MAP.get(args.aruco_dict, cv2.aruco.DICT_4X4_50)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_enum)
    board = cv2.aruco.CharucoBoard((args.squares_x, args.squares_y), args.square_size, args.marker_size, dictionary)
    detector = cv2.aruco.CharucoDetector(board)
    cam_poses = detect_board_poses(DATA_DIR, board, detector, camera_matrix, dist_coeffs, args.show_corners)

    # 4. 按编号对齐并计算 T_base_board 离散度
    exclude = {int(v) for v in args.exclude_groups.split(',') if v.strip()}
    if exclude:
        print(f"\n按参数剔除组编号: {sorted(exclude)}")
    arm_poses_dict = process_arm_pose(ARM_POSE_FILE)
    T_base_boards, group_ids = [], []
    for idx, T_cam_board in cam_poses.items():
        if idx not in arm_poses_dict or idx in exclude:
            continue
        R_arm, t_arm = arm_poses_dict[idx]
        T_base_flange = np.eye(4)
        T_base_flange[:3, :3] = R_arm
        T_base_flange[:3, 3] = t_arm.flatten()
        T_flange_cam = np.eye(4)
        T_flange_cam[:3, :3] = R_X
        T_flange_cam[:3, 3] = t_X.flatten()
        T_base_boards.append(T_base_flange @ T_flange_cam @ T_cam_board)
        group_ids.append(idx)

    n = len(T_base_boards)
    if n < 5:
        print(f"\n错误: 有效对齐样本对数仅 {n} (<5), 复测中止。")
        return

    print(f"\n有效对齐样本对数: {n}")
    print(f"[4/4] 计算标定板位姿一致性 ...")

    T_base_boards = np.asarray(T_base_boards)
    positions = np.asarray([T[:3, 3] for T in T_base_boards])
    rotations = np.asarray([T[:3, :3] for T in T_base_boards])

    pos_center = np.mean(positions, axis=0)
    pos_errors = np.linalg.norm(positions - pos_center, axis=1)
    rot_center = mean_rotation(rotations)
    rot_errors = np.asarray([rotation_error_deg(rot_center, R) for R in rotations])

    pos_rms = float(np.sqrt(np.mean(pos_errors ** 2)))
    rot_rms = float(np.sqrt(np.mean(rot_errors ** 2)))

    print()
    print("=" * 65)
    print("  复测结果 (未参与标定的新位姿)")
    print("=" * 65)
    print(f"  数据组数:              {n}")
    print(f"  标定板在基座系位置均值: [{pos_center[0]:.4f}, {pos_center[1]:.4f}, {pos_center[2]:.4f}] m")
    print(f"  ─────────────────────────────────")
    print(f"  位置 RMS:              {pos_rms * 1000:.2f} mm")
    print(f"  位置最大/最小误差:      {pos_errors.max() * 1000:.2f} / {pos_errors.min() * 1000:.2f} mm")
    print(f"  旋转 RMS:              {rot_rms:.3f} deg")
    print(f"  旋转最大误差:          {rot_errors.max():.3f} deg")
    print("=" * 65)

    print("\n逐组偏差 (相对标定板位姿均值):")
    for gid, pe, re_ in zip(group_ids, pos_errors, rot_errors):
        marker = "  <-- ⚠ 离群嫌疑, 建议核对该组位姿记录" if pe * 1000 > 10 else ""
        print(f"  组 {gid:2d}: 位置 {pe * 1000:6.2f} mm, 旋转 {re_:5.3f} deg{marker}")

    print()
    if pos_rms * 1000 < 5:
        print("✅ 复测通过: 位置 RMS < 5mm, 与标定自洽水平相当, 标定泛化精度可信。")
    elif pos_rms * 1000 < 10:
        print("✅ 复测基本通过: 位置 RMS < 10mm, 可用, 但建议核对离群组后重采几组。")
    elif pos_rms * 1000 < 20:
        print("⚠️  复测一般: 位置 RMS < 20mm, 建议检查板是否被碰动/板面反光/位姿记录。")
    else:
        print("❌ 复测较差: 位置 RMS > 20mm, 检查采集数据或重新标定。")

    print("\n提示: 本方法测不出全体系统性偏移 (如 square-size 尺度误差),")
    print("      绝对精度请配合尺子实测标定板中心位置 (方法 2) 交叉验证。")


if __name__ == "__main__":
    main()
