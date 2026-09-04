# coding=utf-8
"""
Z1 + RealSense D435i 眼在手上 (Eye-in-Hand) 标定脚本
专属 ChArUco 标定板版本 (深度适配 OpenCV 4.10)
"""

import os
import argparse
import json
import cv2
import numpy as np

np.set_printoptions(precision=8, suppress=True)

# OpenCV 4.10 字典映射
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

def pose_to_homogeneous_matrix(pose):
    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)
    return R, t

def save_matrices_to_txt(matrices_dict, output_dir, filename="all"):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{filename}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        for idx, (name, matrix) in enumerate(matrices_dict.items()):
            matrix = np.array(matrix)
            f.write(f"=== {name} ===\n[\n")
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
            rows, cols = matrix.shape
            for i in range(rows):
                f.write("[")
                for j in range(cols):
                    f.write(repr(matrix[i, j]))
                    if j < cols - 1:
                        f.write(", ")
                f.write("]" + (",\n" if i < rows - 1 else "\n"))
            f.write("]\n" + ("\n" if idx < len(matrices_dict) - 1 else ""))
    print(f"已保存数据到: {filepath}")

def process_arm_pose(arm_pose_file):
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
                R, t = pose_to_homogeneous_matrix(values)
                arm_poses[line_num] = (R, t)
        except ValueError:
            continue
    print(f"成功读取 {len(arm_poses)} 组机械臂位姿数据")
    return arm_poses

def camera_calibrate_charuco_410(images_path, board, detector, camera_matrix, dist_coeffs, show_corners=False):
    """基于 OpenCV 4.10 CharucoDetector 的位姿提取"""
    print("++++++++++ 开始 ChArUco 位姿估计 (OpenCV 4.10) ++++++++++++++")
    image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    valid_cam_poses = {}

    for f in image_files:
        idx = int(os.path.splitext(f)[0])
        img = cv2.imread(os.path.join(images_path, f))
        if img is None:
            continue

        # OpenCV 4.10 一步式检测
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(img)

        # 至少 4 个角点进行 PnP 求解
        if charuco_ids is not None and len(charuco_ids) >= 4:
            # 匹配 3D 物方坐标与 2D 像方坐标
            obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
            solved, rvec, tvec = cv2.solvePnP(
                obj_points, img_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if solved:
                valid_cam_poses[idx] = (rvec, tvec)
                print(f"  [√] 图片 {f} 识别成功 (识别角点数: {len(charuco_ids)})")

                if show_corners:
                    vis_img = img.copy()
                    cv2.aruco.drawDetectedCornersCharuco(vis_img, charuco_corners, charuco_ids)
                    cv2.drawFrameAxes(vis_img, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
                    cv2.imshow("ChArUco PnP", vis_img)
                    cv2.waitKey(200)
                continue

        print(f"  [X] 图片 {f} 识别角点不足，跳过。")

    cv2.destroyAllWindows()
    return valid_cam_poses

def make_transform(rotation, translation):
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return transform

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

def evaluate_hand_eye(rotation, translation, R_arm, t_arm, R_target, t_target):
    tcp_camera = make_transform(rotation, translation)
    base_targets = [
        make_transform(arm_r, arm_t) @ tcp_camera @ make_transform(target_r, target_t)
        for arm_r, arm_t, target_r, target_t in zip(R_arm, t_arm, R_target, t_target)
    ]
    positions = np.asarray([transform[:3, 3] for transform in base_targets])
    center = np.mean(positions, axis=0)
    position_errors = np.linalg.norm(positions - center, axis=1)

    rotation_center = mean_rotation(np.asarray([transform[:3, :3] for transform in base_targets]))
    rotation_errors = np.asarray([rotation_error_deg(rotation_center, transform[:3, :3]) for transform in base_targets])

    return {
        "position_rms_m": float(np.sqrt(np.mean(position_errors ** 2))),
        "position_max_m": float(np.max(position_errors)),
        "rotation_rms_deg": float(np.sqrt(np.mean(rotation_errors ** 2))),
        "rotation_max_deg": float(np.max(rotation_errors)),
    }

def hand_eye_calibrate(images_path, arm_pose_file, board, detector, camera_matrix, dist_coeffs, show_corners=False):
    valid_cam_poses = camera_calibrate_charuco_410(
        images_path, board, detector, camera_matrix, dist_coeffs, show_corners=show_corners
    )
    arm_poses_dict = process_arm_pose(arm_pose_file)

    # 按照实际有效索引对齐
    R_arm, t_arm, R_target, t_target = [], [], [], []
    for idx, (rvec, tvec) in valid_cam_poses.items():
        if idx in arm_poses_dict:
            r_arm_mat, t_arm_vec = arm_poses_dict[idx]
            R_arm.append(r_arm_mat)
            t_arm.append(t_arm_vec)
            R_target.append(cv2.Rodrigues(rvec)[0])
            t_target.append(tvec)

    print(f"\n有效对齐样本对数: {len(R_arm)}")
    if len(R_arm) < 5:
        print("错误: 有效样本对数过少，标定中止。")
        return None, None, None

    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    results = []
    for name, method in methods.items():
        try:
            rotation, translation = cv2.calibrateHandEye(R_arm, t_arm, R_target, t_target, method=method)
            if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
                continue
            metrics = evaluate_hand_eye(rotation, translation, R_arm, t_arm, R_target, t_target)
            metrics["method"] = name
            metrics["rotation"] = rotation.tolist()
            metrics["translation"] = np.asarray(translation).reshape(3).tolist()
            results.append((rotation, translation, metrics))
            print(f"{name}: 位置 RMS={metrics['position_rms_m'] * 1000:.2f} mm, 旋转 RMS={metrics['rotation_rms_deg']:.3f} deg")
        except Exception as e:
            print(f"{name} 求解失败: {e}")

    if not results:
        return None, None, None

    results.sort(key=lambda item: (item[2]["position_rms_m"] + np.radians(item[2]["rotation_rms_deg"]) * 0.1))
    rotation, translation, best = results[0]
    report = {"selected_method": best["method"], "sample_count": len(R_arm), "selected": best}
    print(f"\n+++++++++++ 选择最优算法: {best['method']} +++++++++++++++")
    return rotation, translation, report

def parse_args():
    parser = argparse.ArgumentParser(description="Z1 眼在手上手眼标定 (ChArUco - OpenCV 4.10)")
    parser.add_argument("--data-dir", default=None)
    # ChArUco 标定板参数
    parser.add_argument("--squares-x", type=int, default=7, help="横向方格数")
    parser.add_argument("--squares-y", type=int, default=9, help="纵向方格数")
    parser.add_argument("--square-size", type=float, default=0.035, help="方格物理边长 (米)")
    parser.add_argument("--marker-size", type=float, default=0.026, help="ArUco 码物理边长 (米)")
    parser.add_argument("--aruco-dict", type=str, default="DICT_4X4_50", help="ArUco 字典类型")
    parser.add_argument("--intrinsics-json", default=None)
    parser.add_argument("--show-corners", action="store_true")
    parser.add_argument("--max-position-rms", type=float, default=0.005)
    parser.add_argument("--max-rotation-rms", type=float, default=1.0)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))

    IMAGES_DIR = args.data_dir if args.data_dir else "collect_data_in"
    if not os.path.isabs(IMAGES_DIR):
        IMAGES_DIR = os.path.join(current_dir, IMAGES_DIR)

    ARM_POSE_FILE = os.path.join(IMAGES_DIR, 'poses.txt')
    INTRINSICS_JSON = args.intrinsics_json or "/home/jonas/Projects/AnyDexGrasp_ws/config/realsense_intrinsics.json"
    OUTPUT_DIR = os.path.join(current_dir, 'measured_data_in')

    # 读取相机内参
    with open(INTRINSICS_JSON, "r", encoding="utf-8") as f:
        intrinsics = json.load(f)
    camera_matrix = np.array([
        [intrinsics["fx"], 0.0, intrinsics["ppx"]],
        [0.0, intrinsics["fy"], intrinsics["ppy"]],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    dist_coeffs = np.asarray(intrinsics.get("distortion_coefficients", [0.0] * 5), dtype=float).reshape(1, -1)

    # OpenCV 4.10 标定板与检测器初始化
    dict_enum = ARUCO_DICT_MAP.get(args.aruco_dict, cv2.aruco.DICT_4X4_50)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_enum)
    board = cv2.aruco.CharucoBoard((args.squares_x, args.squares_y), args.square_size, args.marker_size, dictionary)
    detector = cv2.aruco.CharucoDetector(board)

    R, t, report = hand_eye_calibrate(
        IMAGES_DIR, ARM_POSE_FILE, board, detector, camera_matrix, dist_coeffs, show_corners=args.show_corners
    )

    if R is not None:
        print("\n最终标定矩阵:")
        print("R =\n", R)
        print("t =\n", t)

        CamPose = np.eye(4)
        CamPose[:3, :3] = R
        CamPose[:3, 3] = t.flatten()

        matrices = {"内参矩阵": camera_matrix, "畸变系数": dist_coeffs, "旋转矩阵": R, "平移向量": t, "相机位姿": CamPose}
        save_matrices_to_txt(matrices, OUTPUT_DIR, filename="all_candidate")
        
        with open(os.path.join(OUTPUT_DIR, "calibration_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)