#!/usr/bin/python3.8
"""
离线验证手眼标定精度 (不需要连接机械臂)

原理:
  标定板固定在世界中不动, 所以对于每一组采集:
    T_base_board[i] = T_base_tcp[i] @ T_tcp_camera @ T_camera_board[i]
  理论上所有 T_base_board[i] 应该相同。
  实际离散度 (RMS) = 标定精度。

用法:
  python3.8 verify_calibration_offline.py
"""

import os
import sys
import cv2
import numpy as np

np.set_printoptions(precision=6, suppress=True)

# 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# COLLECT_DIR = os.path.join(SCRIPT_DIR, "collect_data_in")
# RESULT_DIR = os.path.join(SCRIPT_DIR, "measured_data_in")
COLLECT_DIR = os.path.join(SCRIPT_DIR, "collect_data_in")
RESULT_DIR = os.path.join(SCRIPT_DIR, "measured_data_in_4")
CALIB_PARENT = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, CALIB_PARENT)


# ═══ 加载数据 ═══

def load_arm_poses(poses_file):
    """加载 poses.txt → R_arm, t_arm."""
    R_list, t_list = [], []
    with open(poses_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(v) for v in line.split(",")]
            x, y, z, rx, ry, rz = vals
            # 欧拉角 Rz@Ry@Rx → 旋转矩阵 (与标定脚本一致)
            cx, sx = np.cos(rx), np.sin(rx)
            cy, sy = np.cos(ry), np.sin(ry)
            cz, sz = np.cos(rz), np.sin(rz)
            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx
            R_list.append(R)
            t_list.append(np.array([x, y, z]))
    return R_list, t_list


def load_calib_camera(images_dir, pattern_cols, pattern_rows, square_size):
    """用采集的图片重新跑相机标定, 得到每张图的 rvecs/tvecs 和内参."""
    XX, YY, L = pattern_cols, pattern_rows, square_size
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    objp = np.zeros((XX * YY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:XX, 0:YY].T.reshape(-1, 2) * L

    obj_points, img_points = [], []
    image_files = sorted(
        [f for f in os.listdir(images_dir) if f.endswith(".jpg")],
        key=lambda x: int(x.replace("images", "").replace(".jpg", "")),
    )

    for f in image_files:
        img = cv2.imread(os.path.join(images_dir, f))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (XX, YY), None)
        if ret:
            obj_points.append(objp)
            img_points.append(cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria))

    if not obj_points:
        raise RuntimeError("没有找到任何棋盘格角点!")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )
    return rvecs, tvecs, mtx, dist


def load_calib_result(result_dir):
    """加载标定结果."""
    R = np.loadtxt(os.path.join(result_dir, "candidate_rotation_matrix.txt"))
    t = np.loadtxt(os.path.join(result_dir, "candidate_translation_vector.txt"))
    if t.ndim == 1:
        t = t.reshape(3, 1)
    return R, t


# ═══ 验证 ═══

def verify(R_arm, t_arm, rvecs, tvecs, R_X, t_X):
    """
    对每一组数据计算 T_base_board, 统计离散度.

    T_base_board[i] = T_base_tcp[i] @ T_tcp_camera @ T_camera_board[i]
    """
    N = len(R_arm)
    positions = []  # 标定板在基座坐标系下的位置

    for i in range(N):
        # T_base_tcp
        T_base_tcp = np.eye(4)
        T_base_tcp[:3, :3] = R_arm[i]
        T_base_tcp[:3, 3] = t_arm[i]

        # T_tcp_camera (标定结果)
        T_tcp_cam = np.eye(4)
        T_tcp_cam[:3, :3] = R_X
        T_tcp_cam[:3, 3] = t_X.ravel()

        # T_camera_board (相机标定得到的棋盘格外参)
        R_cb, _ = cv2.Rodrigues(rvecs[i])
        T_cam_board = np.eye(4)
        T_cam_board[:3, :3] = R_cb
        T_cam_board[:3, 3] = tvecs[i].ravel()

        # 标定板在基座下的位姿
        T_base_board = T_base_tcp @ T_tcp_cam @ T_cam_board
        positions.append(T_base_board[:3, 3].copy())

    positions = np.array(positions)
    centroid = np.mean(positions, axis=0)
    errors = np.linalg.norm(positions - centroid, axis=1)

    return positions, centroid, errors


# ═══ 主程序 ═══

if __name__ == "__main__":
    pattern_cols = 11
    pattern_rows = 8
    square_size = 0.012  # 12mm

    print("=" * 65)
    print("  离线验证: 标定板投影一致性检查")
    print("=" * 65)
    print(f"  棋盘格: {pattern_cols}×{pattern_rows} 内角点, {square_size*1000:.0f}mm 格子")
    print()

    # 1. 加载机械臂位姿
    poses_file = os.path.join(COLLECT_DIR, "poses.txt")
    print(f"[1/4] 加载机械臂位姿: {poses_file}")
    R_arm, t_arm = load_arm_poses(poses_file)
    print(f"      读取 {len(R_arm)} 组位姿")

    # 2. 重新标定相机 (得到 rvecs/tvecs)
    print(f"[2/4] 相机标定 (从 {COLLECT_DIR}/*.jpg) ...")
    rvecs, tvecs, mtx, dist = load_calib_camera(
        COLLECT_DIR, pattern_cols, pattern_rows, square_size
    )
    print(f"      有效图片: {len(rvecs)} 张")

    # 3. 加载手眼标定结果
    print(f"[3/4] 加载手眼标定结果: {RESULT_DIR}")
    R_X, t_X = load_calib_result(RESULT_DIR)
    print(f"      旋转矩阵 R:\n{R_X}")
    print(f"      平移向量 t: {t_X.ravel()}")

    # 4. 验证
    print(f"[4/4] 计算标定板投影一致性 ...")
    n = min(len(R_arm), len(rvecs))
    positions, centroid, errors = verify(R_arm[:n], t_arm[:n], rvecs[:n], tvecs[:n], R_X, t_X)

    rms = np.sqrt(np.mean(errors**2))
    max_err = np.max(errors)
    min_err = np.min(errors)
    std_err = np.std(errors)

    print()
    print("=" * 65)
    print("  验证结果")
    print("=" * 65)
    print(f"  数据组数:           {n}")
    print(f"  标定板质心 (基座):  [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}] m")
    print(f"  ─────────────────────────────────")
    print(f"  RMS 误差:           {rms*1000:.2f} mm")
    print(f"  最大误差:           {max_err*1000:.2f} mm")
    print(f"  最小误差:           {min_err*1000:.2f} mm")
    print(f"  标准差:             {std_err*1000:.2f} mm")
    print("=" * 65)

    # 逐组输出
    print()
    print("逐组误差 (mm):")
    for i, err in enumerate(errors):
        marker = " <-- ⚠" if err * 1000 > 10 else ""
        print(f"  组 {i:2d}: {err*1000:6.2f} mm{marker}")

    # 结论
    print()
    if rms * 1000 < 5:
        print("✅ 标定精度优秀 (RMS < 5mm)")
    elif rms * 1000 < 10:
        print("✅ 标定精度良好 (RMS < 10mm)")
    elif rms * 1000 < 20:
        print("⚠️  标定精度一般 (RMS < 20mm)")
    else:
        print("❌ 标定精度较差 (RMS > 20mm), 建议检查采集数据")
