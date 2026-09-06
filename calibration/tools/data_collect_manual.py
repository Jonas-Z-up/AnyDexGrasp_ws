#!/usr/bin/python3.8
"""
Z1 + D435i eye-in-hand 数据采集脚本（手动输入位姿版本）

- 不需要导入 Z1 SDK
- 不需要 ROS
- 机械臂位姿通过手动输入（与 ../Move_Key.py 命令模式发送的 pose/jogpose 参数一致）
- 按 S 键 → 粘贴位姿 → 自动保存图片 + poses.txt
- 按 Q 键退出

用法 (在 tools/ 下运行):
  python data_collect_manual.py                       # 存到 ../sessions/<今日日期>_collect/
  python data_collect_manual.py --out ../sessions/20260907_verify
                                                       # 指定会话目录

Z1 位姿格式: [roll, pitch, yaw, x, y, z]        ← 你输入/发送的格式
标定保存格式:  [x, y, z, rx, ry, rz]             ← hand_in_eye_ChArUco.py 期望的格式
两者旋转约定相同 (Rz@Ry@Rx)，仅数组重排即可。
"""

import os
import time
import argparse
import cv2
import numpy as np
import pyrealsense2 as rs

# ========== 配置 ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_WIDTH = 1280
IMG_HEIGHT = 720
FPS = 30

count = 0

# ========== 会话目录 ==========
parser = argparse.ArgumentParser(description="Z1+D435i 标定数据采集 (手动位姿)")
parser.add_argument("--out", default=None,
                    help="保存目录 (默认 ../sessions/<今日日期>_collect/)")
args = parser.parse_args()
if args.out:
    IMAGE_SAVE_PATH = args.out if args.out.endswith('/') else args.out + '/'
else:
    IMAGE_SAVE_PATH = os.path.join(SCRIPT_DIR, '..', 'sessions',
                                   time.strftime('%Y%m%d') + '_collect') + '/'

# ========== 初始化相机 ==========
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, FPS)
pipeline.start(config)


def z1_pose_to_calib_pose(z1_pose):
    """
    Z1 格式 [roll, pitch, yaw, x, y, z] → 标定格式 [x, y, z, rx, ry, rz]
    旋转约定一致 (Rz@Ry@Rx)，只做元素重排。
    """
    r, p, yaw, x, y_pos, z = z1_pose
    return [x, y_pos, z, r, p, yaw]


def data_collect():
    global count

    # 创建保存目录
    if not os.path.exists(IMAGE_SAVE_PATH):
        os.makedirs(IMAGE_SAVE_PATH)

    print("=" * 60)
    print("  Z1 + D435i  眼在手上 (Eye-in-Hand) 数据采集")
    print("=" * 60)
    print()
    print("操作流程:")
    print("  1. 在终端2 用 Move_Coordinate.py --server 发送 jogpose 移动机械臂")
    print("     格式: jogpose r p y x y z")
    print("  2. 等待机械臂到位后，在此窗口按 S 键")
    print("  3. 粘贴/输入刚才发送的 6 个数字 (r p y x y z)")
    print("  4. 重复 1-3，采集 15-25 组数据")
    print("  5. 按 Q 键退出")
    print()
    print("提示: 标定板需在画面不同位置/角度出现，覆盖全视野")
    print("=" * 60)
    print()

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # 在画面上叠加提示信息
        display_img = color_image.copy()
        hint_lines = [
            f"S: 保存 (已采集 {count} 组)",
            "Q: 退出",
        ]
        for i, line in enumerate(hint_lines):
            cv2.putText(display_img, line, (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL |
                                           cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow("detection", display_img)

        k = cv2.waitKey(1) & 0xFF

        if k == ord('s'):
            print(f"\n{'─' * 40}")
            print(f"📷 采集第 {count} 组数据")
            print(f"{'─' * 40}")

            # 手动输入位姿
            try:
                raw = input("请输入 Z1 位姿 [roll pitch yaw x y z]: ").strip()
                parts = raw.replace(",", " ").split()
                if len(parts) != 6:
                    print(f"❌ 需要 6 个数字，实际输入 {len(parts)} 个，跳过。")
                    continue

                z1_pose = [float(v) for v in parts]
            except ValueError as e:
                print(f"❌ 输入格式错误: {e}，跳过。")
                continue

            # 格式转换
            calib_pose = z1_pose_to_calib_pose(z1_pose)
            r, p, yaw, x, y_pos, z = z1_pose

            print(f"  Z1 输入:    [r={r:.4f}, p={p:.4f}, y={yaw:.4f}, "
                  f"x={x:.4f}, y={y_pos:.4f}, z={z:.4f}]")
            print(f"  标定保存:   [x={x:.4f}, y={y_pos:.4f}, z={z:.4f}, "
                  f"rx={r:.4f}, ry={p:.4f}, rz={yaw:.4f}]")

            # 保存位姿
            with open(f'{IMAGE_SAVE_PATH}poses.txt', 'a+') as f:
                pose_str = [str(v) for v in calib_pose]
                f.write(f'{",".join(pose_str)}\n')

            # 保存图片
            img_path = IMAGE_SAVE_PATH + str(count) + '.jpg'
            cv2.imwrite(img_path, color_image)

            print(f"  ✅ 已保存: {img_path}")
            print(f"  ✅ 已记录位姿到 poses.txt 第 {count + 1} 行")
            count += 1

        elif k == ord('q'):
            break

    pipeline.stop()
    cv2.destroyAllWindows()

    print(f"\n{'=' * 40}")
    print(f"采集结束，共 {count} 组数据")
    print(f"图片: {IMAGE_SAVE_PATH}*.jpg")
    print(f"位姿: {IMAGE_SAVE_PATH}poses.txt")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    try:
        data_collect()
    except KeyboardInterrupt:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\n中断退出，已采集 {count} 组数据。")
