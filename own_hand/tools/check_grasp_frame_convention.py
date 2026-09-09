#!/usr/bin/env python3
"""抓取系构造约定校验 — 对比自研手 R_own 与 Inspire 查表 R_lk。

原理: 两只手抓同一物体、同一两指抓取系约定时,
  R_own (抓取系[XYZ]在自研手base系) 与 R_lk (抓取系在Inspire掌心系)
  之间的相对旋转 R_rel = R_own @ R_lk^T 若构造正确应 ≈ 绕抓取系X(接近轴)的
  滚转(拇指方位差 + 装配90°), **面外倾斜分量应小**。
  面外倾斜大 ⇒ 该类型的接触三角形构造与上游约定不符。

输出: 每类型的 (绕X滚转, 面外倾斜角) 表, 并给出构造判定。

用法: python own_hand/tools/check_grasp_frame_convention.py
"""

import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(WS_ROOT, "own_hand"))         # own_hand_grasp

import own_hand_grasp as ohg  # noqa: E402

LOOKUP_JSON = os.path.join(WS_ROOT, "assets", "inspire_urdf",
                           "width_12Dangle_6Dangle.json")
TYPE_NAMES = {1: "Ring", 2: "Prismatic_2_Finger", 3: "Prismatic_3_Finger",
              4: "Large_Diameter", 5: "Medium_Wrap", 6: "Tripod",
              7: "Sphere_3_Finger", 8: "Distal_Type"}
WIDTHS_CM = ["4.0", "6.0", "8.0"]


def decompose_rel(R_own, R_lk):
    """R_rel = R_own @ R_lk^T 分解: 绕抓取X的滚转 + 面外倾斜角。

    面外倾斜 = R_rel 的旋转轴与抓取X(grip系)的夹角的余角 →
    直接用: 把 R_rel 变换到 grip 系, 旋转轴 = 特征向量; 取轴与 X 的夹角。
    """
    R_rel = R_own @ R_lk.T
    r = Rotation.from_matrix(R_rel)
    axis_grip = r.as_rotvec()
    ang = np.linalg.norm(axis_grip)
    if ang < 1e-9:
        return 0.0, 0.0
    axis = axis_grip / ang
    # 轴在 grip 系: R_rel 的轴在 grip 坐标 = R_rel^T @ ... 简化: 轴旋转不变量
    # 直接用 rotvec 在 grip 系表达: rotvec 定义在原帧, 原帧=grip系已对齐?
    # R_own/R_lk 列均为 grip 轴在各自身基的表示, R_rel 作用相同基向量,
    # 故 rotvec 已近似处于"共同 grip 语义"下, 各分量即绕 grip X/Y/Z 的角。
    tilt_out_of_plane = np.degrees(np.arctan2(
        np.linalg.norm(axis_grip[1:]), abs(axis_grip[0])))
    roll_x = np.degrees(axis_grip[0])
    return roll_x, tilt_out_of_plane


def main():
    with open(LOOKUP_JSON) as f:
        lookup = json.load(f)

    print(f"{'类型':<24}{'宽度':>5}{'绕X滚转(°)':>11}{'面外倾斜(°)':>12}")
    summary = {}
    for tid, name in TYPE_NAMES.items():
        rows = []
        for wcm in WIDTHS_CM:
            entry = lookup.get(name, {}).get(wcm)
            if entry is None:
                continue
            R_lk = np.array(entry["rotation"], dtype=float)
            center, R_own = ohg.envelope_data(tid)
            roll, tilt = decompose_rel(R_own, R_lk)
            rows.append((roll, tilt))
            print(f"Type {tid:<2}{name:<21}{wcm:>6}{roll:>10.1f}{tilt:>12.1f}")
        if rows:
            summary[tid] = (np.mean([r[0] for r in rows]),
                            np.mean([r[1] for r in rows]))
    print("\n[按类型均值] (面外倾斜 <20° = 构造与上游约定一致)")
    for tid, (roll, tilt) in summary.items():
        mark = "✅" if tilt < 20 else ("⚠️" if tilt < 40 else "❌")
        print(f"  Type {tid} {TYPE_NAMES[tid]:<20} roll={roll:+7.1f}°  "
              f"tilt={tilt:5.1f}°  {mark}")


if __name__ == "__main__":
    main()
