#!/usr/bin/env python3
"""推导 hand_mount_z_own 初值 — 让旧链路(Inspire 查表)与新链路(自研手锚定)的
法兰位置在同一两指抓取下重合。

原理:
  旧: p_flange_old = p_palm_inspire − 0.09 × approach
  新: p_flange_new = p_base_own   − hz_own × approach
  手不动 ⇒ 两式相等 ⇒ hz_own = 0.09 + (p_palm_inspire − p_base_own)·approach

  取单位两指抓取系(T_wg = I, approach = X):
    p_palm_inspire = −R_lookup^T @ t_lookup   (查表项的逆复合, 见 IHG:325-333)
    p_base_own     = −R_own^T @ C_base        (own_hand_grasp 的锚定几何)

输出: 按 8 类型 × 3 宽度的 hz_own 表 + 均值±散布(建议值)。
判据: 结果在 0.06–0.13m; 跨类型散布 <25mm, 否则说明锚定几何有类型性错误。

用法: python own_hand/tools/derive_hand_mount_z_own.py [--old-hz 0.09]
"""

import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(WS_ROOT, "own_hand"))         # own_hand_grasp

from own_hand_grasp import envelope_data  # noqa: E402

LOOKUP_JSON = os.path.join(WS_ROOT, "assets", "inspire_urdf",
                           "width_12Dangle_6Dangle.json")
TYPE_NAMES = {1: "Ring", 2: "Prismatic_2_Finger", 3: "Prismatic_3_Finger",
              4: "Large_Diameter", 5: "Medium_Wrap", 6: "Tripod",
              7: "Sphere_3_Finger", 8: "Distal_Type"}
WIDTHS_CM = ["4.0", "6.0", "8.0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-hz", type=float, default=0.09,
                    help="旧链路 hand_mount_z (config 当前值, 默认 0.09)")
    args = ap.parse_args()

    with open(LOOKUP_JSON) as f:
        lookup = json.load(f)

    rows = []
    print(f"{'类型':<22}{'宽度':>5}{'Δ沿新approach(mm)':>17}{'ΔY(mm)':>9}{'ΔZ(mm)':>9}{'hz_own(mm)':>11}")
    for tid, name in TYPE_NAMES.items():
        for wcm in WIDTHS_CM:
            entry = lookup.get(name, {}).get(wcm)
            if entry is None:
                continue
            R_lk = np.array(entry["rotation"], dtype=float)
            t_lk = np.array(entry["translation"], dtype=float)
            p_palm = -R_lk.T @ t_lk                     # Inspire 掌心原点(抓取系)

            center, R_own = envelope_data(tid)          # R_own 列=抓取系[XYZ]在base系
            p_base = -R_own.T @ center                  # 自研手 base 原点(抓取系)

            # 接近轴修正(2026-09-07): 执行器 approach = R_flange[:,0]。
            #   旧链路: R_flange = R_grip←palm @ Rx(-90) ⇒ a_old = R_lk^T 的第1列
            #           (Inspire 法兰X, 相对抓取X倾斜~50°, 非纯抓取X!)
            #   新链路: mount_yaw_own=0 ⇒ a_new = R_own^T 的第1列 (= 手指方向base X)
            # 法兰位置重合: p_base − hz·a_new = p_palm − 0.09·a_old
            # ⇒ hz_own = 0.09·(a_old·a_new) + (p_base − p_palm)·a_new
            a_old = R_lk.T[:, 0]
            a_new = R_own.T[:, 0]
            d = p_base - p_palm
            hz = args.old_hz * float(np.dot(a_old, a_new)) + float(np.dot(d, a_new))
            d_proj = d - np.dot(d, a_new) * a_new       # 垂直于 a_new 的残差(由锚点吸收)
            rows.append((tid, float(wcm), hz * 1000,
                         d_proj[1] * 1000, d_proj[2] * 1000))
            print(f"Type {tid} {name:<12}{wcm:>6}{hz*1000:>14.1f}"
                  f"{d_proj[1]*1000:>10.1f}{d_proj[2]*1000:>10.1f}")

    hzs = np.array([r[2] for r in rows])
    print(f"\n[汇总] n={len(hzs)}  hz_own = {hzs.mean():.1f} ± {hzs.std():.1f} mm "
          f"(min {hzs.min():.1f} / max {hzs.max():.1f})")
    per_type = {}
    for r in rows:
        per_type.setdefault(r[0], []).append(r[2])
    spread = {t: f"{np.mean(v):.0f}" for t, v in per_type.items()}
    print(f"[按类型均值 mm] {spread}")
    print(f"\n[建议] config 写 hand_mount_z_own = {round(hzs.mean()/1000, 4)} "
          f"(散布 {hzs.max()-hzs.min():.0f}mm, 判据 <25mm; 残差交 jog 闭环)")
    if hzs.max() - hzs.min() > 25:
        print("[警告] 跨类型散布超 25mm —— 检查 own_hand_grasp 的类型接触定义!")


if __name__ == "__main__":
    main()
