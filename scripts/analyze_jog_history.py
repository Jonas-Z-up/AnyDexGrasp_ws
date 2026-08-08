#!/usr/bin/env python3
"""读取 jog_calib_history.jsonl,分析系统性抓取偏差。

用法:
  python analyze_jog_history.py [history.jsonl]

数据来源:anydex_eih_grasp.py --calib-jog 空格记录的人工微调偏移。
每条 offset = 人工对准后的实际位姿 − 到达目标时的实际位姿(基座系)。
偏移越大 ⇒ 网络/标定/安装引入的系统性误差越大。

输出四类结论:
  1. 恒定偏置    → mean(offset.xyz)         ⇒ 建议 offset_xyz 调整
  2. 接近轴误差  → mean(offset·approach)     ⇒ 建议 hand_mount_z 调整
  3. 随深度变化  → offset 各分量对 depth_cam 线性拟合斜率显著 ⇒ R_c2f 旋转误差,
                   offset_xyz / hand_mount_z 都救不了,需重标定或闭环修正。
  4. 朝向残余    → mean(offset.rpy)          ⇒ 建议 offset_rpy_deg 调整(pitch 为主;
                   注: jog 绕法兰中心 vs config offset_rpy 绕掌心中心,小角度近似)。
"""

import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation


def load_records(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[警告] 跳过无法解析的行: {e}", file=sys.stderr)
    return recs


def _xyz(pose):
    return np.asarray(pose["xyz"], dtype=float)


def _rpy(pose):
    return np.asarray(pose["rpy"], dtype=float)


def _approach(rec):
    if rec.get("approach_base"):
        return np.asarray(rec["approach_base"], dtype=float)
    return Rotation.from_euler("xyz", _rpy(rec["after"])).as_matrix()[:, 0]


def analyze(path):
    recs = load_records(path)
    if not recs:
        print(f"无记录: {path}")
        return

    print(f"样本数: {len(recs)}   ({path})\n")

    header = (f"{'#':<4}{'type':>5}{'score':>7}{'depth':>7}{'回退':>5}"
              f"{'dx(mm)':>8}{'dy(mm)':>8}{'dz(mm)':>8}{'接近轴(mm)':>11}")
    print(header)
    print("-" * len(header))

    offsets = []          # (N,3) mm
    rpy_offsets = []      # (N,3) rad, 操作员绕法兰中心的 rpy 调整量（r/f 只动 pitch）
    projs = []            # (N,) mm, offset·approach
    depths = []           # (N,) m
    fallback_n = 0
    first_cfg = recs[0].get("cfg", {})

    for i, r in enumerate(recs):
        offset = (_xyz(r["after"]) - _xyz(r["before"])) * 1000.0
        approach = _approach(r)
        p = float(offset @ approach)
        depth = float(r.get("depth_cam", np.nan))
        fb = not bool(r.get("used_network", True))
        fallback_n += fb

        offsets.append(offset)
        rpy_offsets.append(_rpy(r["after"]) - _rpy(r["before"]))
        projs.append(p)
        depths.append(depth)

        print(f"{i+1:<4}{r.get('grasp_type', '?'):>5}{r.get('score', 0):>7.2f}"
              f"{depth:>7.3f}{'●' if fb else '':>5}"
              f"{offset[0]:>8.1f}{offset[1]:>8.1f}{offset[2]:>8.1f}{p:>11.1f}")

    offsets = np.asarray(offsets)
    rpy_offsets = np.asarray(rpy_offsets)
    projs = np.asarray(projs)
    depths = np.asarray(depths)

    print("\n===== 汇总 =====")

    # 1) 恒定偏置
    mean_off = offsets.mean(axis=0)
    std_off = offsets.std(axis=0)
    print(f"\n[1] 恒定偏置 offset.xyz (mm):")
    print(f"    均值 = [{mean_off[0]:+7.1f}, {mean_off[1]:+7.1f}, {mean_off[2]:+7.1f}]")
    print(f"    标准差 = [{std_off[0]:.1f}, {std_off[1]:.1f}, {std_off[2]:.1f}]")
    if std_off.max() > 5.0:
        print(f"    ⚠ 标准差偏大(>5mm) ⇒ 除恒定偏置外还有较大随机/未建模分量,"
              f"建议多采几组再看均值。")

    # 2) 接近轴误差 (hand_mount_z)
    p_mean = projs.mean()
    p_std = projs.std()
    print(f"\n[2] 接近轴投影 offset·approach (mm):")
    print(f"    均值 = {p_mean:+7.1f}   标准差 = {p_std:.1f}")
    mz_cfg = first_cfg.get("hand_mount_z")
    if mz_cfg is not None:
        mz_new = mz_cfg - p_mean / 1000.0
        print(f"    当前 hand_mount_z = {mz_cfg}")
        print(f"    → 建议 hand_mount_z = {mz_new:.4f}  "
              f"(mz_new = mz_cfg − mean(offset·approach);"
              f"人工向前推才对准 ⇒ 手偏浅 ⇒ 减小)")

    # 3) 随深度变化 (R_c2f 旋转误差)
    valid = ~np.isnan(depths) & (np.abs(depths) > 1e-6)
    if valid.sum() >= 3:
        print(f"\n[3] offset 各分量对 depth_cam 的线性拟合 (mm 每米深度):")
        d = depths[valid]
        max_slope = 0.0
        for j, name in enumerate(["X", "Y", "Z"]):
            slope, intercept = np.polyfit(d, offsets[valid, j], 1)
            max_slope = max(max_slope, abs(slope))
            print(f"    {name}: slope={slope:+6.1f} mm/m   intercept={intercept:+6.1f} mm")
        if max_slope > 30.0:
            print(f"    ⚠ 最大斜率 {max_slope:.0f} mm/m 显著 "
                  f"(≈ 深度 0.5m 处带来 {max_slope*0.5:.0f}mm 偏差)"
                  f" ⇒ R_c2f 旋转误差,offset_xyz/hand_mount_z 救不了,"
                  f"需重标定或闭环修正。")
        else:
            print(f"    (最大斜率 {max_slope:.0f} mm/m 较小,暂不指向旋转误差)")

    # 建议配置片段
    off_xyz = first_cfg.get("offset_xyz", [0.0, 0.0, 0.0])
    new_off_xyz = [float(off_xyz[j]) + mean_off[j] / 1000.0 for j in range(3)]
    print("\n===== 建议修改 (config/grasp_config.json) =====")
    print(f'    "offset_xyz": [{new_off_xyz[0]:.3f}, {new_off_xyz[1]:.3f}, {new_off_xyz[2]:.3f}],')
    if mz_cfg is not None:
        print(f'    "hand_mount_z": {mz_cfg - p_mean / 1000.0:.4f},')

    # 4) 朝向残余 (offset_rpy_deg, 度)：r/f 只动 pitch，均值显著才建议。
    #    注: jog 的 pitch 绕法兰中心, config offset_rpy 绕掌心中心——小角度下角度
    #    修正可迁移, 法兰位置摆动不同(±hand_mount_z·sinθ), 改后重新采集几组确认。
    mean_rpy = rpy_offsets.mean(axis=0) * 180.0 / np.pi
    std_rpy = rpy_offsets.std(axis=0) * 180.0 / np.pi
    print(f"\n[4] 朝向残余 offset.rpy (°):"
          f"  均值 = [{mean_rpy[0]:+6.2f}, {mean_rpy[1]:+6.2f}, {mean_rpy[2]:+6.2f}]"
          f"  标准差 = [{std_rpy[0]:.2f}, {std_rpy[1]:.2f}, {std_rpy[2]:.2f}]")
    off_rpy_deg = first_cfg.get("offset_rpy_deg", [0.0, 0.0, 0.0])
    if abs(mean_rpy[1]) > 0.5:
        new_pitch = float(off_rpy_deg[1]) + mean_rpy[1]
        print(f'    "offset_rpy_deg": [{off_rpy_deg[0]:.1f}, {new_pitch:.1f}, '
              f'{off_rpy_deg[2]:.1f}],  (pitch +{mean_rpy[1]:+.2f}°)')
    elif abs(mean_rpy[1]) > 0.05:
        print(f"    (pitch 均值 {mean_rpy[1]:+.2f}° <0.5°,暂不写 offset_rpy_deg)")
    for j, name in [(0, "roll"), (2, "yaw")]:
        if abs(mean_rpy[j]) > 0.5:
            print(f"    ⚠ {name} 均值 {mean_rpy[j]:+.2f}° 异常大——r/f 不调 {name},"
                  f"此残余可能来自 R_c2f 旋转误差或安装滚转,建议重标定而非 offset_rpy_deg")

    if fallback_n:
        print(f"\n(注: {fallback_n}/{len(recs)} 条为网络朝向回退样本,已标记●;"
              f"若回退样本过多,建议用网络朝向样本单独分析)")

    # 建议迭代
    print("\n提示: 修改 config 后重新采集几组,确认偏移收敛(均值趋近 0);")
    print("      若标准差仍大,才是随机分量(网络预测/机械臂重复性/深度噪声)。")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "logs", "jog_calib_history.jsonl")
    analyze(path)


if __name__ == "__main__":
    main()
