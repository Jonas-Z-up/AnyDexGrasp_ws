# 标定数据

## 当前方案：眼在手上 (Eye-in-Hand)

相机安装在 Z1 机械臂末端，随机械臂一起运动。

### 使用的标定文件

| 文件 | 说明 |
|------|------|
| `cam2flange_current.json` | **当前使用** — Z1 相机到法兰变换矩阵 (HORAUD 方法)，包含 `R_cam2flange` 和 `t_cam2flange` |

### 坐标变换链

```
相机坐标系 (网络输出)  →  法兰坐标系 (cam2flange)  →  Z1 基座坐标系 (命令观测位姿 OBSERVE_RPYXYZ)
```

```python
# grasp_common.compute_grasp_target 内部逻辑
# Step 1: 相机 → 法兰 (手眼标定矩阵 X)
p_flange = R_cam2flange @ p_cam + t_cam2flange

# Step 2: 法兰 → 基座 (当前用命令观测位姿 OBSERVE_RPYXYZ 构造 R_flange2base/t_flange2base;
#          原用 getpose() 实时查询, 因 ~40mm FK 虚报误差已切换, 已确认正确不回退)
p_base = R_flange2base @ p_flange + t_flange2base
```

### 观测位姿

`0 1.057 0 0.3 0 0.35` (roll pitch yaw x y z) — Z1 移动到该位姿后进行场景拍摄和抓取计算。

---

## 备用方案：眼在手外 (Eye-to-Hand, 已废弃)

| 文件 | 说明 |
|------|------|
| `cam2base_current.json` | 旧方案 — 相机固定桌面的相机到基座变换矩阵。仅作备用，状态已标记为 deprecated。 |

---

## 历史标定数据

`measured_results/` 保存了本次 Z1 眼在手上的原始标定数据：

| 文件 | 说明 |
|------|------|
| `calibration_report.json` | 完整标定报告，含 HORAUD/PARK/TSAI/DANIILIDIS/ANDREFF 五种方法 |
| `calibration_summary.txt` | 标定摘要（内参矩阵 + 候选解） |
| `camera_pose_4x4.txt` | 4×4 相机位姿矩阵 |
| `rotation_matrix_3x3.txt` | 3×3 旋转矩阵 |
| `translation_vector_3x1.txt` | 3×1 平移向量 |
| `camera_intrinsics_3x3.txt` | 3×3 相机内参矩阵 |
| `distortion_coeffs.txt` | 畸变系数 |

---

## 重新标定

如需重新进行手眼标定，更新 `cam2flange_current.json` 中的 `R_cam2flange` 和 `t_cam2flange` 即可，无需修改其他代码。
