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

# Step 2: 法兰 → 基座 (用命令观测位姿 OBSERVE_RPYXYZ 构造 R_flange2base/t_flange2base。
#          历史注: 曾用 getpose() 查询, 观测到 ~38.2mm 偏差并记为"FK 虚报";
#          2026-09-04 定位真实根因 = ArmInterface(hasGripper=True) 把 FK 末端帧
#          前移 38.2mm 当夹爪 TCP。Move_Key.py / MoveZ1.py 已改 hasGripper=False,
#          读回帧 == 命令帧, 两种方式现已等价。用命令位姿仍更稳(不受关节柔顺影响)。
p_base = R_flange2base @ p_flange + t_flange2base
```

### 观测位姿

`0 1.057 0 0.3 0 0.35` (roll pitch yaw x y z) — Z1 移动到该位姿后进行场景拍摄和抓取计算。

---

## 标定程序

手眼标定工具链已从 `Robotic_Hand/Robotic_Hand/HandEyeCalibrate` 迁入本目录：

| 文件 | 作用 |
|------|------|
| `Move_Key.py` | Z1 键盘点动控制器（直连 SDK + Unix-socket 位姿服务 `/tmp/z1_pose_server.sock`） |
| `z1_pose_client.py` | Move_Key 的位姿 IPC 共享模块（须与 Move_Key.py 同目录） |
| `hand_in/data_collect_manual.py` | D435i 图像 + 机械臂位姿采集（输出到 `hand_in/collect_data_in/`） |
| `hand_in/hand_in_eye_calibrate_by_aq.py` | 主求解器（HORAUD/PARK/TSAI/DANIILIDIS/ANDREFF 五法对比，输出到 `hand_in/measured_data_in_4/`） |
| `hand_in/verify_calibration_offline.py` | 用采集数据离线复验标定结果 |

说明：
- **运行环境：`object_locator`**（cv2 4.10）。求解一步必须用它——`anydex` 的 cv2 是 5.0，既没有 `cv2.calibrateHandEye`，`solvePnP` 还会在部分尺度下发散到错误局部解（实测 12mm 时重投影 138px；cv2 4.10 同数据 0.5px）。Move_Key/数据采集在 anydex 也能跑，但统一用 object_locator 最省心
- `Move_Key.py` 通过绝对路径加载 Z1 SDK（`~/Projects/unitree_z1/.../z1_sdk/lib`），不受本目录位置影响
- `hand_in/` 内脚本均用相对本脚本的路径，采集数据落在 `hand_in/collect_data_in/`，求解输出落在 `hand_in/measured_data_in_4/`
- 内参默认取 `../config/realsense_intrinsics.json`（与抓取管线共用）

---

## 重新标定操作流程

### 0. 标定前必查：棋盘格边长（只能尺量）

```bash
# 先用尺量棋盘格实际边长！ --square-size 默认 0.012 (12mm) 只是历史假设值
python hand_in/hand_in_eye_calibrate_by_aq.py --help   # 确认 --square-size / --pattern-cols / --pattern-rows
```

⚠️ 图片**无法**自检边长：重投影误差对棋盘尺度不敏感（实测 6-40mm 全部 ~0.5px，平面 PnP 尺度不变）。
但边长错多少，PnP 解出的平移就按比例带偏 `t_cam2flange`，只能靠尺量排除。
（旁证：上次 12mm 假设下手眼一致性 RMS 7.7mm 属健康水平，尺度大错会显著放大该值，故尺度偏差估计在 ~10% 以内。）

### 1. 终端1：Z1 底层控制器

```bash
cd ~/Projects/unitree_z1/Unitree-Z1_SDK/z1_controller/build && ./z1_ctrl
```

### 2. 终端2：机械臂点动控制器

```bash
conda activate object_locator
cd calibration
python Move_Key.py --speed 0.2     # 自动 MoveJ 到准备点 [0.075, 1.057, 0.073, 0.327, -0.08, 0.35]
```

⚠️ 启动后必须看到 **`[自检] 帧一致 ✅`** 才能继续。若报"帧不一致"，说明读回帧 ≠
命令帧（典型是 `hasGripper` 配错，会让 FK 末端帧前移 38.2mm 当夹爪 TCP），此时
`P` 打印的位姿是错的，**存进 poses.txt 会让外参沿接近轴整体偏 ~38mm**，且标定报告的
一致性 RMS 照样好看（系统性偏差在各组间抵消），事后极难发现。

键盘点动：`W/S` ±X（末端系）、`A/D` ±Y、`Q/E` ±Z、`I/K` Roll、`J/L` Pitch、`U/O` Yaw、`P` 打印位姿、`F` 回准备点、`H` 回零、`[` `]`/`-` `=` 步长缩放、`X` 退出。
按**空格**进入命令模式：`pose r p y x y z`（MoveJ 跳转，推荐）、`jogpose ...`（MoveL 微调）、`step 0.01`、`speed 0.2`、`quit`。退出自动 backToStart 回零。

### 3. 终端3：数据采集

```bash
conda activate object_locator
cd calibration/hand_in
python data_collect_manual.py    # 按 S → 粘贴终端2 的 6 个数 (r p y x y z) → 存图+poses.txt；Q 退出
```

### 4. 循环采集 15-25 组

推荐混合操作：命令模式 `pose` 跳到目标附近 → 键盘微调对位 → `P` 打印 → 终端3 按 `S` 粘贴保存。
要点：标定板覆盖视野不同位置/角度；画面清晰无运动模糊。

历史文档记录的激励位姿序列（`pose r p y x y z`，标注 1-20 组、实际收录 18 条；第 14 条 y=-0.520 疑似原文笔误，采集时自行斟酌）：

```
# 1-3 组 (基准与初步变位)
pose 0.075 1.057 0.073 0.327 -0.08 0.35
pose 0.075 1.057 -0.15 0.35 0.08 0.25
pose -0.075 1.057 0.2 0.327 -0.1 0.35
# 4-9 组 (四角极限与中心，增加 Y 轴微调以引入深度变化)
pose 0.000 1.057 0.000 0.340 0.000 0.300
pose 0.100 0.920 0.100 0.3 -0.050 0.320
pose -0.100 1.040 0.100 0.250 0.050 0.400
pose 0.100 1.040 -0.100 0.350 -0.100 0.250
pose -0.300 1.040 -0.300 0.360 0.100 0.200
pose 0.000 1.070 0.150 0.300 -0.120 0.380
# 10-15 组 (强化 Z 轴和 X 轴的独立平移，配合小幅旋转)
pose -0.050 1.070 -0.200 0.380 0.120 0.220
pose 0.080 1.030 0.050 0.350 -0.050 0.250
pose -0.080 1.080 0.050 0.310 0.050 0.350
pose 0.020 1.050 0.180 0.280 -0.150 0.320
pose -0.020 1.060 -0.520 0.420 0.150 0.220
pose 0.060 1.057 0.000 0.370 0.000 0.280
# 16-20 组 (极端姿态激励，确保旋转矩阵 R 的多向性)
pose -0.060 1.057 0.000 0.290 0.000 0.320
pose 0.000 1.045 0.120 0.360 -0.080 0.380
pose 0.000 0.98 -0.080 0.350 0.020 0.190
```

### 5. 求解（必须 object_locator）

```bash
conda activate object_locator
cd calibration/hand_in
python hand_in_eye_calibrate_by_aq.py --square-size 0.012   # ← 换成尺量值！
# 可选: --data-dir <目录> --intrinsics-json <路径> --show-corners
# 五种方法 RMS 对比后自动选优，输出 hand_in/measured_data_in_4/calibration_report.json
```

自检：日志里的 `PnP 重投影误差` 应为 **~0.5px** 量级；若飙到几十上百 px，说明 solvePnP 发散（cv2 5.0 环境的典型症状），换 object_locator 重跑。

### 6. 离线验证

```bash
conda activate object_locator
python verify_calibration_offline.py   # 复核重投影误差与手眼一致性
```

### 7. 更新标定文件

把 `measured_data_in_4/` 选中解的 `rotation_matrix_3x3.txt` / `translation_vector_3x1.txt` 写入 `cam2flange_current.json` 的 `R_cam2flange` / `t_cam2flange`（连同方法名、RMS、日期），并把整份 `calibration_report.json` 归档到 `measured_results/`。执行链路（执行器/预览/验证）无需改代码。

### 8. 上机验证 + offset 归零重测 ⚠️

- 上机验证可用 `scripts/tools/validate_grasp_setup.py --params/--target/--move` 分步检查
- **`config/grasp_config.json` 的 `offset_xyz=[0.072, 0, -0.085]` 是"手眼外参 ~11cm 误差 + hasGripper 帧偏移 3.82cm"两层合计的总账补偿；外参根修后必须归零重测**，否则双重补偿。按 CLAUDE.md「抓取偏差标定」流程重新 jog 标定得出新 offset
- 重标后应只剩外参本身那 ~7.5cm；若新外参正确，jog 标定得出的 offset 应接近 0。若仍有 3~4cm 量级残差，优先怀疑 `hasGripper` 又配回 True（看 Move_Key 启动自检）

---
