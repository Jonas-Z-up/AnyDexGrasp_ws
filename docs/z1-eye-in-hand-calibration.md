# Unitree Z1 眼在手上 (Eye-in-Hand) 手眼标定

RealSense D435i 安装在 Z1 末端法兰，标定板固定于桌面。采集若干组「图像 + 法兰位姿」数据对，用 OpenCV `calibrateHandEye` 求解相机→法兰的固定变换 `T_cam2flange`，并通过三级验证确认真机精度。

> 参考精度（Z1 + D435i + calib.io ChArUco 9×9）：自洽位置 RMS 2.46 mm / 旋转 RMS 0.175°，独立两组解算互差 3.13 mm，离线复测（未参与标定的位姿）3.11 mm。

---

## 1. 系统组成

| 文件 | 作用 |
|------|------|
| `Move_Key.py` | Z1 键盘点动控制器（直连 SDK），内置 Unix-socket 位姿服务 `/tmp/z1_pose_server.sock` |
| `z1_pose_client.py` | 位姿 IPC 客户端模块，须与 `Move_Key.py` 同目录 |
| `tools/data_collect_manual.py` | 图像 + 机械臂位姿采集（位姿手动粘贴，不依赖 SDK/ROS） |
| `tools/hand_in_eye_ChArUco.py` | 主求解器：ChArUco 检测 + PnP + 五种算法求解 + 自动选优 |
| `tools/verify_offline.py` | 离线复测：用未参与标定的新位姿测泛化精度 |
| `tools/verify_on_robot.py` | 真机在线验证：多位姿聚合推算 + 到达测试 |
| `tools/test_poses.txt` | 激励位姿序列（可直接粘贴执行） |
| `cam2flange_current.json` | 现行手眼外参（标定产物，供下游管线消费） |

**环境要求**

- Python 3.8+，依赖：`opencv-python 4.10`（**不要用 5.x**：缺 `cv2.calibrateHandEye`，且 `solvePnP` 会发散到错误解）、`pyrealsense2`、`numpy`
- Unitree Z1 SDK（`z1_controller` 底层控制器 + `unitree_arm_interface` Python 模块）
- 无 ROS 依赖

---

## 2. 相较常见开源方案的差异

GitHub 上现成的手眼标定方案（`easy_handeye`、`moveit_calibration` 及各类单文件脚本）大多面向 ROS + 通用机械臂生态。本项目的差异不在"能标"，而在**数据可信度与排错能力**：

| 能力 | 常见开源方案 | 本项目 |
|------|-------------|--------|
| 生态依赖 | 大多绑定 ROS/TF（easy_handeye、moveit_calibration） | 无 ROS，裸 Z1 SDK；**采集脚本连 SDK 都不需要** |
| 机械臂接口 | 绑定特定机器人 API（UR/Franka/...） | 采集时位姿手动粘贴，与机器人 API 完全解耦——任何能输出末端位姿的机械臂都能复用采集/求解/验证链路 |
| 求解算法 | 通常调用单一算法 | TSAI/PARK/HORAUD/ANDREFF/DANIILIDIS 并行求解 + 自洽评分自动选优；**五法结果是否一致本身就是数据质量报警** |
| 标定板 | 棋盘格/ArUco 居多，需完整入镜 | ChArUco：遮挡、部分入镜可用，单帧 ≥4 角点即可求解，亚像素角点精度；板参数全部命令行可配 |
| 质量评估 | 重投影误差 / 标定自洽 RMS | 三级验证：未参与标定位姿的**泛化复测** → 真机**多位姿聚合** → **卷尺绝对位置抽查 + 到达测试** |
| Z1 帧陷阱 | 无感知（换机即踩） | `hasGripper` FK 帧偏移 38.2 mm 由启动自检拦截——此类系统性偏差在各数据对间互相抵消，**不会体现在标定 RMS 中** |
| 坏数据定位 | 人工翻数据 | 每组检测可视化自动存盘；离群组（>10 mm）自动标记；`--exclude-groups` 剔除重算 |
| 结果交付 | 打印到终端/截图 | JSON schema 固化，下游抓取管线直接读取，无需改代码 |

几条设计取舍的理由：

- **"位姿手动粘贴"是刻意的**。把采集与机器人接口解耦后，可迁移性问题被压缩为"你的机械臂能不能给出末端位姿 6 个数"；Z1 侧的全部耦合集中在 `Move_Key.py` 一个文件里。
- **五法对比的价值不止于选优**。五种经典算法吃同一批数据，若输出的外参明显分叉，几乎必然存在坏数据对或模型误差——这比任何单一 RMS 阈值都灵敏。我们的内参畸变问题（两组独立解互差 8.25 mm）就是靠这类交叉对比暴露的。
- **三级验证补足自洽性盲区**。标定/复测的 RMS 测不出全体系统性偏差（方格边长尺度误差、38 mm 帧偏移、内参尺度误差各组间同样抵消），只有卷尺绝对抽查和真机到达测试能兜底。

---

## 3. 开箱适配（换成你的机器前必改）

所有可迁移配置集中在这几处，其余无需改动：

| 配置项 | 位置 | 说明 |
|--------|------|------|
| Z1 SDK 库路径 | `Move_Key.py` 顶部 `SDK_LIB_DIR` | 指向你本机的 `z1_sdk/lib` |
| 准备点 / 观测位姿 | `Move_Key.py`（启动自动 MoveJ 的目标）、`test_poses.txt` | 因机械臂安装方式、可达空间而异，按你的场景重给 |
| 相机内参 JSON | 各脚本 `--intrinsics-json`，默认 `config/realsense_intrinsics.json` | 字段：`fx, fy, ppx, ppy, distortion_coefficients`（D435i 出厂值可用但不准，建议自标定，见 §7） |
| 标定板参数 | 求解/复测/验证脚本的命令行参数 `--squares-x/y --square-size --marker-size --aruco-dict` | 默认 calib.io ChArUco 9×9 / 方格 50 mm / marker 37 mm / DICT_4X4_50；换板只改命令行 |
| 相机流分辨率 | 采集/验证脚本内 `1280×720@30` | 与内参对应的分辨率必须一致 |
| 位姿服务 socket | `/tmp/z1_pose_server.sock`（`z1_pose_client.DEFAULT_SOCKET_PATH`） | 冲突时换路径即可 |

**不可改动的约定**

- `Move_Key.py` 必须以 `ArmInterface(hasGripper=False)` 连接（脚本已固化，启动自检「帧一致 ✅」）。若为 `True`，FK 末端帧前移 38.2 mm，外参整体沿接近轴偏 ~38 mm 且标定 RMS 看不出来。
- Z1 位姿格式 `[roll, pitch, yaw, x, y, z]`（rad/m），旋转约定 `R = Rz(yaw)·Ry(pitch)·Rx(roll)`；`poses.txt` 存储格式为 `[x, y, z, rx, ry, rz]`，采集脚本自动重排。
- 标定过程中 `T_base_flange` 取自**命令位姿**（MoveJ/MoveL 的目标），不用 FK 查询读回。

---

## 4. 标定操作步骤

### 0. 标定前必查

1. **尺量标定板方格实际边长**。边长标称 50 mm 不等于实际 50 mm；错多少，平移外参就按比例偏多少，且旋转/RMS 不受影响、事后测不出，只能靠尺量排除。
2. 标定板平贴固定（大板背后垫硬板压平），避免灯光直射板面（反光吞角点）。
3. 确认内参文件存在且与相机分辨率匹配。

### 1. 终端 1：Z1 底层控制器

```bash
cd <你的z1_controller路径>/build && ./z1_ctrl
```

### 2. 终端 2：机械臂点动控制器

```bash
conda activate <你的环境>
cd calibration        # Move_Key.py 与 z1_pose_client.py 须同目录
python Move_Key.py --speed 0.2
```

启动后自动 MoveJ 到准备点。**必须看到 `[自检] 帧一致 ✅` 才能继续**；报「帧不一致」说明命令帧与 FK 读回帧脱节（典型原因是 `hasGripper` 配错），此时采集的位姿全是错的。

键盘点动：`W/S/A/D/Q/E` 末端系平移，`I/K/J/L/U/O` roll/pitch/yaw，`P` 打印位姿；按**空格**进命令模式：`pose r p y x y z`（MoveJ 跳转）、`jogpose ...`（MoveL 微调）、`step 0.01`（步长）、`quit`。

### 3. 终端 3：数据采集

```bash
conda activate <你的环境>
cd calibration/tools
python data_collect_manual.py                    # 默认存到 ../sessions/<日期>_collect/
python data_collect_manual.py --out ../sessions/<自定义会话名>
```

操作：机械臂移到位 → 终端 2 按 `P` 打印位姿 → 终端 3 按 `S` → 粘贴 6 个数 → 自动存图 + `poses.txt`；`Q` 退出。图片编号与 `poses.txt` 行号严格一一对应（`0.jpg` ↔ 第 1 行），中途不要删图或改文件。

### 4. 循环采集 15–25 组

要点：

- **标定板全程固定不动**（板/基座间任何相对移动直接毁数据对）；
- **旋转多样性比平移更重要**：roll/pitch/yaw 多向覆盖，保留几组 ±30°~45° 的极端姿态；平移上让板覆盖视野各区域与不同距离；
- 机械臂**完全停稳**再拍（运动模糊毁角点）；
- 预览中单帧识别角点 < 15 的组不要保存（PnP 易解翻转，坏组混进去难查）；
- 可直接粘贴 `tools/test_poses.txt` 中的序列（按你的可达性微调）。

### 5. 求解

```bash
cd calibration/tools
python hand_in_eye_ChArUco.py --data-dir ../sessions/<会话>
# 输出到 <会话>/solve/：calibration_report.json + all_candidate.txt + 检测可视化图
```

流程：ChArUco 检测（`CharucoDetector`）→ `matchImagePoints` → `solvePnP`（ITERATIVE）求板在相机系位姿 → 与法兰位姿按图号对齐 → `cv2.calibrateHandEye` 分别用 TSAI / PARK / HORAUD / ANDREFF / DANIILIDIS 求解 → 按「板在基座系推算位姿的散布」自动选优（score = 位置RMS + 0.1×旋转RMS(rad)）。

**求解自检（不达标不要往下走）**：

- 日志中各组识别角点数普遍 ≥ 25；
- 位置 RMS < 5 mm、旋转 RMS < 1°（内置阈值 `--max-position-rms` / `--max-rotation-rms`）；
- **五种方法的 RMS 彼此接近**（如都在 2–4 mm）；明显分叉说明有坏数据对，回可视化图定位后重采。

### 6. 验证（三级）

**6a. 离线复测（泛化精度）**：保持板不动，用**全新位姿**再采 8–10 组到独立会话，然后：

```bash
python verify_offline.py --data-dir ../sessions/<复测会话>
# 换解: --result-json <解json>；剔离群: --exclude-groups 5,8（>10mm 自动标离群嫌疑）
```

判读：位置 RMS < 5 mm 且与标定自洽水平相当 → 泛化可信。⚠️ 此法测不出全体系统性偏差（尺量边长、38 mm 帧、内参尺度误差都测不出），所以必须配合 6b/6c。

**6b. 真机在线验证（链路一致性）**：终端 2 的 Move_Key 保持运行，然后：

```bash
python verify_on_robot.py --show-window
```

板固定，机械臂换 3–5 个差异大的位姿各按**空格**，实时推算板特征点基座系坐标；散布 RMS < 5 mm → 手眼链路（FK + 外参 + 内参 + PnP）在真机稳定。

**6c. 绝对位置抽查 + 到达测试（最终验收）**：

- 退出汇总给出板中心基座系坐标。在 Move_Key 命令模式执行 `pose <当前r> <当前p> <当前y> <板中心x> <板中心y> <板中心z+0.05>`（抬高 5 cm 接近），到位后目视法兰对中、尺量偏差——这是下游管线真实吃到的误差；
- 卷尺量板中心相对 Z1 基座原点的实际位置，与推算均值对比（板中心 = 棋盘图案外角各进 `方格数×边长/2`）。**这是唯一能暴露尺度/系统性偏差的检查**。

### 7. 写入结果文件

把 `<会话>/solve/calibration_report.json` 中选中解的 `rotation` / `translation` 连同方法名、RMS、日期写入 `cam2flange_current.json`（格式见 §5）。下游管线读取路径不变，无需改代码。

---

## 5. 结果文件格式

`cam2flange_current.json`：

```json
{
  "R_cam2flange": [[...], [...], [...]],   // 3x3, 相机系→法兰系旋转
  "t_cam2flange": [tx, ty, tz],            // 米, 相机系原点在法兰系下的坐标
  "method": "DANIILIDIS",
  "position_rms_m": 0.00246,
  "rotation_rms_deg": 0.175,
  "setup": "eye-in-hand",
  "camera": "RealSense D435i",
  "robot": "Unitree Z1",
  "date": "2026-09-06",
  "board": "calib.io ChArUco 9x9/50mm/37mm/DICT_4X4_50, 31组激励位姿",
  "intrinsics": "自标定 K+径向畸变 (61图)"
}
```

下游使用（把相机系观测点变换到基座系）：

```python
p_flange = R_cam2flange @ p_cam + t_cam2flange          # 相机 → 法兰
p_base   = R_flange2base @ p_flange + t_flange2base     # 法兰 → 基座 (命令位姿构造)
```

---

## 6. 常见问题

| 现象 | 原因与处置 |
|------|-----------|
| Move_Key 启动报「帧不一致」 | `hasGripper` 应为 `False`；未修复前采集的位姿全部无效 |
| 标定 RMS 很好看，真机法兰却整体偏 ~38 mm | FK 帧 ≠ 命令帧（同上）；或卷尺绝对抽查能暴露 |
| 标定 RMS 达标，真机测量整体偏固定比例 | 方格边长标称值与实物不符 → 尺量后改 `--square-size` 重标 |
| 两组独立标定解相差 5–10 mm | 内参畸变没建模（D435i 出厂值零畸变假设不够）→ 见 §7 |
| `solvePnP` 重投影异常大 / 无 `calibrateHandEye` | cv2 版本问题，换 opencv-python 4.10 |
| 五种方法 RMS 明显分叉 | 有坏数据对：查 `detection_visualizations/`，剔除角点过少/反光/模糊的组重采 |
| 换了内参之后 | **所有手眼外参必须用新内参重算**（内外参强耦合） |
| 下游原有偏移补偿 | 外参更新后旧的 `offset` 类补偿必须归零重测，否则双重补偿 |

---

## 7. 内参建议（可选但强烈推荐）

D435i RGB 出厂内参在标定距离/大角度下误差不可忽略。做法：用采集的标定图另跑一次 `cv2.calibrateCamera`（61 图联合自标定），得到含径向畸变的 K，存成同格式 JSON。参考效果：重投影误差 0.996 px → 0.270 px，两组独立手眼解互差 8.25 mm / 0.418° → 3.13 mm / 0.058°。更新内参后**必须重算外参**。
