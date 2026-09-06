# 标定数据

## 当前方案：眼在手上 (Eye-in-Hand)

相机安装在 Z1 机械臂末端，随机械臂一起运动。

### 目录结构（2026-09-06 整理）

```
calibration/
├── Move_Key.py / z1_pose_client.py   Z1 点动控制器 + 位姿 IPC（须同目录）
├── cam2flange_current.json           ★ 现行手眼外参（抓取管线消费点）
├── cam2base_current.json             旧方案遗留，仍被 scripts/legacy 引用
├── tools/                            现行标定脚本（采集/求解/复测/真机验证/test_poses）
├── board/ChArUco.jpeg                标定板实物参考照
├── sessions/                         历史会话（采集数据 + 各次解算），只增不改
├── measured_results/                 结果归档（旧 HORAUD 存档 + 20260906 报告）
└── archive/                          弃用存档（v1 棋盘格方案、旧 txt 零件）
```

现行外参 = `cam2flange_current.json`（由 `sessions/20260906_crosscheck/solve_more` 的解写入）；
现行内参 = `../config/realsense_intrinsics.json`（61 图自标定 K+畸变，出厂值备份 `realsense_intrinsics_factory_backup.json`）。

### 使用的标定文件

| 文件 | 说明 |
|------|------|
| `cam2flange_current.json` | **当前使用** — Z1 相机到法兰变换矩阵 (DANIILIDIS, 2026-09-06, ChArUco 31组 + 自标定内参, 自洽 RMS 2.46mm/0.175°)。旧 HORAUD 结果存档于 `measured_results/cam2flange_horaud_archive.json` |

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

### 标定历史

| 日期 | 方案 | 结果 |
|------|------|------|
| 2026-09-05 | v2 ChArUco（9x9 / 50mm / 37mm / DICT_4X4_50），21 组 | DANIILIDIS 选优：位置 RMS 2.99mm / 旋转 RMS 0.394°（自洽）；离线复测 10 组（剔 1 组角点不足导致的离群）位置 RMS 3.11mm，板位置复现差 0.7mm。报告见 `sessions/20260905_calib_21/solve/calibration_report.json`（当时仍用零畸变内参） |
| 2026-09-06 | 内参修正 + 外参更新 | 61 张采集图联合自标定检出 RGB 径向畸变（零畸变假设重投影 0.996px → 含畸变 0.270px；两组独立求解差从 8.25mm/0.418° 收窄到 3.13mm/0.058°）。内参更新为自标定 K+畸变（出厂值存档 `config/realsense_intrinsics_factory_backup.json`），外参采用 31 组重解（`sessions/20260906_crosscheck/solve_more`，DANIILIDIS，2.46mm/0.175°），写入 `cam2flange_current.json`，报告归档 `measured_results/calibration_report_20260906_charuco.json` |

---

## 标定程序

当前方案（ChArUco 标定板）：

| 文件 | 作用 |
|------|------|
| `Move_Key.py` | Z1 键盘点动控制器（直连 SDK + Unix-socket 位姿服务 `/tmp/z1_pose_server.sock`） |
| `z1_pose_client.py` | Move_Key 的位姿 IPC 共享模块（须与 Move_Key.py 同目录） |
| `tools/data_collect_manual.py` | D435i 图像 + 机械臂位姿采集（`--out` 指定会话目录，默认 `sessions/<日期>_collect/`） |
| `tools/hand_in_eye_ChArUco.py` | 主求解器 — ChArUco 检测 + PnP，五法对比自动选优，默认输出 `<data-dir>/solve/` |
| `tools/verify_offline.py` | 离线复验：用未参与标定的新位姿测泛化精度（`--exclude-groups` 剔除离群组） |
| `tools/verify_on_robot.py` | 真机在线验证：实时推算板特征点基座系坐标（多位姿聚合 + 到达测试） |
| `tools/test_poses.txt` | 标定激励位姿序列（20 组基准 + 补充位姿，含可达性备注） |
| `archive/v1_hand_in/` | 旧方案（普通棋盘格 + OpenCV 旧 API），已弃用仅存档 |

说明：
- **运行环境：`object_locator`**（cv2 4.10）。求解一步必须用它——`anydex` 的 cv2 是 5.0，既没有 `cv2.calibrateHandEye`，`solvePnP` 还会在部分尺度下发散到错误局部解（实测 12mm 时重投影 138px；cv2 4.10 同数据 0.5px）。Move_Key/数据采集在 anydex 也能跑，但统一用 object_locator 最省心
- `Move_Key.py` 通过绝对路径加载 Z1 SDK（`~/Projects/unitree_z1/.../z1_sdk/lib`），不受本目录位置影响
- `tools/` 内脚本均用相对本脚本的路径：采集数据落在 `sessions/<会话>/`，求解输出跟随数据落在 `sessions/<会话>/solve/`，**采集、求解、复测都必须在 `tools` 目录下运行**
- 内参默认取 `../config/realsense_intrinsics.json`（与抓取管线共用，2026-09-06 起为 61 图联合自标定 K+径向畸变；出厂值备份在 `realsense_intrinsics_factory_backup.json`）。⚠️ 内参与外参配套：若再更换内参，所有手眼外参必须用新内参重算
- 标定板：calib.io ChArUco，9x9 方格 / 方格 50mm / marker 37mm / DICT_4X4_50，marker 印在全部白格共 40 个（ID 0-39）。板参数已写入 `v2` 两个脚本的默认值，已与实物（`board/ChArUco.jpeg`）做过生成比对 + 检测验证（40 marker / 64 角点全部识别）

---

## 重新标定操作流程（ChArUco）

### 0. 标定前必查

```bash
# ① 尺量棋盘格实际边长！tools 脚本默认 0.050 (50mm) 只是标称值。
#    方格边长错多少，PnP 平移就按比例带偏 t_cam2flange（旋转不受影响、RMS 照样好看），
#    离线复验测不出这种全体系统性偏差，只能靠尺量排除。
# ② 板子平贴固定（45cm 大板易翘弯，背后垫硬板压平），避免灯光直射板面（反光会吞角点）。
python tools/hand_in_eye_ChArUco.py --help   # 确认 --squares-x/y / --square-size / --marker-size
```

### 1. 终端1：Z1 底层控制器

```bash
cd ~/Projects/unitree_z1/Unitree-Z1_SDK/z1_controller/build && ./z1_ctrl
```

### 2. 终端2：机械臂点动控制器

```bash
conda activate object_locator
cd /home/jonas/Projects/AnyDexGrasp_ws/calibration
python Move_Key.py --speed 0.2     # 自动 MoveJ 到准备点 [0.075, 1.057, 0.073, 0.327, -0.08, 0.35]
```

⚠️ 启动后必须看到 **`[自检] 帧一致 ✅`** 才能继续。若报"帧不一致"，说明读回帧 ≠
命令帧（典型是 `hasGripper` 配错，会让 FK 末端帧前移 38.2mm 当夹爪 TCP），此时
`P` 打印的位姿是错的，**存进 poses.txt 会让外参沿接近轴整体偏 ~38mm**，且标定报告的
一致性 RMS 照样好看（系统性偏差在各组间抵消），事后极难发现。

键盘点动：`W/S` ±X（末端系）、`A/D` ±Y、`Q/E` ±Z、`I/K` Roll、`J/L` Pitch、`U/O` Yaw、`P` 打印位姿、`F` 回准备点、`H` 回零、`[` `]`/`-` `=` 步长缩放、`X` 退出。
按**空格**进入命令模式：`pose r p y x y z`（MoveJ 跳转，推荐）、`jogpose ...`（MoveL 微调）、`step 0.01`、`speed 0.2`、`quit`。退出自动 backToStart 回零。

### 3. 终端3：数据采集（必须在 tools 目录下运行）

```bash
conda activate object_locator
cd calibration/tools
python data_collect_manual.py    # 默认存到 ../sessions/<日期>_collect/；--out 可指定目录
```

操作：按 S → 粘贴终端2 的 6 个数 (r p y x y z) → 存图+poses.txt；Q 退出。
注意：图片编号与 poses.txt 行号严格一一对应（`0.jpg` ↔ 第 1 行），中途不要手动删图或改 poses.txt（输入出错不保存不会破坏对齐）。

### 4. 循环采集 15-25 组

推荐混合操作：命令模式 `pose` 跳到目标附近 → 键盘微调对位 → `P` 打印 → 终端3 按 `S` 粘贴保存。
要点：
- **标定板全程固定不动**（板/基座间任何相对移动直接毁数据对），板面压平
- 激励位姿序列见 `tools/test_poses.txt`；**旋转多样性（roll/pitch/yaw 多向）比平移更重要**，"极端姿态激励"几条别省略
- 每组保存前看预览：板子大部分在画面内、无严重反光、机械臂完全停稳（运动模糊毁角点）
- 单帧识别角点 <15 的组不要保存（求解门槛虽是 4 角点，但少角点 PnP 易解翻转，混进坏组事后难查）

### 5. 求解（必须 object_locator，必须在 tools 目录下）

```bash
cd calibration/tools
python hand_in_eye_ChArUco.py --data-dir ../sessions/<今日会话>
# 解算结果输出到 <data-dir>/solve/（calibration_report.json + all_candidate.txt + 检测可视化图）
# 可选: --output-dir <目录> --intrinsics-json <路径> --squares-x/y --square-size --marker-size --no-save-vis
```

自检：
- 日志中各组识别角点数应普遍 ≥25
- 位置 RMS < 5mm（阈值 `--max-position-rms`）、旋转 RMS < 1°（`--max-rotation-rms`）
- 五种方法的 RMS 应彼此接近（如位置 RMS 都在 3-4mm）；明显分叉说明有坏数据对

### 6. 验证（离线复测 + 真机验证）

**6a. 离线复测（泛化精度）**：保持标定板不动，用**全新位姿**再采 8-10 组到独立会话目录
（`python data_collect_manual.py --out ../sessions/<日期>_verify`），然后：

```bash
cd calibration/tools
python verify_offline.py --data-dir ../sessions/<复测会话>
# 默认读取当前采用的解 (sessions/20260906_crosscheck/solve_more)；换解用 --result-json 指定
# 有离群组时: --exclude-groups 5,8  剔除后重算（脚本会把偏差 >10mm 的组标为离群嫌疑）
```

判读：位置 RMS < 5mm 且与标定自洽水平相当 → 泛化可信；对比剔除离群组前后的板位置均值还能确认板子中途没被碰动。

**6b. 真机在线验证（最终验收）**：终端2 的 Move_Key 保持运行（提供位姿服务），然后：

```bash
cd calibration/tools
python verify_on_robot.py --show-window
```

- **多位姿聚合**：板固定，机械臂换 3-5 个差异大的位姿各按空格，推算的板特征点基座系坐标散布 RMS < 5mm → 手眼链路真机上稳定
- **到达测试**：用退出时汇总的板中心坐标，在 Move_Key 命令模式执行 `pose <当前r> <当前p> <当前y> <板中心x> <板中心y> <板中心z+0.05>`（抬高 5cm 接近），到位后目视法兰对中、尺量偏差——这是抓取管线真实吃到的误差
- **绝对位置抽查**：卷尺量板特征点相对 Z1 基座原点的实际位置与推算均值对比（推算的"板位置"指棋盘图案外角原点；板中心 = 图案外角各进 22.5cm）。这是唯一能暴露尺度/系统性偏差的检查

### 7. 更新标定文件

把 `<data-dir>/solve/calibration_report.json` 中选中解的 `rotation` / `translation` 写入 `cam2flange_current.json` 的 `R_cam2flange` / `t_cam2flange`（连同方法名、RMS、日期），并把整份 `calibration_report.json` 归档到 `measured_results/`。执行链路（执行器/预览/验证）无需改代码。

### 8. 上机验证 + offset 归零重测 ⚠️

- 上机验证可用 `scripts/tools/validate_grasp_setup.py --params/--target/--move` 分步检查
- **`config/grasp_config.json` 的 `offset_xyz`（当前 [0.085, 0.01, -0.065]，2026-08-10 设定）是对旧外参 ~11cm 误差的总账补偿；外参已于 2026-09-06 根修，必须归零重测**，否则双重补偿。按 CLAUDE.md「抓取偏差标定」流程重新 jog 标定得出新 offset

---
