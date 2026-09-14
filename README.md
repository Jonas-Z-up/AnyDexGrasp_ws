# AnyDexGrasp Workspace

基于 [AnyDexGrasp](https://graspnet.net/anydexgrasp/) 论文的灵巧抓取复现工作空间。

**当前方案**: 眼在手上 (Eye-in-Hand) — D435i 相机安装在 Z1 末端

**硬件配置**: Unitree Z1 机械臂 + RealSense D435i 深度相机 + 自研五指欠驱灵巧手（兼容 Inspire 手协议）

## 目录结构

```
AnyDexGrasp_ws/
├── anydex_core/           ← AnyDexGrasp 核心库（不改动）
│   ├── models/            ← 神经网络模型（MinkowskiGraspNet）
│   ├── pointnet2/         ← PointNet++ CUDA 扩展
│   ├── knn/               ← KNN CUDA 加速模块
│   ├── utils/             ← 碰撞检测、点云工具
│   ├── dataset/           ← 数据加载器
│   └── ur_toolbox/        ← UR/Inspire 手工具箱
├── scripts/               ← 所有运行脚本
├── hand_control/          ← 自研五指手串口控制 + GUI
├── calibration/           ← 手眼标定参数
├── config/                ← 配置文件
├── logs/                  ← 模型权重 & 推理输出
├── docs/                  ← 文档
└── archive/               ← 归档的旧代码
```

## 快速开始

### 1. 启动 Z1 机械臂控制器（终端0）
```bash
ping 192.168.123.110

cd /home/jonas/Projects/unitree_z1/Unitree-Z1_SDK/z1_controller/build
./z1_ctrl
```
预期: 终端显示 Z1 启动信息，机械臂上电保持当前位置


### 2. 启动 RealSense 相机（终端1）
```bash
conda activate anydex
cd ~/Projects/AnyDexGrasp_ws/scripts
python realsense.py
```
预期：D435i 开始采集，深度+彩色帧持续写入共享内存 `realsense_depth` 和 `realsense_color`。


### 3. 启动抓取编排器 → 演示循环（终端A）
```bash
conda activate anydex
cd ~/Projects/AnyDexGrasp_ws/scripts
python anydex_eih_grasp.py

python anydex_eih_grasp_calib.py
# 演示循环: 每轮拍照后回车自动执行抓取, 松手后自动回观测位姿循环; 仅 Ctrl+C 回零下电退出
# --no-hand: 无手桥, 手部步骤用固定等待(HAND_WAIT_S)模拟
# 偏移微调版: python anydex_eih_grasp_calib.py --calib-jog  ← 见下方"抓取偏差标定"
```
预期：Z1 移动到观测位姿，终端输出"确认抓取位姿无误..."并等待。不要关这个终端——它保持着 Z1 连接，关掉机械臂会掉力。所有可调参数（下降步长/等待时长/抓取高度/手安装参数等）在 `scripts/anydex_eih_grasp.py` 文件顶部「参数设置」区。


### 4. 拍照推理（终端B）
```bash
conda activate anydex
cd ~/Projects/AnyDexGrasp_ws/scripts
python run_inspire_preview.py --visualize --collision \
    --min_depth 0.2 --max_depth 0.6
python run_inspire_preview.py --visualize --collision \
    --min_depth 0.2 --max_depth 0.6 --fixed-type 5
```
预期：弹出 Open3D 窗口显示抓取姿态（红色 Inspire 手 mesh + 蓝色两指夹爪 + 场景点云）。用鼠标检查姿态合理性。


### 5. 确认并执行抓取（每轮）
回到**终端A**，检查 Open3D 中的抓取姿态无误后，按**回车**。
预期：Z1 自动执行抓取流程：接近→**空中张手+预塑形**→可打断下降（Space/Esc 暂停接管人工微调、Ctrl+C 回零）→抓取点逐指闭合→上提→丢弃位姿→松手→**`DROP_SETTLE_S`（默认 2s）后自动回观测位姿，进入下一轮**。每轮都需重新在终端B拍照推理、回到终端A按回车确认新位姿。手部等待/下降步长等时长都是**文件顶部常量**（`HAND_WAIT_S`/`DROP_SETTLE_S`/`DESCEND_STEP_MM`），非命令行参数。仅 Ctrl+C 回零下电退出。
### 眼在手上 vs 眼在手外


| | 眼在手上 (当前) | 眼在手外 (备用) |
|---|---|---|
| 相机位置 | Z1 末端 | 桌面固定 |
| 标定文件 | `calibration/cam2flange_current.json` | `calibration/cam2base_current.json` |
| 变换链 | cam→flange→base (两步) | cam→base (一步) |
| 脚本 | `anydex_eih_grasp.py` | `legacy/z1_from_anydex_preview.py`（已归档） |
| Z1 需查询 | 是（命令观测位姿；原 getpose 的 ~38mm 偏差已定位为 `hasGripper` 配错，2026-09-04 修复） | 否 |


## 自研手控制
```bash
cd ~/Projects/AnyDexGrasp_ws/hand_control
python gui_ub.py
```

## 抓取朝向/安装验证
```bash
cd ~/Projects/AnyDexGrasp_ws/scripts
python tools/validate_grasp_setup.py --move    # 移臂到检查位姿，目视核对指尖/掌心方向（Ctrl+C 先回零再断开）
python tools/validate_grasp_setup.py --target  # 打印最新 grasp 的目标位姿（法兰/掌心/指尖位置）
```

## 抓取偏差标定（人工微调 + 记录）

> 演示版 `anydex_eih_grasp.py` 不做标定记录；标定仅走调参版 `anydex_eih_grasp_calib.py`。

真机发现不同抓取位姿存在程度不同的**位置/朝向偏差**。用人工示教点标定定位系统性误差来源：
到达目标抓取点后键盘微调机械臂对准物体，空格记录调整前后的位姿与偏移；
多组数据后用分析脚本输出 `offset_xyz` / `hand_mount_z` / `offset_rpy_deg` 建议，并判断是否需要重标定手眼。

```bash
# 1. 键盘自测（不动机械臂、不写文件）
cd ~/Projects/AnyDexGrasp_ws/scripts
python jog_manual.py --dry-run

# 2. 真机微调 + 记录（调参版）
python anydex_eih_grasp_calib.py --calib-jog

# 3. 采集 ≥5 组（物体放不同位置/触发不同抓取类型）后分析
python analyze_jog_history.py
```

- 按键：`w/s/a/d/q/e` = 基座系前后/左右/上下微调（初始步长 5mm）；`r/f` = 俯仰 ±（绕法兰中心，默认 1°，`--jog-pitch-step-deg` 可调）；`+`/`-` = 切平移步长（2/5/10 mm）；`空格` = 记录并继续抓取；`回车` = 有微调会补记、没微调跳过；`Esc`/`Ctrl+C` = 中止（有调整会补记）。
- 记录保存：`logs/jog_calib_history.jsonl`（可用 `--jog-history <路径>` 覆盖），每行一条 JSON 样本。
- 分析输出：恒定偏置 → `offset_xyz` 建议；接近轴投影 → `hand_mount_z` 建议；朝向残余 → `offset_rpy_deg` 建议（pitch 为主，绕掌心中心，与 r/f 绕法兰中心不同，小角度近似）；offset 对物体深度线性拟合 → 斜率显著即指向手眼标定旋转误差（需重标定）。
- ✅ 已知系统偏移（2026-09-06 更新）：①`getpose`(FK) 与命令位姿差 **38.2mm 沿法兰 +X**——**不是 FK 虚报**，根因是 `ArmInterface(hasGripper=True)`：末端装的是相机+自研手、没有 Unitree 夹爪，该参数让 SDK 把 FK 末端帧前移 38.2mm 当"夹爪 TCP"。`Move_Key.py`/`MoveZ1.py` 已改 `hasGripper=False`（读回帧 == 命令帧），`Move_Key.py` 启动新增 `check_frame_consistency()` 自检拦截复发。②~~外参错误~~ **已于 2026-09-06 根修**：ChArUco 重标 + 自标定内参（详见 `calibration/README.md` 标定历史），新外参 t_cam2flange=[-9.4, 31.8, 72.4]mm。③遗留：`offset_xyz`（当前 [0.085, 0.01, -0.065]）仍是旧外参时代的补偿值，须归零后用 jog 标定重测新外参基线。定位工具：`python tools/verify_z1_pose.py`（结果在 `logs/pose_verify_history.jsonl`）。

## 训练
```bash
# 阶段1：训练 CGR 表示模型（上游代码，已归档至 scripts/legacy/）
python scripts/legacy/train.py --log_dir logs/model/rep_log --batch_size 4 --learning_rate 0.001 --camera realsense

# 阶段2：训练抓取决策模型
python scripts/legacy/train_multifinger.py \
    --dataset_root=logs/data/decision_model/inspire/obj140 \
    --log_dir=logs/model/inspire_model/obj140/1_class_1_model/ \
    --gripper_type=Inspire --train_multifinger_type=8
```
