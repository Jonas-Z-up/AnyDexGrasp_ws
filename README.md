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

### 1. 启动 Z1 机械臂控制器
```bash
ping 192.168.123.110
cd /home/jonas/Projects/unitree_z1/Unitree-Z1_SDK/z1_controller/build
./z1_ctrl
```
预期: 终端显示 Z1 启动信息，机械臂上电保持当前位置


### 2. 启动 RealSense 相机
```bash
conda activate anydex
cd ~/Projects/AnyDexGrasp_ws/scripts
python realsense.py
```
预期：D435i 开始采集，深度+彩色帧持续写入共享内存 `realsense_depth` 和 `realsense_color`。


### 3. 启动抓取编排器 → 移动到观测位姿（终端A）
```bash
conda activate anydex
cd ~/Projects/AnyDexGrasp_ws/scripts
python anydex_eih_grasp.py
# 2. 真机微调 + 记录
python anydex_eih_grasp.py --calib-jog
```
预期：Z1 移动到观测位姿，终端输出"请在另一个终端运行..."并等待。不要关这个终端——它保持着 Z1 连接，关掉机械臂会掉力。


### 4. 拍照推理（终端B）
```bash
conda activate anydex
cd ~/Projects/AnyDexGrasp_ws/scripts
python run_inspire_preview.py --visualize --collision \
    --min_depth 0.2 --max_depth 0.6 \
    --inspire_mesh_json_path /home/jonas/Projects/AnyDexGrasp/generate_mesh_and_pointcloud/inspire_urdf
```
预期：弹出 Open3D 窗口显示抓取姿态（红色 Inspire 手 mesh + 蓝色两指夹爪 + 场景点云）。用鼠标检查姿态合理性。

### 5. 确认并执行抓取
回到**终端A**，检查 Open3D 中的抓取姿态无误后，按**回车**。
预期：Z1 自动执行抓取流程：接近→**空中张手+预塑形**→可打断下降（Space/Esc 暂停接管、Ctrl+C 回零）→抓取点逐指闭合→[回车确认抓取]→上提→丢弃位姿→[回车确认丢弃]→松手→**3s 后自动回零**。加 `--no-hand` 可回退手动模式。
### 眼在手上 vs 眼在手外


| | 眼在手上 (当前) | 眼在手外 (备用) |
|---|---|---|
| 相机位置 | Z1 末端 | 桌面固定 |
| 标定文件 | `calibration/cam2flange_current.json` | `calibration/cam2base_current.json` |
| 变换链 | cam→flange→base (两步) | cam→base (一步) |
| 脚本 | `anydex_eih_grasp.py` | `z1_from_anydex_preview.py` |
| Z1 需查询 | 是（命令观测位姿；原 getpose 有 ~40mm FK 误差，已切换，评估中） | 否 |


## 自研手控制
```bash
cd ~/Projects/AnyDexGrasp_ws/hand_control
python gui_ub.py
```

## 抓取朝向/安装验证
```bash
cd ~/Projects/AnyDexGrasp_ws/scripts
python validate_grasp_setup.py --move    # 移臂到检查位姿，目视核对指尖/掌心方向（Ctrl+C 先回零再断开）
python validate_grasp_setup.py --target  # 打印最新 grasp 的目标位姿（法兰/掌心/指尖位置）
```

## 抓取偏差标定（人工微调 + 记录）

真机发现不同抓取位姿存在程度不同的**位置偏差**。用人工示教点标定定位系统性误差来源：
到达目标抓取点后键盘微调机械臂对准物体，空格记录调整前后的位姿与偏移；
多组数据后用分析脚本输出 `offset_xyz` / `hand_mount_z` 建议，并判断是否需要重标定手眼。

```bash
# 1. 键盘自测（不动机械臂、不写文件）
cd ~/Projects/AnyDexGrasp_ws/scripts
python jog_manual.py --dry-run

# 2. 真机微调 + 记录（代替抓取编排器命令）
python anydex_eih_grasp.py --calib-jog

# 3. 采集 ≥5 组（物体放不同位置/触发不同抓取类型）后分析
python analyze_jog_history.py
```

- 按键：`w/s/a/d/q/e` = 基座系前后/左右/上下微调（初始步长 5mm）；`r/f` = 俯仰 ±（绕法兰中心，默认 1°，`--jog-pitch-step-deg` 可调）；`+`/`-` = 切平移步长（2/5/10 mm）；`空格` = 记录并继续抓取；`回车` = 有微调会补记、没微调跳过；`Esc`/`Ctrl+C` = 中止（有调整会补记）。
- 记录保存：`logs/jog_calib_history.jsonl`（可用 `--jog-history <路径>` 覆盖），每行一条 JSON 样本。
- 分析输出：恒定偏置 → `offset_xyz` 建议；接近轴投影 → `hand_mount_z` 建议；offset 对物体深度线性拟合 → 斜率显著即指向手眼标定旋转误差（需重标定）。
- ⚠️ 已知系统偏移：`verify_z1_pose.py` 实测 `getpose` 与命令位姿差 39.4±1.0mm（沿接近轴）。cam→base 已改用命令观测位姿（原用 getpose 有 ~40mm FK 误差），切换后目标上抬 ~35mm 致部分抓取 IK 不可达，**切换评估中**。定位工具：`python verify_z1_pose.py`（结果在 `logs/pose_verify_history.jsonl`）。

## 训练
```bash
# 阶段1：训练 CGR 表示模型
python scripts/train.py --log_dir logs/model/rep_log --batch_size 4 --learning_rate 0.001 --camera realsense

# 阶段2：训练抓取决策模型
python scripts/train_multifinger.py \
    --dataset_root=logs/data/decision_model/inspire/obj140 \
    --log_dir=logs/model/inspire_model/obj140/1_class_1_model/ \
    --gripper_type=Inspire --train_multifinger_type=8
```
