# AnyDexGrasp 眼在手上抓取复现

基于 AnyDexGrasp 论文的灵巧抓取复现，**当前为眼在手上方案**（D435i 安装在 Z1 末端）。

## 怎么跑

```bash
# 终端0: Z1 底层控制器
./z1_ctrl  (在 unitree_z1 SDK build 目录)

# 终端1: 相机
conda activate anydex && cd scripts && python realsense.py

# 终端A: Z1 编排器（常驻，演示循环）
conda activate anydex && cd scripts && python anydex_eih_grasp.py   # 标定用 anydex_eih_grasp_calib.py（微调默认开启，--calib-jog 仅兼容写法）

# 终端B: 视觉推理
conda activate anydex && cd scripts && python run_inspire_preview.py --visualize --collision --preprocess --min_depth 0.2 --max_depth 0.6
```

## 技术栈
- Python 3.8 (conda `anydex` 用于推理; `/usr/bin/python3.8` 用于 Z1 SDK)
- PyTorch + MinkowskiEngine (CGR 模型推理)
- Open3D (碰撞检测 + 可视化)
- Unitree Z1 SDK (`unitree_arm_interface`) + ROS (机械臂控制)
- RealSense D435i + 共享内存 (`realsense.py`)

## 目录
- `scripts/` — 入口脚本。`anydex_eih_grasp.py` 是主编排器（演示循环），`anydex_eih_grasp_calib.py` 是标定版（标定微调默认开启；`--calib-jog` 为兼容 no-op，关闭用 `--no-calib-jog`），`MoveZ1.py` 是 Z1 驱动子进程
- `anydex_core/` — 不改动的算法库 (models, pointnet2, knn, utils, dataset, ur_toolbox)
- `calibration/` — `cam2flange_current.json` (当前眼在手上标定) + 标定程序（`Move_Key.py` 点动控制器、`tools/` 采集/求解/离线复测/真机验证，`sessions/` 历史会话数据，`archive/` v1 棋盘格旧方案存档；重标流程见 `calibration/README.md`）
- `assets/` — 静态大资源，均 git 忽略（仅 `own_urdf/.gitkeep` 入库；`inspire_urdf/` 无任何文件入库）：`inspire_urdf/`（Inspire 手 mesh/点云/宽度查表，来自上游仓库）、`own_urdf/`（自研手 SolidWorks 导出 URDF/STL，需线下同步）
- `hand_control/` — 自研手串口控制
- `logs/` — 模型权重 + 推理输出

## 当前状态
- ✅ 阶段一 CGR 模型直接复用（权重: logs/model/checkpoint.tar.18）
- ✅ 阶段二 Inspire 手决策模型复用（权重: logs/model/inspire_model/obj140）
- ✅ 眼在手上推理管线打通（`anydex_eih_grasp.py` + `run_inspire_preview.py`）
- ✅ workspace 滤波统一为 ±0.40（现役管线两处：`get_grasp` 默认参数 ±0.40 + 预览 `filter_grasp_group_by_z_axis` 经 CLI 参数化；`robot_inspire.py` 旧演示路径另有一处硬编码 0.4、不在现役 EIH 管线；点云 crop 层默认 ±0.5；注意 `filter_grasp_group_by_z_axis` 首参 0.4 实为 score 阈值，workspace 限幅靠 `ws_x/y_range` 关键字参数）
- ✅ 点云预处理模块 `scripts/pointcloud_preprocess.py`（crop + RANSAC 桌面分割 + SOR 去噪，`--preprocess` 启用）
- ✅ `realsense.py` 默认开启实时相机预览窗口
- ✅ 演示循环抓取（默认行为，无 `--auto` 开关）：`anydex_eih_grasp.py` 集成 `hand_bridge.py`——接近后**空中先张手+预塑形**（`preshape()` 广播含拇指内旋，避免低空旋转撞桌）→**步进下降 12cm（可打断接管）**→抓取点逐指闭合（`close()`）→上提→丢弃位姿→松手→**`DROP_SETTLE_S`（demo 默认 1s）后自动回观测位姿循环下一轮**；仅 Ctrl+C/硬错误才回零下电退出
- ✅ 每轮流程：仍按一次回车确认新抓取位姿，之后 Phase 2（接近→下降→闭合→上提→丢弃→松手）**全自动执行**，手部步骤用固定等待（`HAND_WAIT_S` 默认 3s，文件顶部常量）替代中间回车闸门；某轮失败（文件缺失/不可达）自动续轮不断电。下降中 **Space/Esc 暂停接管**人工微调（`_manual_takeover`，命令位姿基准、无记录）
- ✅ 两处人工确认闸门（标定版 `anydex_eih_grasp_calib.py` 保留）：闭合后、到达丢弃位姿后各需按一次回车才继续（防碰撞/防误丢）
- ✅ 预塑形动作库 `PRESHAPE_LIBRARY`（数据驱动 8 种抓取类型，与 `controller_ub.py` 动作同名对应、参数各自独立）
- ✅ 预塑形调试工具 `scripts/tools/test_hand_actions.py`（16 键直达 8 种类型：1-8 仅预塑形，Q-I 预塑形+闭合完整动作，O 张开）
- ✅ 抓取朝向（已打通）：执行器"读网络 `rotation_matrix` + 安装偏移补偿（`hand_mount_z`）+ 角度偏差保护"已工作——`mount_yaw_deg=-90°` 修正后网络朝向偏差可低至 9.1°（真机实测），会采用网络朝向。偏差 = 命令法兰朝向与固定朝向（兜底）的夹角，超过 config `orient_deviation_thresh_deg`（当前 90°）回退固定朝向；预览绿框=采用网络朝向、橙框=回退固定朝向
- ✅ 真机验证工具 `scripts/tools/validate_grasp_setup.py`：`--params` 打印参数/方向，`--target` 计算目标位姿，`--move` 移到检查位姿；**第 1 步静态朝向真机验证已通过**（指尖前下/掌心朝下/Ctrl+C 先回零）
- ✅ 根因验证工具 `scripts/tools/verify_z1_pose.py`：自动化对比 getpose vs 命令位姿（到位稳定/姿态扫描/物理尺量三测），结果写入 `logs/pose_verify_history.jsonl`；实测 getpose 恒定偏 ~38.2mm——根因是 `hasGripper` 配错，**已修**（见下方"已知系统偏移"层1）
- ⚠️ 碰撞检测默认开启，偶尔滤掉全部候选（预处理后应改善）
- ⚠️ `--preprocess` 会把桌面从点云和碰撞检测中一并滤除，**无指尖/掌心防穿桌检查**（只剩 MoveZ1 法兰 z≥0.12 兜底，指尖仍可能穿桌）；下降过程可按 **Space/Esc** 暂停接管人工微调避免碰撞
- ✅ 网络朝向真机执行验证（进行中）：偏差 ≤ config `orient_deviation_thresh_deg`（当前 90°）即采用网络朝向；真机已多次执行网络朝向抓取。指尖落点修正：`hand_mount_z` 0.12→0.09（整手沿接近轴推深）+ `offset_xyz` 基座系水平偏移（当前 [0.085, 0.01, -0.065]，2026-08-10 设定，见"已知系统偏移"层2 遗留③）
- ✅ 抓取偏差标定工具：`anydex_eih_grasp_calib.py --calib-jog` 到达抓取点后进入键盘微调（w/s/a/d/q/e 基座系 XYZ + **r/f 俯仰**绕法兰中心，`+`/`-` 切平移步长 2/5/10mm 默认 5mm，`--jog-pitch-step-deg` 默认 1°），空格记录调整前后**命令位姿**与偏移（offset=操作员调整量）到 `logs/jog_calib_history.jsonl`（`--jog-history` 可改路径），回车在有微调时也会补记（防顺手按回车导致数据静默丢弃），`analyze_jog_history.py` 汇总恒定偏置/接近轴/深度相关/朝向残余四类误差并给出 `offset_xyz`/`hand_mount_z`/`offset_rpy_deg`(pitch) 建议
- ✅ 已知系统偏移（2026-09-04 更新）：分两层，**层1 已根修**。
  - **层1 = `hasGripper` 配错导致 38.2mm 帧偏移（已修）**：**不是 FK 虚报**。末端装的是 D435i + 自研手、**没有 Unitree 夹爪**，但 `Move_Key.py`/`MoveZ1.py` 用 `ArmInterface(hasGripper=True)` 连接，SDK 会把 FK 末端帧（index=6）从法兰安装面前移 **38.2mm** 当作"夹爪 TCP"（SDK 实测 q=0：False→`[0.049,0,0.1605]`，True→`[0.0872,0,0.1605]`）。真机 `verify_z1_pose.py` 的 scan 数据把误差投影到法兰 +X：pitch=0/0.4/0.8 → **38.20/38.24/38.47 mm**，完全吻合（原记"39.4±1.0mm 沿接近轴"即此）。→ 两处已改 `hasGripper=False`，读回帧 == 命令帧；`Move_Key.py` 启动新增 `check_frame_consistency()` 自检（3mm/1°，配错打印错误警告块但不中止初始化）。**`getpose`/`P` 键位姿现在可直接用于标定**。代价：因标定时存的是 TCP 帧位姿，旧外参里混入了这 3.82cm，必须重标。
  - **层2 = 手眼外参错误（✅ 2026-09-06 已根修，见下条）**：当年 `t_cam2flange` 把相机位置算错 **~11cm 沿法兰+X / ~3cm 高**。扣除层1 的 3.82cm 后**仍剩 ~7.5cm 属外参本身**，靠 ChArUco 重标消除。点云与网络已验证正确（网络两指抓点深度 0.383 ≈ 物理尺量 0.38m，口头实测无备忘）；6 次 jog 反推相机实际位置≈基座 [0.414, 0.03, 0.356]（⚠️ 无落盘记录：jog_calib_history.jsonl 已于 2026-09-01 删除，git 历史亦无此值，待复测确认）
  - ✅ **外参根修完成（2026-09-06）**：ChArUco（calib.io 9x9/50mm/37mm/DICT_4X4_50）重标，新解 t_cam2flange=[-9.4, 31.8, 72.4]mm，写入 `cam2flange_current.json`。证据链：两组独立数据（21+31 组，板位不同）交叉求解差 3.13mm/0.058°；检出 RGB 径向畸变并更新内参（零畸变重投影 0.996px → 含畸变 0.270px，畸变曾是 8mm 光轴歧义主因；0.996px 仅文档文字记录，0.270px 有 realsense_intrinsics.json 0.26951 硬证据）。旧 HORAUD 结果存档 `calibration/measured_results/cam2flange_horaud_archive.json`。**遗留**：①新板 50/37mm 边长尚未尺量确认（默认值=标称值）；②绝对精度尚未尺量仲裁（复测算得新板位置 [0.819, 0.236, -0.011]m 可对照——⚠️ verify_on_robot.py 只打印不落盘，此值无记录，需重跑生成）；③`offset_xyz`（当前 [0.085, 0.01, -0.065]，2026-08-10 设定）仍是**旧外参时代的补偿值**，新外参基线必须归零后用 `anydex_eih_grasp_calib.py --calib-jog` 重测（旧值 [0.072,0,-0.085] 曾存于 config git 历史 5301798 与 measured_results/README.md，系更早的过期值）
- ✅ 自研手几何锚定 hand_model=own（2026-09-08，代码集中在 `own_hand/`，进度见 `own_hand/docs/自研手几何替换Inspire-进度与计划.md`）：`own_hand/own_hand_grasp.py` 用自研手 URDF FK 替换 Inspire 查表锚定——接触三角形法向=接近轴、拇指方向=开合轴、包络中心对准抓取点并推进 depth；**grasp_type 按 hand_bridge 语义解释**（2=Ring，用 `--fixed-type 2`）。经三轮渲染判定迭代修复：①"指尖插桌"=接近轴误绑手指轴→改接触三角形法向（上游 angles2stl 公式）；②方向反转→符号判据改 **C·n>0**（旧"掌体负侧"规则对环轴手势失效）；③高度偏浅→锚定点推进 `.depth`。预览新增 `--hand-model own|inspire`（缺省读 config `hand_model`，现为 own）与 `--fixed-type 1-8`；own 模式 JSON 顶层 translation/rotation_matrix 语义=**自研手 base 帧位姿**。config own 组：`hand_mount_z_own=0.05`（物理安装距离，类型2/3/4 独立收敛 49-52mm 印证）、`mount_yaw_deg_own=0`（base 帧≈法兰帧）、朝向闸门关闭（180°，滚转分量与固定朝向差异属正常，IK 级固定朝向兜底）。工具：`own_hand/tools/`（闭合回归/约定校验/接触组搜索/hz推导/渲染对比）。选型/碰撞/深度仍走 Inspire 管线（v2 待换）。⬜ 真机 jog 重标定（own 模式）+ A/B 对比待做
- ⬜ 真机细调闭环（own 模式）：offset_xyz/offset_rpy_deg/z_comp_knots 已归零（2026-09-07），own 参数初值就位；待 `--calib-jog`（own 模式，建议 `--fixed-type 3`、历史写 `jog_calib_history_own.jsonl`）收敛 + A/B（inspire vs own 各 10 次）验证
- ⬜ 自研手阶段二：需采集试错数据 + 训练专属决策模型（当前沿用 Inspire 选型模型）
- ⬜ 自研手 URDF/Mesh 用于碰撞检测（当前碰撞过滤仍用 Inspire 点云，仅锚定几何已换）

## 关键约定
- `MoveZ1.py` 作为子进程运行，走 `--server` 模式，通过 stdin/stdout 文本协议通信；MoveL 失败自动回退 MoveJ
- `start_new_session=True` 防止 Ctrl+C 传播到 Z1 子进程；下电前任何退出路径（正常完成/Ctrl+C/异常）都先 `home` 回零再断开——**顺序固定为：先机械臂回零、再松手复位灵巧手**（低空直接张开手指会扫到桌面损坏手，故 Ctrl+C 不再提前松手）——编排器不传 `--no-return-home`，MoveZ1 shutdown 默认 `backToStart` 兜底，断电停在零位不半空下垂。demo 版正常完成：松手后等 `DROP_SETTLE_S`（demo 默认 1s，文件顶部常量）回观测位姿循环，仅 Ctrl+C 回零下电；标定版仍等 `DROP_SETTLE_S`（默认 3s）后自动回零下电（`--drop-settle-s` 可调），无二次回车
- 移动策略: Phase1 MoveJ→准备点→MoveJ→观测位姿; Phase2 MoveJ→接近（IK 不可达时回退固定朝向、并降接近高度 12→8→6cm 重试；全失败打印"超出 Z1 可达范围"并优雅回零中止）→**空中张手+预塑形**→**步进下降**（`jog_manual.descend_interruptible`，demo 用 `DESCEND_STEP_MM` 常量默认 10mm，标定版用 `--jog-descend-step-mm`；Space/Esc 暂停接管、Ctrl+C 中断回零）→抓取点逐指闭合→上提→丢弃位姿→松手→`DROP_SETTLE_S` 后回观测位姿循环下一轮。demo 中间闸门用固定等待 `HAND_WAIT_S`；标定版闭合后/到丢弃位姿后仍各需回车确认
- 抓取朝向: 手与相机同轴沿法兰 **+X** 安装（指尖前下方，真机已验证）。规范手帧：palm +X=指尖、掌心朝 -Y（URDF 指节朝 -Y 弯曲）。物理安装 = Rx(+90°)（掌心朝法兰 -Z，默认状态掌心向下）；`mount_yaw_deg=-90°` 是**补偿符号**，使命令法兰放出的真实手 = 网络渲染（若设 +90° 会掌心翻转朝上，偏差~171° 全回退固定）。指尖方向=法兰+X首列，法兰位置=掌心−`hand_mount_z`×指尖方向（当前 0.09，减小=整手沿接近轴推深）。参数在 `config/grasp_config.json`（hand_mount_z/hand_fingertip_z/mount_yaw_deg/orient_deviation_thresh_deg/offset_xyz/offset_rpy_deg）
- 基座系抓取偏移 `offset_xyz`（config，当前 [0.085, 0.01, -0.065]，待新外参基线归零重测）：整只手纯水平平移（不改变接近轴方向与 z），`grasp_common.load_offset_xyz`/`compute_grasp_target` 统一处理，执行器/预览/验证三处一致。改 x/y 分量即可水平微调
- 基座系旋转偏移 `offset_rpy_deg`（config，[roll,pitch,yaw]°，**当前 [0, 15, 0]：pitch=15° 处于激活状态，每轮真机抓取的执行朝向都被改写；实验值还是残留待人工确认**）：offset_xyz 的旋转孪生——绕**掌心**中心、基座系旋转，作用于最终执行的朝向（网络或固定回退都生效）；在偏差闸门判定**之后**应用，故 `deviation_deg`/预览绿橙框不受影响。零偏移=旧行为。`grasp_common.load_offset_rpy_deg`/`compute_grasp_target` 统一处理，执行器/预览/验证三处一致（预览把基座系偏移共轭进相机系）。注意与 `anydex_eih_grasp_calib.py --calib-jog` 的 r/f 俯仰不同：jog 绕**法兰**中心、offset_rpy 绕**掌心**中心（小角度下角度效果相同，区别在法兰位置摆动 hand_mount_z·sinθ）；调试用小角度
- 偏差标定 `anydex_eih_grasp_calib.py --calib-jog`：微调阶段在 Phase 2b 下降到位、手预塑形后、逐指闭合前。**键盘点动 = MoveJ + 以命令抓取位姿为基准累加**（不用 MoveL：其轨迹起点缓存 `_endPosturePast` 陈旧；也不用 `getpose` 当基准：见"已知系统偏移"）。平移 w/s/a/d/q/e（命令系 XYZ）+ 俯仰 r/f（绕法兰中心改 pitch，纯 rpy 调整时 offset.xyz=0，不污染平移分析）。记录字段 `target`/`before`/`after` 均为**命令系**位姿、`offset=after−before`(基座系，=操作员调整量，含 rpy/xyz)、`approach_base`、`depth_cam`。分析公式见 `analyze_jog_history.py`：`offset_xyz_new = 原值 + mean(offset)`、`mz_new = mz_cfg − mean(offset·approach)`、offset 对 `depth_cam` 线性拟合斜率显著⇒R_c2f 旋转误差需重标定/闭环。键盘自测 `python jog_manual.py --dry-run`（微调）/ `--descend-dry-run`（下降，均不动机械臂、不写文件）
- 预览 `run_inspire_preview.py` 的法兰框用相机系 `R_palm @ R_mount` 绘制（X 沿指尖，Y/Z 差 90°=安装滚转）；偏差公式与执行器同式（基座系），绿/橙框=是否回退固定朝向
- 手控制通过 `hand_bridge.py` 桥接，数据驱动预塑形 `PRESHAPE_LIBRARY`；`GraspHand` 接口：open/release（action_reset 复位：拇指 1000 半开、其余 2000、ID6 归 0）/preshape（空中广播含拇指内旋）/close（逐指闭合）；`--no-hand` = 无手桥时手部步骤用固定等待（HAND_WAIT_S）模拟，机械臂流程照常全自动
- 电机映射: ID1=拇 ID2=食 ID3=中 ID4=无名 ID5=小 ID6=拇指内旋(0=外展/张开 2000=贴掌)【2026-09-06 实机核对修正，旧注释方向写反】
- 编程时注意 `scripts/legacy/` 下是旧脚本/上游代码：旧 Z1 方案 (`z1_from_anydex_preview.py` 眼在手外, `z1_from_anydex_preview_eih.py` 旧 EIH, `yprxyz_to_z1.py`) + 上游 AnyDexGrasp 训练/数据集脚本 + 一次性调试脚本，不要与当前管线混淆；`scripts/tools/` 是验证/测试工具
