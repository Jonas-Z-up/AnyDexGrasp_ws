# AnyDexGrasp 眼在手上抓取复现

基于 AnyDexGrasp 论文的灵巧抓取复现，**当前为眼在手上方案**（D435i 安装在 Z1 末端）。

## 怎么跑

```bash
# 终端0: Z1 底层控制器
./z1_ctrl  (在 unitree_z1 SDK build 目录)

# 终端1: 相机
conda activate anydex && cd scripts && python realsense.py

# 终端A: Z1 编排器（常驻）
conda activate anydex && cd scripts && python anydex_eih_grasp.py

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
- `scripts/` — 入口脚本。`anydex_eih_grasp.py` 是主编排器，`MoveZ1.py` 是 Z1 驱动子进程
- `anydex_core/` — 不改动的算法库 (models, pointnet2, knn, utils, dataset)
- `calibration/` — `cam2flange_current.json` (当前眼在手上标定)
- `hand_control/` — 自研手串口控制
- `logs/` — 模型权重 + 推理输出

## 当前状态
- ✅ 阶段一 CGR 模型直接复用（权重: logs/model/checkpoint.tar.18）
- ✅ 阶段二 Inspire 手决策模型复用（权重: logs/model/inspire_model/obj140）
- ✅ 眼在手上推理管线打通（`anydex_eih_grasp.py` + `run_inspire_preview.py`）
- ✅ workspace 滤波三层统一为 ±0.40（`get_grasp` + `filter_grasp_group_by_z_axis` 已参数化）
- ✅ 点云预处理模块 `scripts/pointcloud_preprocess.py`（crop + RANSAC 桌面分割 + SOR 去噪，`--preprocess` 启用）
- ✅ `realsense.py` 默认开启实时相机预览窗口
- ✅ 全自动闭环抓取：`anydex_eih_grasp.py` 集成 `hand_bridge.py`——接近后**空中先张手+预塑形**（`preshape()` 广播含拇指内旋，避免低空旋转撞桌）→**步进下降 12cm（可打断）**→抓取点逐指闭合（`close()`）→上提→丢弃位姿→松手→**3s 后自动回零下电**
- ✅ 两处人工确认闸门：闭合后、到达丢弃位姿后各需按一次回车才继续（防碰撞/防误丢）
- ✅ 预塑形动作库 `PRESHAPE_LIBRARY`（数据驱动 8 种抓取类型，对应 `controller_ub.py` 动作）
- ✅ 预塑形调试工具 `scripts/test_hand_actions.py`（16 键直达 8 种类型：1-8 预塑形，Q-I 闭合，O 张开）
- ✅ 抓取朝向（已打通）：执行器"读网络 `rotation_matrix` + 安装偏移补偿（`hand_mount_z`）+ 角度偏差保护"已工作——`mount_yaw_deg=-90°` 修正后网络朝向偏差可低至 9.1°（真机实测），会采用网络朝向。偏差 = 命令法兰朝向与固定朝向（兜底）的夹角，超过 config `orient_deviation_thresh_deg`（当前 90°）回退固定朝向；预览绿框=采用网络朝向、橙框=回退固定朝向
- ✅ 真机验证工具 `scripts/validate_grasp_setup.py`：`--params` 打印参数/方向，`--target` 计算目标位姿，`--move` 移到检查位姿；**第 1 步静态朝向真机验证已通过**（指尖前下/掌心朝下/Ctrl+C 先回零）
- ✅ 根因验证工具 `scripts/verify_z1_pose.py`：自动化对比 getpose vs 命令位姿（到位稳定/姿态扫描/物理尺量三测），结果写入 `logs/pose_verify_history.jsonl`；已实测 getpose 沿接近轴恒定偏 ~39.4mm（见下方"已知系统偏移"）
- ⚠️ 碰撞检测默认开启，偶尔滤掉全部候选（预处理后应改善）
- ⚠️ `--preprocess` 会把桌面从点云和碰撞检测中一并滤除，**无指尖/掌心防穿桌检查**（只剩 MoveZ1 法兰 z≥0.12 兜底，指尖仍可能穿桌）；下降过程可按 **Space/Esc** 暂停接管人工微调避免碰撞
- ✅ 网络朝向真机执行验证（进行中）：偏差 ≤ config `orient_deviation_thresh_deg`（当前 90°）即采用网络朝向；真机已多次执行网络朝向抓取。指尖落点修正：`hand_mount_z` 0.12→0.09（整手沿接近轴推深）+ `offset_xyz` 基座系水平偏移（当前 [0.072, 0.0, -0.085]）
- ✅ 抓取偏差标定工具：`anydex_eih_grasp.py --calib-jog` 到达抓取点后进入键盘微调（w/s/a/d/q/e 基座系 XYZ + **r/f 俯仰**绕法兰中心，`+`/`-` 切平移步长 2/5/10mm 默认 5mm，`--jog-pitch-step-deg` 默认 1°），空格记录调整前后**命令位姿**与偏移（offset=操作员调整量）到 `logs/jog_calib_history.jsonl`（`--jog-history` 可改路径），回车在有微调时也会补记（防顺手按回车导致数据静默丢弃），`analyze_jog_history.py` 汇总恒定偏置/接近轴/深度相关/朝向残余四类误差并给出 `offset_xyz`/`hand_mount_z`/`offset_rpy_deg`(pitch) 建议
- ✅ 已知系统偏移（已定位并闭环，2026-08-08）：分两层。
  - **层1 getpose 虚报**：`getpose`(FK 读回) 与命令位姿差 **39.4±1.0mm 恒定沿接近轴**（与 42mm 同源）。真机实测法兰物理位置=命令位姿（MoveJ 落点正确）⇒ getpose 是**虚报**，cam→base 改用命令观测位姿 `OBSERVE_RPYXYZ`（a595c2e）**确认正确、不回退**。
  - **层2 主因=手眼外参错误**：`t_cam2flange` 把相机位置算错 **~11cm**（相机实际比外参说的靠前 ~11cm 沿法兰+X、高 ~3cm）。点云与网络已验证正确（网络两指抓点深度 0.383 ≈ 物理尺量 0.38m）；操作员 6 次 jog 修正量反推相机实际位置≈基座 [0.414, 0.03, 0.356]，与尺量吻合。当前 `offset_xyz=[0.072, 0, -0.085]` 即其**总账补偿**（抓取已精确，但改 OBSERVE/动相机需重推）。
  - ⬜ 外参根修：查棋盘格实际边长是否 12mm（`hand_in_eye_calibrate_by_aq.py --square-size`；边长错会系统性带偏 PnP→外参）→ 修正 `calibration/cam2flange_current.json` 或重标定 → offset 归零重测
- ⬜ 真机细调闭环（平移已收敛）：`offset_xyz=[0.072, 0, -0.085]` 后抓取已精确；剩余=手眼外参根修（见"已知系统偏移"层2）+ `offset_rpy_deg` 朝向残余的数据积累验证
- ⬜ 自研手阶段二：需采集试错数据 + 训练专属决策模型
- ⬜ 自研手 URDF/Mesh 用于碰撞检测

## 关键约定
- `MoveZ1.py` 作为子进程运行，走 `--server` 模式，通过 stdin/stdout 文本协议通信；MoveL 失败自动回退 MoveJ
- `start_new_session=True` 防止 Ctrl+C 传播到 Z1 子进程；下电前任何退出路径（正常完成/Ctrl+C/异常）都先 `home` 回零再断开——编排器不传 `--no-return-home`，MoveZ1 shutdown 默认 `backToStart` 兜底，断电停在零位不半空下垂；正常完成：松手后等 `DROP_SETTLE_S`（默认 3s，`--drop-settle-s` 可调）自动回零下电，无二次回车
- 移动策略: Phase1 MoveJ→准备点→MoveJ→观测位姿; Phase2 MoveJ→接近（IK 不可达时回退固定朝向、并降接近高度 12→8→6cm 重试；全失败打印"超出 Z1 可达范围"并优雅回零中止）→**空中张手+预塑形**→**步进下降**（`jog_manual.descend_interruptible`，`--jog-descend-step-mm` 默认 10mm；Space/Esc 暂停接管、Ctrl+C 中断回零）→抓取点逐指闭合→上提→丢弃位姿→松手→3s 后自动回零；闭合后/到丢弃位姿后各需回车确认
- 抓取朝向: 手与相机同轴沿法兰 **+X** 安装（指尖前下方，真机已验证）。规范手帧：palm +X=指尖、掌心朝 -Y（URDF 指节朝 -Y 弯曲）。物理安装 = Rx(+90°)（掌心朝法兰 -Z，默认状态掌心向下）；`mount_yaw_deg=-90°` 是**补偿符号**，使命令法兰放出的真实手 = 网络渲染（若设 +90° 会掌心翻转朝上，偏差~171° 全回退固定）。指尖方向=法兰+X首列，法兰位置=掌心−`hand_mount_z`×指尖方向（当前 0.09，减小=整手沿接近轴推深）。参数在 `config/grasp_config.json`（hand_mount_z/hand_fingertip_z/mount_yaw_deg/orient_deviation_thresh_deg/offset_xyz/offset_rpy_deg）
- 基座系抓取偏移 `offset_xyz`（config，当前 [0.072, 0.0, -0.085]）：整只手纯水平平移（不改变接近轴方向与 z），`grasp_common.load_offset_xyz`/`compute_grasp_target` 统一处理，执行器/预览/验证三处一致。改 x/y 分量即可水平微调
- 基座系旋转偏移 `offset_rpy_deg`（config，[roll,pitch,yaw]°）：offset_xyz 的旋转孪生——绕**掌心**中心、基座系旋转，作用于最终执行的朝向（网络或固定回退都生效）；在偏差闸门判定**之后**应用，故 `deviation_deg`/预览绿橙框不受影响。零偏移=旧行为。`grasp_common.load_offset_rpy_deg`/`compute_grasp_target` 统一处理，执行器/预览/验证三处一致（预览把基座系偏移共轭进相机系）。注意与 `--calib-jog` 的 r/f 俯仰不同：jog 绕**法兰**中心、offset_rpy 绕**掌心**中心（小角度下角度效果相同，区别在法兰位置摆动 hand_mount_z·sinθ）；调试用小角度
- 偏差标定 `--calib-jog`：微调阶段在 Phase 2b 下降到位、手预塑形后、逐指闭合前。**键盘点动 = MoveJ + 以命令抓取位姿为基准累加**（不用 MoveL：其轨迹起点缓存 `_endPosturePast` 陈旧；也不用 `getpose` 当基准：见"已知系统偏移"）。平移 w/s/a/d/q/e（命令系 XYZ）+ 俯仰 r/f（绕法兰中心改 pitch，纯 rpy 调整时 offset.xyz=0，不污染平移分析）。记录字段 `target`/`before`/`after` 均为**命令系**位姿、`offset=after−before`(基座系，=操作员调整量，含 rpy/xyz)、`approach_base`、`depth_cam`。分析公式见 `analyze_jog_history.py`：`offset_xyz_new = 原值 + mean(offset)`、`mz_new = mz_cfg − mean(offset·approach)`、offset 对 `depth_cam` 线性拟合斜率显著⇒R_c2f 旋转误差需重标定/闭环。键盘自测 `python jog_manual.py --dry-run`（微调）/ `--descend-dry-run`（下降，均不动机械臂、不写文件）
- 预览 `run_inspire_preview.py` 的法兰框用相机系 `R_palm @ R_mount` 绘制（X 沿指尖，Y/Z 差 90°=安装滚转）；偏差公式与执行器同式（基座系），绿/橙框=是否回退固定朝向
- 手控制通过 `hand_bridge.py` 桥接，数据驱动预塑形 `PRESHAPE_LIBRARY`；`GraspHand` 接口：open（全开）/preshape（空中广播含拇指内旋）/close（逐指闭合）/release（复位全开）；`--no-hand` 回退手动模式
- 电机映射: ID1=拇 ID2=食 ID3=中 ID4=无名 ID5=小 ID6=拇指内旋(0=贴掌心 2000=垂直)
- 编程时注意 `scripts/` 下有旧脚本 (`z1_from_anydex_preview.py` 眼在手外, `z1_from_anydex_preview_eih.py` 旧 EIH)，不要混淆
