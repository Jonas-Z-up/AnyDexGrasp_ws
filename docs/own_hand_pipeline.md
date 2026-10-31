# 自研手抓取管线（2026-10-31）

本次将 1030 的候选解码修正、自研手碰撞筛选思路、执行前 IK 筛选和离线记录机制适配到 ws。保留 ws 的手型编号、电机动作和手眼标定；未迁移 1030 的其他平台控制器或安装常数。`anydex_core/` 未改。

## 使用

按 README 启动控制器、相机和演示编排器。**更新后重启 `realsense.py`**，新版预览需要它提供的 `realsense_frame_meta` 共享内存，以避免读到写入一半的 RGB-D 数据。

终端 A 运行 `python anydex_eih_grasp.py`，到达观察位后会记录实际读回法兰位姿和本轮 ID；等待 A 提示后再在终端 B 运行：

```bash
conda activate anydex
cd ~/Projects/AnyDexGrasp_ws/scripts
python run_inspire_preview.py --hand-model own --collision --visualize \
    --preprocess --min_depth 0.2 --max_depth 0.6 --topk 10 --candidate-limit 50
```

预览显示**预塑形状态的自研手**和最终执行法兰坐标系。模型输入可去桌面/去噪，但碰撞场景保留原始深度里的桌面（有效深度 0–1.5 m）。默认碰撞余量 3 mm，平移扫掠步长不超过 3 mm。`--fixed-type 5` 使用该类型自己的评分和深度预测，不再给其他类型的结果改标签。

碰撞筛选逐候选输出进度、耗时及碰撞部件。默认一旦碰撞点数超过阈值就拒绝该候选，避免继续统计整条路径；通过的候选仍完成全路径检查。日志中的 `>=` 和记录中的 `collision_points_exact=false` 表示点数只是下界，不能用于比较碰撞严重程度。加 `--collision-full-count` 可恢复完整统计（较慢），检测几何、余量与步长不变。

全部候选被拒绝时，仍保存空的可执行候选列表，关闭诊断窗口后返回码为 6；启用可视化时展示排名第一的拒绝候选。窗口标题为 `COLLISION - DIAGNOSTIC ONLY`，红色手形处于检测到碰撞的扫掠位置，黄色标记为该位置的碰撞点采样（最多显示 30 点），法兰坐标系仍表示最终目标。检查日志中的碰撞部件、朝向回退与最终法兰位置，排查桌面干涉、标定、安装变换和高度补偿。此视图不表示可执行抓取，也不覆盖机械臂连杆；不自动放宽碰撞阈值。

回到 A 回车后，按评分顺序检查保留候选的下降和上提路径采样点：工作范围、SDK IK、关节限位和相邻解连续性。全部失败则本轮不发送抓取运动；若改选了预览首选之外的候选，会显示编号及法兰目标并再次等待确认。最后下降和上提使用严格 MoveL，失败不上切 MoveJ；下降时 Space/Esc 中止本次执行并进入既有回零清理流程。人工微调仍使用独立标定脚本。

结果仅限当前观察轮次，图像默认有效期 120 s。机械臂离开采集位、配置/标定/手部几何或相关代码变化后需要重新推理。没有观察记录时可以生成离线预览，但演示编排器不执行它。旧版 JSON、Inspire 模式和 `--no-collision` 结果不用于新版 own 自动执行。

## 记录和离线回放

每次 own 推理保存 `logs/preview/runs/<run_id>/`：

- `capture.npz`：原始 RGB-D、碰撞场景点、网络候选和对应特征。
- `result.json`：实际检查过的候选、拒绝原因、坐标变换、配置/源码哈希与采集文件校验值。
- `selected_plan.json`、`execution.json`：执行阶段选中的候选、IK 筛选结果及动作完成/失败阶段。

运行产物不入 Git。动作序列完成记录为 `sequence_completed`；没有视觉/触觉反馈，因此 `physical_success` 仍为空，不自动认定抓取成功。

在工作区根目录运行（无需相机或机器人）：

```bash
python scripts/tools/replay_grasp.py logs/preview/runs/<run_id>/result.json
# 用当前配置/标定重算坐标，再检查碰撞：
python scripts/tools/replay_grasp.py logs/preview/runs/<run_id>/result.json --replan
python -m unittest discover -s scripts/tests -v
python own_hand/tools/test_own_chain_closure.py
```

回放默认使用保存的法兰计划和当前手部网格，会报告源文件变化；`--replan` 还使用当前配置与标定重算计划。报告是诊断结果，不覆盖可执行的最新预览。记录目前使用绝对路径，移动工作区后需要同步修正记录中的路径。

## 坐标与参数

`grasp_pipeline.build_plan()` 是 own 预览、碰撞和执行共用的坐标入口。两指 depth 在自研手锚定时使用一次，碰撞阶段不重复添加。所有位置/朝向调试偏移、高度补偿与朝向回退先生成最终法兰目标，再反算碰撞/渲染所用手基座位姿。

当前安装参数保持原值。若以后测得完整安装矩阵，可在 `config/grasp_config.json` 添加 `T_flange_handbase_own`（4×4，`p_flange = T_flange_handbase_own @ p_handbase`，平移单位 m）。缺省继续使用原安装距离及角度约定。修改后重新推理；不要直接套用其他手的安装矩阵。

## 验证范围与限制

已做离线回归、真实 URDF/FK 网格加载及原坐标闭环测试；尚未启动设备或验证实机抓取成功率。

碰撞只覆盖**预塑形手部沿下降路径的平移扫掠**；不覆盖机械臂连杆、MoveJ 接近/中转/丢弃、手指张开到预塑形的过渡、闭合过程和带物体上提。现有 11 个 CAD 链接中 10 个不是封闭网格，对这些链接使用表面距离，不能保证识别完全落在网格内部的点。深度遮挡、无效像素和相机视野外区域仍然未知；采集元数据是主机时间及一致性检查，不是硬件级 RGB-D/机械臂同步。

评分仍来自 Inspire 模型迁移，尚未对自研手重新标定。采样 IK 也不能保证 SDK 实际 MoveL 必然成功或采用同一关节分支。因此这版提供可检查、可回放的软件链路，实机仍需从单物体和低速验证开始。
