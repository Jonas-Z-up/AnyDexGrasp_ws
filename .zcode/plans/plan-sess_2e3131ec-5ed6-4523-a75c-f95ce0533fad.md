## 修改计划：CLAUDE.md 误判修正 + 周边失效引用修复

依据：三路代码核验（约 60 条声明逐条对照代码/git 历史）。以下每条修改都附核验证据。

### A. CLAUDE.md 修正（确认误判，共 11 处）

1. **DROP_SETTLE_S 默认值**（:43、:65 两处）："默认 2s" → demo 版 **1.0s**（`anydex_eih_grasp.py:53`）；标定版默认 **3.0s**（`anydex_eih_grasp_calib.py:68`，`--drop-settle-s` 可调）。
2. **--calib-jog 表述**（:15、:29、:54 等处首次出现处）：注明"标定微调**默认开启**，`--calib-jog` 为兼容保留的 no-op，关闭用 `--no-calib-jog`"（`anydex_eih_grasp_calib.py:58-60`）。
3. **offset_rpy_deg 补当前值**（:69）：补充"当前 [0, **15**, 0]（config/grasp_config.json），pitch=15° 处于激活状态——每个抓取的执行朝向都被改写；是有意实验值还是残留待人工确认"。
4. **--no-hand 语义**（:72）："回退手动模式" → "无手桥时手部步骤用固定等待（HAND_WAIT_S）模拟，机械臂流程照常全自动"（`anydex_eih_grasp.py:33,427`）。
5. **assets 入库描述**（:32）："仅占位 .gitkeep 入库" → "仅 `own_urdf/.gitkeep` 入库；`inspire_urdf/` 整目录忽略、无任何文件入库"（git ls-files 核实）。
6. **anydex_core 清单**（:30）：补 `ur_toolbox/`（`filter_grasp_group_by_z_axis` 就定义在这里）。
7. **workspace 滤波表述**（:40）："三层统一为 ±0.40（已参数化）" → 改为准确版：`get_grasp` 默认 ±0.40、preview 的 `filter_grasp_group_by_z_axis` ±0.40（CLI 参数化）；`robot_inspire.py:755`（旧演示路径，不在现役 EIH 管线）硬编码 0.4 未参数化；点云 crop 层默认 ±0.5；`filter_grasp_group_by_z_axis` 首参 0.4 实为 score 阈值（上游命名误导）。
8. **check_frame_consistency**（:56）："配错即报错" → "配错时打印错误警告块，**不中止**初始化"（`Move_Key.py:221-229`）。
9. **"docs 曾记 [0.072,0,-0.085]"**（:58）→ "旧值曾存于 config git 历史（5301798）与 measured_results/README.md"（docs/ 目录从未记录过）。
10. **test_hand_actions 键位**（:47）："Q-I 闭合" → "Q-I = 预塑形+闭合完整动作，O 张开"（`test_hand_actions.py:18,41-49`）。
11. **PRESHAPE_LIBRARY 对应关系**（:46）："对应 controller_ub.py 动作" → "与 controller_ub.py 动作**同名**对应，参数各自独立"（broadcast/pos/force 均不同）；顺带 :72 open/release"全开"→"action_reset：拇指 1000 半开、其余 2000、ID6 归 0"（`controller_ub.py:112-118`）。

### B. CLAUDE.md 无出处数值标注（保留数值、注明证据状态）

- :57 "6 次 jog 反推相机位置 [0.414, 0.03, 0.356]" → 加注"⚠️ 无落盘记录（jog_calib_history.jsonl 已于 2026-09-01 删除，git 历史中亦无此值），待复测确认"。
- :58 "复测算得新板位置 [0.819, 0.236, -0.011]m" → 加注"⚠️ verify_on_robot.py 只打印不落盘，此值无记录，需重跑生成"。
- :58 "零畸变重投影 0.996px" → 加注"（仅文档文字记录；含畸变端 0.270px 有 realsense_intrinsics.json 0.26951 硬证据）"。
- :57 "深度 0.383 ≈ 尺量 0.38m" → 加注"（口头实测，无备忘记录）"。

### C. 周边文件失效引用修复（3 处）

1. `calibration/README.md:15、:82`：`board/ChArUco.jpeg` → `ChArUco.jpeg`（文件实际在 calibration/ 根，无 board/ 子目录）。
2. `calibration/README.md:58`：`calibration_report_20260906_charuco.json` → `calibration_charuco_current.json`（measured_results/README.md:4 用的才是实际文件名）。
3. `calibration/cam2flange_current.json:33`：cross_check 路径 `v2/cross_check_selfcal/` → `sessions/20260906_crosscheck/`（仅改路径字符串，不动数值）。

### D. 验证

- 修改后逐条 grep 复查：DROP_SETTLE_S、--calib-jog、offset_rpy_deg、四个标注数值、三个路径修复。
- git diff 自查，无代码改动（本次纯文档）。

### 遗留（不在本次范围，需你决策）

- **offset_rpy_deg pitch=15°**：实验值还是残留？确认后决定保留（文档已注明）或归零。
- 四个无出处数值的真实性需你复测/回忆确认。
- 代码疑点：IK 重试 8/6cm 成功后下降起点仍写死 12cm（`anydex_eih_grasp.py:414-416`），会先回升再下降——文档不改，代码是否修由你定。