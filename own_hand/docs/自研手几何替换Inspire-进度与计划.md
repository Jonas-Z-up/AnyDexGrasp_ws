# 自研手几何替换 Inspire 模型 — 进度文档与执行计划

> 创建：2026-09-07。本文件是"用自研手 URDF 几何替换 Inspire 查表几何"工作的进度档案与任务清单，每步附预期产出与 Debug 判据，便于逐项核对。

---

## 一、已完成进度（截至 2026-09-07）

| # | 完成项 | 产出物 |
|---|---|---|
| 1 | URDF 结构确认：10 关节/6 独立自由度，四指 mimic 1:1，base_link +X=手指延伸方向 | — |
| 2 | 几何计算工具（pybullet FK + 贴掌端距离场分析），已验证运行 | `scripts/tools/compute_hand_geometry.py` |
| 3 | 自研手几何数据：贴掌端角（四指 90°/拇指内旋 120°）、电机→关节角映射、指尖开口、8 类型包络中心 | `docs/自研手几何数据.md` + `docs/hand_renders/` |
| 4 | ID6 方向实机修正：p=2000=贴掌、p=0=外展（旧注释写反，仅改注释，动作库数值不动） | `hand_bridge.py`/`CLAUDE.md` 注释修正 |
| 5 | 装配渲染工具 + **滚转 +90° 实机定案**（掌心帧↔自研手 base 帧的固定差） | `scripts/tools/render_arm_hand_assembly.py`（默认含修正）+ `docs/arm_renders/` |
| 6 | 双源漂移修复：编排器硬编码补偿改读 config，三端（执行/预览/标定）同源 | `anydex_eih_grasp.py:77-84` |
| 7 | 零手工测量标定流程文档 | `docs/自研手抓取重标定操作步骤.md` |
| 8 | 真机 jog 实测：**Inspire 模型偏差过大**（几何量化：Inspire 查表横向偏移 −72mm vs 自研手 −9~−18mm，厘米级且随手的朝向变化，`offset_xyz` 常量补不动）→ 决定替换为自研手几何 | 本计划 |
| 9 | 集成点调研完成（数据流/消费字段/参数影响面/已知坑） | 见下"关键事实" |

### 集成点关键事实（调研结论，设计依据）

- 预览 `run_inspire_preview.py` 同进程 import robot_inspire，自建管线；JSON 写出在 preview:289-309，13 个字段；
- **执行器只消费 4 个键**：`translation/rotation_matrix/score/grasp_type`——重锚定只需替换前两个的语义，零破坏；
- `chosen_two_finger`（graspnetAPI Grasp）在写 JSON 处可用，但碰撞检测已就地修改其 width（×1.7）和 translation（重定心到物体点云中心）——v1 **接受调整后的值**作锚定输入（它已对准物体）；
- 碰撞检测内部会对 Inspire 数组再查表（collision_detector.py:143）——own 模式在 detect() **之后**重锚，不受影响；
- `--random_grasp` 分支已损坏（类型取 0-7，查表 KeyError），预览默认走模型选型模式（线上实际路径）；
- `load_hand_config()` 共 6 个调用方，按键名不变方式加 `hand_model` 维度完全兼容；jog 历史记录含 hand_cfg，**勿改 4 个原键名**。

---

## 二、待办任务清单

| ID | 任务 | 状态 |
|---|---|---|
| D0 | 写入本文档 | ✅ 2026-09-07 |
| A1 | `own_hand/own_hand_grasp.py` 锚定模块 + selftest 渲染 | ✅ 三项判据全过（Type 3/5/6） |
| A2 | `hand_mount_z_own` 初值数值推导 | ✅ 77.5mm（own_hand/tools/derive_hand_mount_z_own.py） |
| B1 | config `hand_model` 参数组 + grasp_common 分支 | ✅ inspire 组逐字节兼容验证通过 |
| B2 | preview `--hand-model own` 重锚定 + 自研手 mesh 渲染 | ✅ 编译通过，待真机预览确认 |
| B3 | preview `--fixed-type` 参数 | ✅ |
| B4 | 变换链闭合离线验证 | ✅ 三项判据 0 误差；真机干跑留给用户 |
| C1 | jog 重标定（own 模式，**用户实机操作**） | ⬜ 待用户 |
| C2 | A/B 对比实验 inspire vs own 各 10 次 | ⬜ 待用户 |
| C3 | 文档收尾 | ✅ 本节 |

### 执行结果摘要（2026-09-07）

- **A1**：锚定数学 `T_world←base = T_world←grasp @ inv(T_base←grasp)`，其中
  `t_own=包络中心`（拇指/四指接触质心连线中点）、`R_own=[x=base+X, u=对向轴, x×u]`。
  selftest：包络中心重投影误差与对向轴夹角均为精确 0（缓存采样后确定性），
  渲染图（docs/hand_renders/own_anchor_selftest_t{3,5,6}.png）手包住圆柱、拇指在闭合轴。
- **A2**：hz_own = 77.5±12.8mm（8类型×3宽度）。跨类型散布 52mm 超判据，分析定论：
  Type 5（54mm 包络更深）与 Type 6/7（87-99mm）为**真实类型物理差异**（Inspire 查表
  本身也逐类型独立标定，旧系统单一 0.09 同样内置该不一致）；v1 用全局均值，A/B 固定
  Type 3（其逐类型值 77mm≈均值）；逐类型表留 v2（jog 按类型收敛后填入）。
- **B1**：`hand_model: "own"` → `hand_mount_z_own=0.0775 / mount_yaw_deg_own=0.0 /
  hand_fingertip_z_own=0.22`；`load_hand_config(path, model=None)` 按模型选组，
  原 4 键名不变（jog 历史兼容），inspire 组行为逐字节不变（已断言验证）。
- **B4 修出一个真 bug**：`mount_yaw_deg_own` 最初误推为 −180（把 +90° 装配滚转与旧
  −90° 错误链式相加）。闭合测试偏差闸门失败（171.8°>90° 恒回退固定朝向）暴露；
  正确值 **0**（装配定案的实际含义：自研手 base 帧 ≈ 法兰帧，手直接栓于法兰）。
  修正后三项判据 0 误差，闸门 24.8° 正常放行网络朝向。
- **真实 JSON 交叉验证**（logs/preview 实抓数据，Type 3）：base 与 Inspire 掌心沿
  接近轴差 +19.8mm（与 A2 一致）；**Inspire 掌心系相对接近轴倾斜 ~49°**（查表几何
  特性，正是自研手朝向系统性歪斜的来源之一）；相对滚转仅 4.2°（拇指方位与 Inspire
  约定基本兼容，选型语义可沿用）。

### 关键坐标约定（已用真实数据核实，勿再踩坑）

- 相机系 +Z = 视线方向（朝下看桌面）；
- graspnetAPI 两指抓取系 **X = 接近方向，指向物体**（沿视线），掌心/base 在抓取点
  −X 侧（机器人一侧）；Y = 开合轴；原点 = 抓取中心；
- 自研手 base 帧 +X = 手指延伸方向 ≈ 法兰 +X（装配定案，滚转 0）。

---

## 三、执行方案（每步含预期产出与 Debug 判据）

### Phase A：自研手锚定模块（纯离线，不碰机器人）

**A1. `own_hand/own_hand_grasp.py` — 核心锚定模块**

步骤：
1. 精读 `InspireHandR_grasp.py` 的 `_graspTR_2_InspireHandRTR / from_graspgroup / graspgroupTR_2_TR`，确定两指抓取系约定（X=接近、Y=开合）与查表复合公式的精确符号；
2. 复用 `compute_hand_geometry` 的 FK，按类型闭合位姿计算包络中心 C_base（用同事文档 §9 的 link 级接触定义）和对向轴（拇指→四指方向）；
3. 实现 `base_pose_from_grasp(two_finger_grasp, width, grasp_type)`：包络中心对准抓取点、对向轴对准开合轴——**+90° 装配滚转由此自然吸收，无需常数**；
4. `--selftest`：合成圆柱点云 + 已知抓取 → 位姿计算 + 自研手 mesh 渲染 PNG。

- 预期产出：模块文件 + `docs/hand_renders/own_anchor_selftest.png` + 控制台数值。
- **Debug 判据**：包络中心重投影到抓取点误差 <1mm（构造性）；对向轴与开合轴夹角 <1°；渲染图手包住圆柱、拇指在闭合轴上。

**A2. `hand_mount_z_own` 初值推导脚本**

- 原理：同一两指抓取下，旧链路法兰位置（Inspire 查表掌心 − 0.09×approach）应等于新链路法兰位置（自研手 base − hz_own×approach）→ `hz_own = 0.09 + (p_palm_inspire − p_base_own)·approach`，按 8 类型分别算取均值。
- 预期产出：一个数值 + 推导脚本（存 scripts/tools/）。
- **Debug 判据**：结果落在 0.06–0.13m 合理区间；跨类型散布 <25mm（物理常数不应随类型剧变；散布大说明 A1 几何有类型性错误）。

### Phase B：集成（预览 + 配置，执行器零改动）

**B1. config 参数组 + grasp_common 分支**

- `config/grasp_config.json` 新增：`"hand_model": "own"`、`hand_mount_z_own`（A2 值）、`mount_yaw_deg_own: -180.0`（由 +90° 定案推导：R_flange = R_base @ Rx(−180)）、`hand_fingertip_z_own: 0.22`；`grasp_common.load_hand_config()` 按 hand_model 返回对应组，**原 4 键名保持不变**（jog 历史兼容）。
- **Debug 判据**：`hand_model: "inspire"` 时 6 个调用方行为逐字节不变；own 时返回 own 组并打印确认。

**B2. preview `--hand-model own` 重锚定 + 自研手 mesh 渲染**

- preview:289-309 选出 `chosen` 后、写 JSON 前：own 模式用 A1 模块从 `chosen_two_finger` 重算 translation/rotation_matrix（depth/score/grasp_type 保留）；JSON 增写 `hand_model`、`two_finger_width/depth`；可视化分支加载自研手各 link mesh 按新位姿变换（替换红色 Inspire mesh），法兰框/指尖球自动用 B1 的 own 参数。
- **Debug 判据**：预览中自研手 mesh 包住目标物体、拇指朝向与两指抓取框 Y 轴一致；法兰框与 inspire 模式位置差在厘米级（非荒谬值）；JSON 字段完整。

**B3. preview `--fixed-type N` 参数**

- 在 `get_inspire_depth_type` 返回后覆盖 inspire_type/inspire_depth（1-8 编号）。
- **Debug 判据**：JSON 的 grasp_type 恒为指定值。

**B4. 变换链闭合离线验证**

- 用真实 `latest_inspire_grasp.json` 离线跑 `compute_grasp_target`（own 参数），与 preview 渲染法兰框的算法逐点对比；真机 `--no-hand` 干跑留给用户。
- **Debug 判据**：两条链算出的法兰位置差 <5mm（即预览所见=执行所得）。

### Phase C：标定与验证（用户实机操作）

**C1. jog 重标定**（own 模式，`--fixed-type 3`，≥5 次，历史写 `jog_calib_history_own.jsonl`）
- **判据**：初始未补偿偏差明显小于 inspire 模式实测；analyze 建议量 <3cm。

**C2. A/B 对比实验**：同物体组，inspire vs own 各 10 次，均 `--fixed-type 3`
- **判据**：own 成功率 ≥ inspire，失败模式从"位置偏"转向"夹持/时机"类。

**C3. 文档收尾**：进度文档更新结论、CLAUDE.md 状态、操作步骤文档补 own 模式用法。

---

## 四、v1 范围外（v2 备忘）

- 宽度插值的包络中心（v1 用闭合位姿常量）；
- 自研手碰撞点云替换 Inspire 碰撞过滤；
- 二阶段选型模型的自研手重训；
- `--random_grasp` 修复（现状不可用）。

## 五、风险与对策

| 风险 | 对策 |
|---|---|
| A1 符号约定错误 | selftest 渲染一眼可辨（手包反/拇指反向） |
| A2 推导散布大 | 说明 A1 类型性几何错误，先修 A1 再进 B |
| mount_yaw_own 依赖 +90° 目视定案 | B2 预览法兰框与实物照片再核一次 |
| jog 历史混样 | own/inspire 分文件记录，分开分析 |

## 六、F 阶段修正（2026-09-07 晚，真机预览反馈后）

真机预览发现手呈"指尖竖直插桌、掌心朝侧"异常（见 `Problem/现有问题.md` 与
`Problem/问题分析回执.md`）。根因：v1 锚定把抓取接近轴硬绑手指延伸轴，而上游
`angles2stl.py` 的抓取系定义是**接触三角形法向**。修正：

| # | 修正 | 结果 |
|---|---|---|
| F1 | 精读 angles2stl 公式：接近轴=接触三角形(拇指,食指两部位)法向、开合轴=拇指方向、锚点=中点公式 | 确认 v1 假设错误 |
| F2 | `own_hand_grasp.py` 重写构造 + 法向符号自动定向（参考体=base_link） | selftest 0 误差，渲染呈斜向包拢 |
| F3 | `search_contact_groups.py` 逐类型枚举接触组，锁定与查表约定面外倾斜最小的组合 | 倾斜 20-63°→**16-35°** |
| F4 | `hand_mount_z_own` 0.0775→**0.05**（物理安装距离常数，类型2/3/4 独立收敛 49-52mm 印证） | 消除逐类型拟合 |
| F5 | own 模式关闭朝向闸门（`orient_deviation_thresh_deg_own=180`）：法兰X=手指方向+逐类型拇指方位，全姿态偏差常超90°（滚转为主），旧闸门错误回退 | B4 回归三判据 0 误差 |

新增工具：`check_grasp_frame_convention.py`（约定校验）、`search_contact_groups.py`
（接触组搜索）。详析见 `Problem/问题分析回执.md`。

## 七、G 阶段修正（2026-09-08，用户判定 Ring 手势）

二次预览反馈：手指走向对了但**手势仍不对**（数值透视：手指 ⊥ 接近轴水平横伸、
掌面朝下呈"上方爪型"，而蓝色两指线框的语义是手指沿接近轴下探斜向包裹）。根因：
锚定用固定闭合位姿（手指蜷缩贴掌，指节连线折叠→三角形法向退化为掌面法向）；
上游资产是**按宽度逐档摆位**生成的（6d 表：600@2.5cm→891@10cm，越宽越开）。

渲染对比（`render_anchor_compare.py`/`render_ring_anchor.py`，用户判定）：
- 2×6d 宽度化方案"方向大致没问题"但细节不符；
- **用户选定 Ring 手势**（hand_bridge Type 2，动作库闭合值）：拇指+食指+中指围环，
  物体入"拇指 ↔ 食/中指缝隙"对向区间。方案 A（动作库值）确认采纳；
  方案 B（Inspire 6d）弃用——Inspire 的 "Ring" 实际只有拇指+食指（中指标记未参与），
  与我们语义不同。
- **概念厘清**：预抓取=空中广播值（全开）≠ 锚定姿态=闭合动作值（环成形态，
  物体入环）≠ 闭合终态（力控压紧≈锚定姿态）。锚定几何用闭合动作值。

实施：`TYPE_CONTACT_GROUPS[2]` = 拇指远节+食指远节+中指远节（环开口三点），
其余类型待逐类型真机判定。回归：selftest Type 2 / B4 全过。
**注意**：grasp_type 按 hand_bridge 语义解释（执行器本就直传 hand_bridge），
Ring 请用 `--fixed-type 2`。

## 八、H 阶段修正（2026-09-08，二次预览反馈）

1. **方向反转**：base 落在抓取点前方 +52mm（数值确诊）。根因：符号定向规则
   "n 远离 base_link"隐含"接近轴=掌面法向"，而 Ring 的接近轴=环轴（⊥掌面法向，
   实测 C·curl≈0），规则必然翻转 180°。→ 改为直接判据 **C·n>0**（环心在 base
   原点 +n 侧，手从后方够物体），裕度 51mm 确定性强，全手势统一。
2. **进深缺失**：锚定点原在 translation，但物体接触点 = translation +
   depth×接近轴（Inspire 执行时推进 depth+1.4cm，ur.py:441）。→
   `base_pose_from_grasp` 现读取 `.depth` 沿接近轴推进锚定点；
   `hand_mesh_at_grasp` 移除二次 depth 叠加。
3. **"平趴"外观澄清**：Ring 手势下香蕉躺在环面内（拇指/缝隙对向夹持），
   接近轴近竖直时环面必然近水平——手势正确，非 bug。低矮物体指尖包至
   桌面高度属固有现象，真机由接近倾角缓解、jog 标定收尾。
