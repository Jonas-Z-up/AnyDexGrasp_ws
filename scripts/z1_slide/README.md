# Z1 平移往复滑移测试（指尖传感器运动激励）

控制 Z1 机械臂做**纯平移往复直线运动**（姿态全程锁定），用于测试自研手指尖传感器的
运动激励程序。**仿真/真机通用**——同一套命令，只换底层控制器（仿真 `sim_ctrl`、
真机 `z1_ctrl`），上层脚本零改动。

- 贴面滑移模式：指尖（探针/真手）贴着桌面水平滑移，接触力近似恒定 —— 触觉滑移测试
- 悬空滑移模式：默认参数的水平往复（不对桌），用于纯运动学验证

## 本文件夹内容

| 文件 | 说明 |
|---|---|
| `z1_slide_test.py` | 往复滑移测试主脚本（子进程拉起 `../MoveZ1.py --server`） |
| `README.md` | 本文件 |
| `logs/` | 运行时自动生成的 CSV 数据（`slide_{axis}_{时间戳}.csv`） |

## 启动：三终端

### 终端 0 —— Gazebo 仿真环境（真机则跳到终端1部分）

```bash
cd ~/Projects/unitree_z1/Unitree-Z1_SDK/z1_ws
source /opt/ros/noetic/setup.bash && source devel/setup.bash
roslaunch unitree_gazebo z1.launch UnitreeGripperYN:=false TipProbeYN:=true wname:=earth_table
```

**预期效果**：Gazebo 窗口弹出；场景 = 地面 + 臂前方灰色桌面板（0.9×0.9m，顶面 z≈0，
即 x∈[0.2, 1.1] 区域）+ Z1 机械臂（无夹爪，法兰带红色探针小球）；日志出现
`Successfully spawned entity`，6 个关节控制器自动加载（`/z1_gazebo/Joint01..06_controller`）。

---

### 终端 1 —— 虚拟控制器（仿真）

```bash
cd ~/Projects/unitree_z1/Unitree-Z1_SDK/z1_controller/build
source /opt/ros/noetic/setup.bash && source ~/Projects/unitree_z1/Unitree-Z1_SDK/z1_ws/devel/setup.bash
./sim_ctrl
```
**预期效果**：持续打印 `connect with z1_sdk wait time out` —— 表示在等待终端 2 的
SDK 客户端连接，属正常；连接建立后停止刷屏。（真机时本终端无需存在，`z1_ctrl` 即终端 0。）

### 终端 1 —— 启动z1控制器
```bash
cd ~/Projects/unitree_z1/Unitree-Z1_SDK/z1_controller/build
./z1_ctrl
```

---

### 终端 2 —— 往复滑移测试（可反复执行，终端 0/1 无需重启）

```bash
cd ~/Projects/AnyDexGrasp_ws/scripts/z1_slide
source /opt/ros/noetic/setup.bash

# ★ X→Y 组合·直接切换版（X 结束后不回准备点，斜线直接进入 Y 往复；
#   贴面高度下指尖会保持接触、斜向拖过桌面）
/usr/bin/python3.8 z1_slide_test.py --axis x,y --direct-switch --stroke 0.2 --speed 0.3 --cycles 2 \
    --rpy 0 0.5 0 --x 0.40 --z 0.15 --tip-print

# ★ X→Y 组合（先 X 往复 2 次 → 自动回准备点 → 再 Y 往复 2 次）
/usr/bin/python3.8 z1_slide_test.py --axis x,y --stroke 0.2 --speed 0.1 --cycles 2 \
    --rpy 0 0.5 0 --x 0.40 --z 0.15 --tip-print

# X 轴贴面滑移（指尖贴桌，推荐工作点）
/usr/bin/python3.8 z1_slide_test.py --axis x --stroke 0.2 --speed 0.1 --cycles 2 \
    --rpy 0 0.5 0 --x 0.40 --z 0.15 --tip-print

# Y 轴贴面滑移
/usr/bin/python3.8 z1_slide_test.py --axis y --stroke 0.2 --speed 0.1 --cycles 2 \
    --rpy 0 0.5 0 --x 0.40 --z 0.15 --tip-print

```

**多轴组合说明**：`--axis x,y` 按"先 X 后 Y"依次执行，每轴各跑 `--cycles` 次；
每次换轴用 `pose` 命令进起点——它自带 **MoveJ → 基准准备点 → MoveL**，即自动
"回到准备位置"再切换方向。CSV 的 phase 列带轴前缀（`X_A_TO_B` / `Y_B_TO_A`...），
文件名也带轴序（如 `slide_x_y_时间戳.csv`）。
加 `--direct-switch` 则换轴不回准备点：从上一轴终点直接 MoveL 斜线到下一轴起点
（例：X 终点 (0.3, 0) → Y 起点 (0.4, −0.1)，斜线长 ≈0.14m），贴面高度下指尖保持
接触斜向滑过——对传感器是连续激励，但斜线路径上不能有杂物；首段仍经准备点进入。

**预期效果**（贴面模式）：
1. 启动打印 `[指尖] tip=(...)  高于桌面 +1.3 mm`（探针恰好贴桌）；
2. 臂先回零，再 MoveJ+MoveL 到起点 A；
3. 红色探针球**贴着桌面板**水平往返 `--cycles` 次（每段耗时 ≈ 行程/速度 + 加减速余量，
   去回对称；z 全程不变，纯平移）；
4. 结束自动回零下电；CSV 写入本文件夹 `logs/`；
5. Ctrl+C 任意时刻中断 → 同样安全回零下电。

## 参数说明

| 参数 | 默认 | 说明 |
|---|---|---|
| `--axis` | x | 滑移方向：`x`、`y`，或逗号组合依次执行（如 `x,y`），每轴各跑 `--cycles` 次 |
| `--stroke` | 0.20 | 总行程 A→B 距离（m） |
| `--x` / `--y` | 0.40 / 0.0 | 行程中点位置（m） |
| `--z` | 0.35 | 法兰固定高度（m），安全箱 [0.12, 0.60] |
| `--rpy R P Y` | 0 -0.094 0 | 全程锁定姿态（rad）。**正 pitch = 指尖下压**（Z1 约定） |
| `--speed` | 0.05 | MoveL 速度（m/s），范围 [0.005, 0.3] |
| `--cycles` | 5 | 往复次数（1 次 = A→B→A） |
| `--dwell` | 0.3 | 端点停顿秒数 |
| `--log` | 自动命名 | CSV 输出路径 |
| `--tip-print` | 关 | 打印指尖探针在端点处的估算高度 + 触桌法兰 z 换算 |
| `--direct-switch` | 关 | 多轴换轴时不回准备点，从上一轴终点直接斜线 MoveL 到下一轴起点（贴面高度下指尖保持接触斜向滑过）；首段仍经准备点进入 |

## 贴面滑移工作点（探针=伸直位中指尖，距命令帧 0.169m）

- 指尖高度公式：`tip_z ≈ z − 0.169·sin(pitch)`；**触桌条件：`z ≈ 0.002 + 0.169·sin(pitch)`**
- 推荐工作点：`pitch=0.96 (55°)` + `z=0.14` → 指尖高于桌面 +1.3mm
- pitch 更陡则法兰相应抬高：pitch=1.057 → z≈0.148；pitch=1.2 → z≈0.158
- 用 `--tip-print` 可对任意参数组合复核指尖高度

---

## 外部依赖（不在本文件夹，勿随意移动）
| 依赖 | 位置 | 说明 |
|---|---|---|
| `MoveZ1.py` | `../MoveZ1.py`（即 `scripts/MoveZ1.py`） | Z1 驱动子进程，与抓取管线共用，本脚本以相对路径引用 |
| `earth_table.world` | `~/Projects/unitree_z1/Unitree-Z1_SDK/z1_ws/src/unitree_ros/unitree_gazebo/worlds/` | 带桌面板的仿真世界（顶面 z≈0.002 对应真机桌面）。必须在该目录，`z1.launch` 按包路径解析 |
| 探针模型 | 同 SDK 包内 `robots/z1_description/xacro/robot.xacro` + `unitree_gazebo/launch/z1.launch` | 法兰上的红色指尖探针小球（`TipProbeYN` 开关，默认关闭，不影响其他用途） |

---

## 已知结论与注意事项（2026-09-27 实测）

1. **yaw-only 不可行**：yaw 旋转不改变探针高度（伸直指尖与法兰面几乎同高），
   且 `z=0.12 + yaw 0.8` 直接 IK 不可达。让伸直指尖到桌面必须靠正 pitch 下压；
   真机若用卷曲指尖（自然垂到法兰面下方），低法兰位 + 任意 yaw 值得实测。
2. **仿真抖动/卡住的教训**：给 Gazebo 模型加虚拟 link 时质量勿取极端小值
   （1e-6 kg 探针曾扰乱 ODE 求解，现 0.01 kg）；桌面板勿铺到臂基座下方
   （连杆下折会顶板，位置控制器与接触力对抗 → 卡住抖动）。板现已置于 x∈[0.2, 1.1]。
3. CSV 只在行程端点采样（协议阻塞式），对称性分析用 `dur_s` 列；行程中段轨迹
   如需连续曲线，可挂 `calibration/z1_pose_client.py` 的 socket 位姿服务（未实现）。
4. 真机首跑建议：保持 0.05 m/s、0.2m 行程，确认与仿真一致后再放大；真机动力学
   非理想，段耗时会略长、端点误差可能到 mm 级，CSV 有逐段记录可对比。
5. 后续扩展（未实现）：`--with-hand` 同步记录指尖力（`hand_control/controller_ub.py`
   的 `get_all_forces()`）；端点不停顿的匀速滑移（SDK CARTESIAN 速度模式）。
