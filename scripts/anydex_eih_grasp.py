#!/usr/bin/env python3
"""AnyDexGrasp 眼在手上 — 演示循环抓取编排脚本 (终端A)。

恒为自动循环: 每轮拍照推理后在终端A按一次回车确认新抓取位姿, 之后 Phase 2
(接近→张手+预塑形→严格直线下降→闭合→上提→丢弃→松手) 全自动执行, 手部步骤用
固定等待 (HAND_WAIT_S) 替代回车闸门; 松手后回到观测位姿循环下一轮。
进程仅在 Ctrl+C 或硬错误时退出(回零下电)。

用法:
  python anydex_eih_grasp.py              # 演示循环
  python anydex_eih_grasp.py --no-hand    # 无手桥: 手部步骤用固定等待模拟

所有可调参数在文件顶部「参数设置」区, 非命令行 flags。
标定(人工微调+记录)请用 anydex_eih_grasp_calib.py --calib-jog。
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid

import numpy as np

import grasp_common
import grasp_pipeline as gp
import jog_manual

# ── 手控制 ──────────────────────────────────────────────────
USE_HAND = "--no-hand" not in sys.argv
if USE_HAND:
    from hand_bridge import GraspHand


# ── 路径常量 ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.join(SCRIPT_DIR, "..")

MOVEZ1_SCRIPT = os.path.join(SCRIPT_DIR, "MoveZ1.py")
Z1_PYTHON = "/usr/bin/python3.8"

DEFAULT_GRASP_JSON = os.path.join(WS_ROOT, "logs", "preview", "latest_inspire_grasp.json")
DEFAULT_CALIB = os.path.join(WS_ROOT, "calibration", "cam2flange_current.json")

# ═══ 参数设置（demo 调参区）═══
# 所有可调参数集中在此；非命令行 flags。
DESCEND_STEP_MM = 10.0          # 可打断下降每步高度 (mm)
GRASP_WAIT_S = 1.0              # 到达抓取点后、闭合前等待秒数 (让手稳定对准)
HAND_WAIT_S = 3.0               # 手部步骤固定等待秒数
DROP_SETTLE_S = 1.0             # 松手后等待秒数 (看物体掉落)

# 准备点 — 安全中间位姿, MoveL 的可靠起点
PREP_RPYXYZ = (0.0, -0.094, 0.0, 0.2, 0.0, 0.3)
# 观测位姿 [roll, pitch, yaw, x, y, z]
OBSERVE_RPYXYZ = (0.0, 1.057, 0.0, 0.3, 0.0, 0.35)
# 抓取参数
APPROACH_Z = gp.APPROACH_Z       # 共享抓取计划的接近距离 (m)
LIFT_Z = gp.LIFT_Z               # 共享抓取计划的上提距离 (m)
FIXED_RPY = grasp_common.FIXED_RPY   # 固定抓取朝向（兜底）
SPEED = 0.15
# 中转节点 [roll, pitch, yaw, x, y, z] — 避障
TRANSIT_RPYXYZ = (0.0, -0.094, 0.0, 0.5, 0.0, 0.4)
# 丢弃位姿 [roll, pitch, yaw, x, y, z] — 抓取后移到这里松手
DROP_RPYXYZ = (0.0, -0.094, 0.5, 0.5, 0.2, 0.4)
# 安全范围
X_RANGE = (-0.05, 0.65)
Y_RANGE = (-0.50, 0.50)
Z_RANGE = (0.12, 0.70)

# 手安装/朝向参数 —— 统一从 config/grasp_config.json 读取（与标定脚本/预览同源）。
# 旧版此处为硬编码常量，会与 config 漂移导致"标定结果不迁移到演示循环"，已废除。
_hand_cfg = grasp_common.load_hand_config()
HAND_MOUNT_Z = _hand_cfg["hand_mount_z"]                          # 法兰→掌心沿接近轴 (m)
HAND_FINGERTIP_Z = _hand_cfg["hand_fingertip_z"]                  # 法兰→指尖 (m), 参考值
MOUNT_YAW_DEG = _hand_cfg["mount_yaw_deg"]                        # 拇指/掌心绕接近轴方位角 α
ORIENT_DEV_THRESH_DEG = _hand_cfg["orient_deviation_thresh_deg"]  # 网络朝向偏差阈值, 超则回退
OFFSET_XYZ = grasp_common.load_offset_xyz()                       # 基座系抓取位置偏移 (m)
OFFSET_RPY_DEG = grasp_common.load_offset_rpy_deg()               # 基座系绕掌心旋转偏移 (度)
Z_COMP_KNOTS = grasp_common.load_z_comp_knots()                   # z 高度补偿锚点 [命令z, 补偿量] (m)


# ── Z1 Server 客户端 ─────────────────────────────────────

class Z1Server:
    """MoveZ1.py --server 子进程管理。"""

    def __init__(self, speed=SPEED):
        cmd = [
            Z1_PYTHON, MOVEZ1_SCRIPT, "--server",
            "--no-start-home",
            "--speed", f"{speed:.6f}",
            "--x-range", f"{X_RANGE[0]:.6f}", f"{X_RANGE[1]:.6f}",
            "--y-range", f"{Y_RANGE[0]:.6f}", f"{Y_RANGE[1]:.6f}",
            "--z-range", f"{Z_RANGE[0]:.6f}", f"{Z_RANGE[1]:.6f}",
        ]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
            start_new_session=True,  # 独立进程组，Ctrl+C 不传播到子进程
        )
        # 后台线程持续排空 stdout，_expect 从队列读取以实现真正的超时
        self._lines = queue.Queue()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._expect("READY", timeout=30)

    def _cmd(self, s):
        self.proc.stdin.write(s + "\n")
        self.proc.stdin.flush()

    def _read_loop(self):
        """后台排空子进程 stdout：进度行即时透传终端，结果行进队列。

        子进程退出时（stdout 关闭）放入 None 哨兵，_expect 据此抛 EOFError，
        避免在死掉的子进程上无限阻塞。
        """
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            # 透传非结果行到终端，方便调试
            if line.startswith("[") or line.startswith("✅") or line.startswith("⚠"):
                print("  [Z1]", line, flush=True)
            self._lines.put(line)
        self._lines.put(None)

    def _expect(self, prefix, timeout=60):
        """等待子进程输出以 prefix 开头的结果行。

        timeout 秒内未收到 → 抛 TimeoutError；子进程已退出 → 抛 EOFError。
        通过队列读取，超时真实生效（不再是阻塞 readline 的空转超时）。
        """
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"等待 Z1 响应超时: {prefix}")
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(f"等待 Z1 响应超时: {prefix}")
            if line is None:
                raise EOFError("Z1 子进程已退出")
            if line.startswith("ERROR"):
                raise RuntimeError(f"Z1 错误: {line}")
            if line.startswith(prefix):
                return line[len(prefix):].strip()

    def pose(self, rpyxyz):
        """MoveJ 到准备点 → MoveL 到目标位姿（可靠，不会 IK 失败）。"""
        self._cmd("pose " + " ".join(f"{v:.6f}" for v in rpyxyz))
        self._expect("DONE pose")

    def jogpose(self, rpyxyz):
        """直接 MoveL 到目标位姿（仅适合小范围微调）。"""
        self._cmd("jogpose " + " ".join(f"{v:.6f}" for v in rpyxyz))
        self._expect("DONE jogpose")

    def movej(self, rpyxyz):
        """纯 MoveJ 关节空间运动 — 只算终点 IK，不会失败。"""
        self._cmd("movej " + " ".join(f"{v:.6f}" for v in rpyxyz))
        self._expect("DONE movej")

    def getpose(self):
        """查询当前法兰位姿 → [r, p, y, x, y, z]."""
        self._cmd("getpose")
        resp = self._expect("POS ")
        return np.array([float(v) for v in resp.split()], dtype=float)

    def check_ik(self, pose, seed=None):
        values = list(pose) + ([] if seed is None else list(seed))
        self._cmd("checkik " + " ".join("%.9f" % v for v in values))
        response = self._expect("IK ")
        if response == "NO":
            return False, np.zeros(6)
        parts = response.split()
        if len(parts) != 7 or parts[0] != "OK":
            raise RuntimeError("无效 IK 响应: " + response)
        return True, np.asarray([float(v) for v in parts[1:]])

    def movel(self, pose):
        self._cmd("movel " + " ".join("%.9f" % v for v in pose))
        self._expect("DONE movel")

    def home(self, timeout=60):
        """MoveJ 回机械零位；timeout 秒内无响应抛超时。"""
        self._cmd("home")
        self._expect("DONE home", timeout=timeout)

    def quit(self, timeout=5):
        """优雅退出子进程；超时或异常时强制结束。"""
        try:
            self._cmd("quit")
            self._expect("BYE", timeout=timeout)
            self.proc.wait(timeout=5)
        except Exception:
            print("[Z1] quit 超时或异常，强制结束子进程", flush=True)
            self._kill()

    def _kill(self):
        """强制结束子进程（幂等，已退出时无副作用）。"""
        try:
            self.proc.kill()
        except (ProcessLookupError, OSError):
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


# ── 主流程 ────────────────────────────────────────────────

def grasp_once(z1, hand, round_no):
    """执行第 round_no 轮完整抓取 (Phase 2), 返回状态码 0=成功, 1=失败。

    Phase 2a 接近 → 空中预塑形 → Phase 2b 严格直线下降(可中止) → 闭合 → 上提 → 丢弃 → 松手。
    手部步骤用固定等待 (HAND_WAIT_S) 替代回车闸门。
    失败 (抓取文件缺失 / 超出可达范围) 直接 return 1, 由 main() 续轮 (机械臂不断电)。
    """
    if not os.path.exists(DEFAULT_GRASP_JSON):
        print("[筛选] 请先运行 run_inspire_preview.py 生成新版抓取计划。")
        return 1
    with open(DEFAULT_GRASP_JSON) as f:
        result = json.load(f)
    observation = gp.load_observation()
    try:
        gp.validate_result(result, observation, z1.getpose())
    except (ValueError, KeyError, OSError) as exc:
        print("[筛选] 本轮未执行: %s" % exc)
        return 1

    report_path = os.path.join(os.path.dirname(result["capture_bundle"]), "execution.json")
    report = {"run_id": result["run_id"], "round": round_no,
              "status": "preflight", "decisions": [], "physical_success": None}
    selected = None
    for candidate in result["candidates"]:
        if (candidate.get("hand_model") != "own" or
                candidate.get("collision", {}).get("model") != "own_urdf_preshape" or
                not candidate.get("collision", {}).get("passed")):
            report["decisions"].append({"candidate_id": candidate.get("candidate_id"),
                                        "reason": "collision_not_passed"})
            continue
        # Rebuild from the canonical pose/config to detect stale or mismatched plans.
        expected = gp.build_plan(candidate, result["calibration"], result["capture_flange_pose"])
        saved = candidate["plan"]
        if any(not np.allclose(saved[k], expected[k], rtol=0, atol=1e-9) for k in
               ("final_pose", "pregrasp_pose", "lift_pose", "descent_poses", "lift_poses",
                "T_camera_hand", "pregrasp_direction_camera", "approach_distance")):
            raise ValueError("候选姿态与保存计划不一致，请重新推理")
        ok, reason = gp.check_plan_ik(saved, z1.check_ik, (X_RANGE, Y_RANGE, Z_RANGE))
        report["decisions"].append({"candidate_id": candidate["candidate_id"], "reason": reason})
        if ok:
            selected = candidate
            break
    if selected is None:
        report["status"] = "no_reachable_candidate"
        gp.write_json(report_path, report)
        print("[筛选] 所有候选均未通过执行前检查，本轮未发送抓取运动指令。")
        return 1

    plan = selected["plan"]
    grasp_type = int(selected["grasp_type"])
    final_pose = np.asarray(plan["final_pose"])
    report.update(status="selected", candidate_id=selected["candidate_id"], plan=plan)
    gp.write_json(report_path, report)
    gp.write_json(os.path.join(os.path.dirname(report_path), "selected_plan.json"), selected)
    print("[筛选] candidate_id=%d type=%d，已通过自研手碰撞与采样 IK 检查。" %
          (selected["candidate_id"], grasp_type))
    print("[目标] 法兰 rpyxyz=" + np.array2string(final_pose, precision=4))
    if selected["candidate_id"] != result["candidate_id"]:
        input("[筛选] 首选不可达，已改选上述候选；确认后按回车执行 (Ctrl+C 取消): ")
    # A long preflight/user confirmation must not make an expired result executable.
    gp.validate_result(result, gp.load_observation(), z1.getpose())
    gp.write_json(gp.OBSERVATION_PATH, {**observation, "state": "executing"})
    try:
        return execute_selected(z1, hand, round_no, grasp_type, plan, report, report_path)
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "execution_error",
                      error=str(exc), finished_at=time.time())
        gp.write_json(report_path, report)
        raise


def execute_selected(z1, hand, round_no, grasp_type, plan, report, report_path):
    approach = np.asarray(plan["pregrasp_pose"])
    final_pose = np.asarray(plan["final_pose"])
    rpy, p_flange = final_pose[:3], final_pose[3:]
    lift_target = np.asarray(plan["lift_pose"])
    transit, drop = np.asarray(TRANSIT_RPYXYZ), np.asarray(DROP_RPYXYZ)
    report.update(status="executing", phase="approach", started_at=time.time())
    gp.write_json(report_path, report)
    print("[执行] Phase 2a: MoveJ → 已检查的预抓取点")
    z1.movej(approach)

    if hand is not None:
        hand.open()                        # 张开手指
        hand.preshape(grasp_type)          # 空中预塑形(含拇指内旋),避免低空旋转撞桌

    report.update(phase="descent")
    gp.write_json(report_path, report)
    print("[执行] Phase 2b: 严格直线下降；Space/Esc 中止并回零")
    interrupted, _ = jog_manual.descend_interruptible(
        z1, rpy, approach[3:], p_flange,
        step_mm=DESCEND_STEP_MM, strict_linear=True)
    if interrupted:
        # Manual pose changes invalidate the checked plan. Calibration remains
        # available in anydex_eih_grasp_calib.py; demo never closes after a jog.
        raise KeyboardInterrupt

    report.update(phase="closure")
    gp.write_json(report_path, report)
    if hand is not None:
        print(f"[确认] 已到达抓取点，{GRASP_WAIT_S:.0f}s 后闭合抓取...")
        time.sleep(GRASP_WAIT_S)
        hand.close(grasp_type)             # 闭合抓取
    else:
        print(f"[手] --no-hand：等待 {HAND_WAIT_S:.0f}s 模拟人工抓取...")
        time.sleep(HAND_WAIT_S)

    print(f"[确认] 手已闭合，{HAND_WAIT_S:.0f}s 后自动上提 (Ctrl+C 可取消)...")
    time.sleep(HAND_WAIT_S)

    print(f"[执行] Phase 2c: 严格 MoveL → 上提 {LIFT_Z*100:.0f}cm")
    report.update(phase="lift")
    gp.write_json(report_path, report)
    z1.movel(lift_target)

    print(f"[执行] Phase 2d: MoveJ → 中转节点 (丢弃位姿上方)")
    report.update(phase="transit")
    gp.write_json(report_path, report)
    z1.movej(transit)

    print(f"[执行] Phase 2e: MoveJ → 丢弃位姿")
    report.update(phase="drop")
    gp.write_json(report_path, report)
    z1.movej(drop)

    print(f"[确认] 已到达丢弃位姿，{HAND_WAIT_S:.0f}s 后自动松手 (Ctrl+C 可取消)...")
    time.sleep(HAND_WAIT_S)

    if hand is not None:
        hand.release()                     # 松手

    report.update(status="sequence_completed", phase="released", finished_at=time.time())
    gp.write_json(report_path, report)
    print(f"\n[第 {round_no} 轮] 动作序列完成，实际抓取结果待观察确认。")
    print(f"[循环] {DROP_SETTLE_S:.0f}s 后自动返回观测位姿，等待下一轮 (Ctrl+C 取消)...")
    time.sleep(DROP_SETTLE_S)
    return 0


def main():
    print("=" * 55)
    print("  AnyDexGrasp 眼在手上 — 演示循环模式")
    print("    每轮: 拍照推理 → 回车自动抓取 → 松手后回观测位姿循环")
    print("    Ctrl+C 回零下电退出(唯一退出方式); 调参改文件顶部「参数设置」区")
    print("=" * 55)
    print("  (保持本终端: 关闭会掉力)")

    # ── 初始化 Z1 与手控制 ─────────────────────────────
    z1 = Z1Server()
    hand = None
    if USE_HAND:
        hand = GraspHand()
        if not hand.connect():
            print("[手] ⚠ 手控制不可用，将跳过手动作")
            hand = None

    try:
        round_no = 0
        while True:
            round_no += 1
            print(f"\n{'=' * 55}\n  第 {round_no} 轮\n{'=' * 55}")
            gp.write_json(gp.OBSERVATION_PATH, {"state": "moving"})
            print("  MoveJ → 准备点...")
            z1.movej(PREP_RPYXYZ)
            print("  MoveJ → 观测位姿...")
            z1.movej(OBSERVE_RPYXYZ)
            time.sleep(0.5)
            captured_pose = z1.getpose()
            if not gp.same_pose(captured_pose, OBSERVE_RPYXYZ):
                raise RuntimeError("观察位实际读回与命令不一致，请先验证法兰坐标系")
            gp.write_json(gp.OBSERVATION_PATH, {
                "id": uuid.uuid4().hex, "state": "ready", "ready_at": time.time(),
                "flange_pose": captured_pose.tolist(), "round": round_no})
            r, p, y, x, y_pos, z = OBSERVE_RPYXYZ
            print(f"  已到达观测位姿 r={r:.3f} p={p:.3f} y={y:.3f} "
                  f"x={x:.3f} y={y_pos:.3f} z={z:.3f}")
            print(f"  [拍照推理] 终端B: python run_inspire_preview.py --visualize --collision "
                  f"--preprocess --min_depth 0.2 --max_depth 0.6")

            input("\n>>> 确认抓取位姿无误，按回车执行抓取 (Ctrl+C 取消): ")

            print(f"── 第 {round_no} 轮 Phase 2: 执行抓取 ──")
            status = grasp_once(z1, hand, round_no)
            if status != 0:
                print(f"\n[第 {round_no} 轮] 本轮失败，返回观测位姿。重新拍照推理后按回车重试。")
                continue                      # 回循环顶 PREP→OBSERVE，机械臂不断电
            # 成功 → 自然进入下一轮: 循环顶重跑 PREP→OBSERVE 即"回到观测位姿"

    except KeyboardInterrupt:
        print("\n\n⚠️  用户取消，先回零再松手...")
        # 不设 SIG_IGN：清理期间再按一次 Ctrl+C 会立即抛 KeyboardInterrupt 强退
        # 不在 except 里松手: 低空(下降中/抓取点)直接张开手指易扫到桌面损坏手,
        # 统一放到 finally 在机械臂回零之后执行。
    finally:
        gp.write_json(gp.OBSERVATION_PATH, {"state": "closed"})
        print("\n[关闭] 安全退出...")
        # 顺序: 先机械臂回零(手随臂离开桌面到安全高度), 再松手复位灵巧手。
        # 方案 A：任何退出路径（正常完成 / Ctrl+C / 异常）下电前先回零。
        # 编排器已不再传 --no-return-home，MoveZ1 shutdown 默认也会 backToStart 兜底；
        # 这里先 home 让子进程下电前的 backToStart 成为空操作，
        # 避免 quit 等待超时把正在回零的子进程强杀。
        try:
            z1.home(timeout=30)
        except Exception:
            print("[Z1] 回零失败或超时，跳过回零", flush=True)
        # 机械臂回零后, 手已在安全高度, 再复位灵巧手(全开)并关串口
        if hand is not None:
            try:
                hand.release()
                hand.close_connection()
            except Exception:
                pass
        # 给子进程留足时间完成 shutdown 内的 backToStart + PASSIVE，超时/异常再强杀
        try:
            z1.quit(timeout=30)
        except Exception:
            pass
        print("✅ 完成。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
