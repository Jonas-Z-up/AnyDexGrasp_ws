#!/usr/bin/env python3
"""Z1 抓取偏差根因验证脚本 (getpose vs MoveJ vs 物理实测)。

定位抓取系统约 42mm 偏移 + 深度相关(旋转)误差的根因,到底出在:
  (a) getpose / FK 报错   —— getpose 读到的位姿与真实法兰不符
  (b) MoveJ 落点不准       —— 命令的位姿与实际到达不符
  (c) 相机侧(标定/网络)   —— 机械臂运动本身没问题,偏差来自 cam2flange 或网络

原理: MoveJ 和 getpose 共用同一套 arm_model 运动学,但 getpose 从"编码器关节角"
      走 FK。因此:
        getpose − 命令  = MoveJ 实际到达 vs 命令           (自动, 无需尺子)
        物理   − getpose = FK 模型 vs 真实世界              (需尺子量一次)
        物理   − 命令    = 总误差

用法:
  python verify_z1_pose.py            # 真机全套测试
  python verify_z1_pose.py --dry-run  # 只打印要执行的位姿,不动机械臂

输出: 终端结论 + logs/pose_verify_history.jsonl (追加)

安全: 每步移动前确认; Ctrl+C 先回零再安全退出; 下电前 home。
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WS_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
MOVEZ1_SCRIPT = os.path.join(SCRIPT_DIR, "..", "MoveZ1.py")
Z1_PYTHON = "/usr/bin/python3.8"
LOG_PATH = os.path.join(WS_ROOT, "logs", "pose_verify_history.jsonl")

DRY_RUN = "--dry-run" in sys.argv

# 安全范围 (与 anydex_eih_grasp.py / anydex_eih_grasp_calib.py 一致)
X_RANGE = (-0.05, 0.65)
Y_RANGE = (-0.50, 0.50)
Z_RANGE = (0.12, 0.70)

# 观测位姿(抓取管线实际工作的姿态)
OBSERVE_RPYXYZ = (0.0, 1.057, 0.0, 0.3, 0.0, 0.35)

# 水平测量位姿: 臂水平前伸,法兰最高/最前都好量
HORIZONTAL_RPYXYZ = (0.0, 0.0, 0.0, 0.35, 0.0, 0.45)

# 姿态扫描位姿:
#   水平族(变 x, 固定 pitch=0)      → 检测 X 方向平移/标度误差
#   倾斜族(固定 xyz=(0.35,0,0.45), 变 pitch)
#                                   → 检测旋转误差(同一 xyz 不同 pitch,
#                                      getpose 若报不同 xyz ⇒ 旋转不一致)
#   观测位姿(0,1.057,0, 0.30,0,0.35) → 抓取管线实际工作的姿态,直接量 42mm
#   注: pitch=1.057 在 (0.35,0,0.45) 时 IK 不可达(实测),只能用观测位姿的 xyz
SCAN_POSES = [
    (0.0, 0.0, 0.0, 0.25, 0.0, 0.45),
    (0.0, 0.0, 0.0, 0.35, 0.0, 0.45),
    (0.0, 0.0, 0.0, 0.45, 0.0, 0.45),
    (0.0, 0.0, 0.0, 0.35, 0.0, 0.45),   # 与上一组同 xyz,检测重复性
    (0.0, 0.4, 0.0, 0.35, 0.0, 0.45),   # 倾斜族,与 pitch=0/0.8 共享 xyz
    (0.0, 0.8, 0.0, 0.35, 0.0, 0.45),   # 倾斜族
    (0.0, 1.057, 0.0, 0.30, 0.0, 0.35), # 观测位姿(已知可达)
]


def _check_range(pose):
    x, y, z = pose[3], pose[4], pose[5]
    if not (X_RANGE[0] <= x <= X_RANGE[1]):
        raise ValueError(f"x={x:.3f} 超安全范围")
    if not (Y_RANGE[0] <= y <= Y_RANGE[1]):
        raise ValueError(f"y={y:.3f} 超安全范围")
    if not (Z_RANGE[0] <= z <= Z_RANGE[1]):
        raise ValueError(f"z={z:.3f} 超安全范围")


# ── Z1 Server 客户端 (与 anydex_eih_grasp.py / anydex_eih_grasp_calib.py 同协议) ─────────────

class Z1Server:
    def __init__(self, speed=0.15):
        cmd = [
            Z1_PYTHON, MOVEZ1_SCRIPT, "--server", "--no-start-home",
            "--speed", f"{speed:.6f}",
            "--x-range", f"{X_RANGE[0]:.6f}", f"{X_RANGE[1]:.6f}",
            "--y-range", f"{Y_RANGE[0]:.6f}", f"{Y_RANGE[1]:.6f}",
            "--z-range", f"{Z_RANGE[0]:.6f}", f"{Z_RANGE[1]:.6f}",
        ]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, start_new_session=True,
        )
        self._lines = queue.Queue()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._expect("READY", timeout=30)

    def _cmd(self, s):
        self.proc.stdin.write(s + "\n")
        self.proc.stdin.flush()

    def _read_loop(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            if line.startswith("[") or line.startswith("✅") or line.startswith("⚠"):
                print("  [Z1]", line, flush=True)
            self._lines.put(line)
        self._lines.put(None)

    def _expect(self, prefix, timeout=60):
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

    def movej(self, rpyxyz):
        self._cmd("movej " + " ".join(f"{v:.6f}" for v in rpyxyz))
        self._expect("DONE movej")

    def getpose(self):
        self._cmd("getpose")
        resp = self._expect("POS ")
        return np.array([float(v) for v in resp.split()], dtype=float)

    def home(self, timeout=30):
        self._cmd("home")
        self._expect("DONE home", timeout=timeout)

    def quit(self, timeout=5):
        try:
            self._cmd("quit")
            self._expect("BYE", timeout=timeout)
            self.proc.wait(timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except (ProcessLookupError, OSError):
                pass


def _fmt(p):
    return (f"r={p[0]:+.3f} p={p[1]:+.3f} y={p[2]:+.3f} "
            f"x={p[3]:+.3f} y={p[4]:+.3f} z={p[5]:+.3f}")


def _rot_xyz(rpy):
    """rpy(rad, xyz 顺序) → 旋转矩阵 (与 grasp_common / scipy 'xyz' 约定一致)。"""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _log(rec):
    rec["ts"] = datetime.now().isoformat()
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _confirm(msg):
    if DRY_RUN:
        return True
    try:
        return input(msg).strip().lower() in ("", "y", "yes")
    except EOFError:
        return True


# ── 测试 A: 到位稳定性 ─────────────────────────────────────────

def test_settle(z1):
    print("\n" + "=" * 58)
    print("Test A 到位稳定性: MoveJ 后立即 getpose vs 延时 1.5s 再读")
    print("  目的: 排除 MoveJ 提前返回(还没真正到位就报 DONE)导致读偏")
    print("=" * 58)
    for i, pose in enumerate([OBSERVE_RPYXYZ, HORIZONTAL_RPYXYZ]):
        print(f"\n[A{i+1}] MoveJ → {_fmt(pose)}")
        if not _confirm("  移动机械臂,继续? [回车/y] "):
            print("  跳过")
            continue
        try:
            z1.movej(pose)
        except RuntimeError as e:
            print(f"  ⚠ 该位姿 IK 不可达,跳过: {e}")
            continue
        time.sleep(0.5)
        p0 = z1.getpose()
        time.sleep(1.5)
        p1 = z1.getpose()
        diff = p1 - p0
        diff_mm = diff * 1000.0
        diff_deg = np.degrees(diff[:3])
        print(f"  立即读: {_fmt(p0)}")
        print(f"  延时读: {_fmt(p1)}")
        print(f"  差值: xyz(mm)=[{diff_mm[3]:+5.1f},{diff_mm[4]:+5.1f},{diff_mm[5]:+5.1f}]  "
              f"rpy(°)=[{diff_deg[0]:+.1f},{diff_deg[1]:+.1f},{diff_deg[2]:+.1f}]")
        if np.abs(diff_mm[3:6]).max() > 5.0:
            print("  ⚠ 两次 getpose 相差 >5mm ⇒ MoveJ 可能提前返回,机械臂还在移动!")
        else:
            print("  ✓ 到位稳定(getpose 重复性好)")
        _log({"test": "settle", "pose": list(pose), "immediate": list(p0),
              "after_delay": list(p1), "diff_mm": list(diff_mm),
              "diff_rpy_deg": list(diff_deg)})


# ── 测试 B: 姿态扫描 (getpose − 命令) ──────────────────────────

def test_scan(z1):
    print("\n" + "=" * 58)
    print("Test B 姿态扫描: getpose − 命令 是否随姿态变化")
    print("  恒定 ⇒ 平移误差;  随 pitch/x 变大 ⇒ 旋转误差")
    print("  同一 xyz 不同 pitch 报不同 xyz ⇒ FK/旋转不一致")
    print("=" * 58)
    rows = []
    for i, pose in enumerate(SCAN_POSES):
        print(f"\n[B{i+1}] MoveJ → {_fmt(pose)}")
        if not _confirm("  继续? [回车/y] "):
            print("  跳过")
            continue
        _check_range(pose)
        try:
            z1.movej(pose)
        except RuntimeError as e:
            print(f"  ⚠ 该位姿 IK 不可达,跳过: {e}")
            continue
        time.sleep(0.5)
        gp = z1.getpose()
        pos = np.asarray(pose, dtype=float)
        err_mm = (gp[3:6] - pos[3:6]) * 1000.0        # xyz 位置误差 (mm)
        err_deg = np.degrees(gp[0:3] - pos[0:3])       # rpy 姿态误差 (度)
        forward = _rot_xyz(pos[:3])[:, 0]              # 命令朝向 +X(接近轴)在基座系的单位向量
        err_fwd = float(err_mm @ forward)              # 误差沿接近轴的分量 (mm)
        rows.append({"pose": list(pose), "getpose": list(gp),
                     "err_mm": list(err_mm), "err_rpy_deg": list(err_deg),
                     "err_forward_mm": err_fwd})
        print(f"  getpose: {_fmt(gp)}")
        print(f"  误差: x=[{err_mm[0]:+6.1f}] y=[{err_mm[1]:+6.1f}] z=[{err_mm[2]:+6.1f}] mm  "
              f"r=[{err_deg[0]:+5.2f}°] p=[{err_deg[1]:+5.2f}°] y=[{err_deg[2]:+5.2f}°]  "
              f"接近轴(+X)={err_fwd:+6.1f}mm")
    if rows:
        errs = np.array([r["err_mm"] for r in rows])  # (N,3) xyz
        spread = errs.ptp(axis=0)
        std = errs.std(axis=0)
        print(f"\n[B汇总] 误差 spread(mm): x={spread[0]:.1f} y={spread[1]:.1f} z={spread[2]:.1f}  "
              f"std: x={std[0]:.1f} y={std[1]:.1f} z={std[2]:.1f}")
        if spread.max() > 10.0:
            print(f"  ⚠ 基座系下误差随姿态变化达 {spread.max():.0f}mm ⇒ 并非基座系恒定平移;"
                  f"需看下面接近轴投影是否恒定")
        # 关键判据: 误差沿"命令朝向 +X"(接近轴)的投影是否恒定
        fwd = np.array([r["err_forward_mm"] for r in rows])
        print(f"[B接近轴] 各姿态下误差沿命令接近轴(+X)投影: "
              f"mean={fwd.mean():+6.1f}mm  std={fwd.std():.1f}mm  "
              f"min={fwd.min():+.1f} max={fwd.max():+.1f}")
        if fwd.std() < 8.0:
            print(f"  ✓ 接近轴投影基本恒定({fwd.mean():+.0f}mm) ⇒ 系统性'沿手前向偏{fwd.mean():.0f}mm'"
                  f"误差,与 CLAUDE.md 的 42mm 同源,基座系看到的变化只是方向不同")
        else:
            print(f"  ⚠ 接近轴投影也不恒定(std={fwd.std():.0f}mm) ⇒ 误差不单纯沿手前向,"
                  f"还需进一步区分")
        # 倾斜族内同一 xyz 的 getpose 是否一致
        tilted = [r for r in rows if abs(r["pose"][3] - 0.35) < 1e-9
                  and abs(r["pose"][4]) < 1e-9 and abs(r["pose"][5] - 0.45) < 1e-9]
        if len(tilted) >= 2:
            gps = np.array([r["getpose"][3:6] for r in tilted])
            rng = gps.ptp(axis=0) * 1000.0             # 米 → 毫米
            print(f"\n[B旋转一致] 同 xyz=(0.35,0,0.45) 不同 pitch 的 getpose 位置:")
            for r in tilted:
                print(f"    pitch={r['pose'][1]:+.3f}rad  getpose xyz=[{r['getpose'][3]:+.3f},"
                      f"{r['getpose'][4]:+.3f},{r['getpose'][5]:+.3f}]")
            print(f"    位置变化范围(mm): x={rng[0]:.1f} y={rng[1]:.1f} z={rng[2]:.1f}")
            if rng.max() > 5.0:
                print(f"    ⚠ 同一 xyz 不同 pitch 报出相差 {rng.max():.0f}mm ⇒ 旋转误差实锤"
                      f"(若物理法兰不动,则 getpose/FK 报错)")
            else:
                print(f"    ✓ 旋转时位置读数一致")
        _log({"test": "scan", "rows": rows})


# ── 测试 C: 物理尺量核对 (判别 getpose 错 vs MoveJ 错) ──────────

def test_physical(z1):
    print("\n" + "=" * 58)
    print("Test C 物理核对: 尺量法兰,判别 getpose 错还是 MoveJ 错")
    print("  关键判别:")
    print("    实测 ≈ getpose ≠ 命令  ⇒ MoveJ 落点不准(命令没到)")
    print("    实测 ≈ 命令   ≠ getpose ⇒ getpose/FK 报错")
    print("    三者一致                ⇒ 机械臂没问题,偏差在相机侧")
    print("=" * 58)
    pose = HORIZONTAL_RPYXYZ
    print(f"\n[C] MoveJ → 水平位姿 {_fmt(pose)}")
    if not _confirm("  确认机械臂前方/上方无遮挡,继续? [回车/y] "):
        return
    try:
        z1.movej(pose)
    except RuntimeError as e:
        print(f"  ⚠ 该位姿 IK 不可达,跳过: {e}")
        return
    time.sleep(0.5)
    gp = z1.getpose()
    print(f"\n  命令位姿 : x={pose[3]:+.3f} y={pose[4]:+.3f} z={pose[5]:+.3f}")
    print(f"  getpose  : x={gp[3]:+.3f} y={gp[4]:+.3f} z={gp[5]:+.3f}")
    print(f"  getpose−命令(mm): x={(gp[3]-pose[3])*1000:+.1f} "
          f"y={(gp[4]-pose[4])*1000:+.1f} z={(gp[5]-pose[5])*1000:+.1f}")
    print("\n  ── 请用尺子测量(约 ±2mm 即可) ──")
    print("  ① 基座回转中心(底座转轴)距桌面高度 —— 只需量一次:")
    try:
        z_base = float(input("     z_base (mm) > ").strip())
    except (ValueError, EOFError):
        print("  跳过 Z 比较")
        z_base = None
    try:
        h = float(input("  ② 法兰中心距桌面高度 (mm) > ").strip())
    except (ValueError, EOFError):
        h = None
    try:
        x_phys = float(input("  ③ 法兰距基座中心水平距离 X (mm, 用铅垂线投影到桌面卷尺; 可选) > ".rstrip()))
    except (ValueError, EOFError):
        x_phys = None

    rec = {"test": "physical", "pose": list(pose), "getpose": list(gp),
           "z_base_mm": z_base, "measured_h_mm": h, "measured_x_mm": x_phys}

    print("\n  ── 对比 (mm) ──")
    if z_base is not None and h is not None:
        phys_z = (h - z_base) * 1.0
        print(f"  Z: 命令={pose[5]*1000:+.1f}  getpose={(gp[5]*1000):+.1f}  实测(基座系)={phys_z:+.1f}")
        d_gp = gp[5] * 1000 - phys_z
        d_cmd = pose[5] * 1000 - phys_z
        print(f"     实测−getpose={d_gp:+.1f}   实测−命令={d_cmd:+.1f}")
        rec["phys_z_base_mm"] = float(phys_z)
        if abs(d_cmd) < abs(d_gp):
            print("  ⇒ 实测更接近【命令】⇒ getpose/FK 报错嫌疑大")
        elif abs(d_gp) < abs(d_cmd):
            print("  ⇒ 实测更接近【getpose】⇒ MoveJ 落点不准嫌疑大")
        else:
            print("  ⇒ 三个都差不少 ⇒ 再核对测量/可能是多因素")
    if x_phys is not None:
        print(f"  X: 命令={pose[3]*1000:+.1f}  getpose={(gp[3]*1000):+.1f}  实测={x_phys:+.1f}")
        d_gp = gp[3] * 1000 - x_phys
        d_cmd = pose[3] * 1000 - x_phys
        print(f"     实测−getpose={d_gp:+.1f}   实测−命令={d_cmd:+.1f}")
    _log(rec)


# ── 主流程 ────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("  Z1 getpose / MoveJ 根因验证")
    print(f"  日志: {LOG_PATH}")
    if DRY_RUN:
        print("  [DRY-RUN] 只打印位姿,不连机械臂")
    print("=" * 58)
    if DRY_RUN:
        print("\n[计划位姿]")
        print("  观测位姿:", _fmt(OBSERVE_RPYXYZ))
        print("  水平位姿:", _fmt(HORIZONTAL_RPYXYZ))
        print("  扫描位姿:")
        for p in SCAN_POSES:
            print("   ", _fmt(p))
        return 0

    z1 = Z1Server()
    try:
        for name, fn in [("A 到位稳定", test_settle),
                         ("B 姿态扫描", test_scan),
                         ("C 物理核对", test_physical)]:
            try:
                fn(z1)
            except Exception as e:
                print(f"\n⚠ [{name}] 测试中断: {type(e).__name__}: {e}")
        print("\n✅ 测试完成。")
        print("  关键判读: 看 [B接近轴] 投影是否 ~恒定(≈40mm)。若是 ⇒ 系统误差沿手前向,")
        print("   与 CLAUDE.md 的 42mm 同源,getpose/FK 嫌疑最大。")
        print("  建议下一步:")
        print("    1. 管线里用『命令观测位姿』替代 getpose() 做 cam→base(改动一行,")
        print("       立刻排除 40mm FK 误差),再用 jog 重测偏移是否大幅缩小")
        print("    2. 若仍偏,回头深挖 arm_model FK 的杆长/关节零位")
        print("    3. Test C 尺量只信『两次高度的差值』(不受基座原点参考影响),")
        print("       用于最终确认 MoveJ 落点精度")
    except KeyboardInterrupt:
        print("\n\n⚠️  用户取消,回零后退出...")
    finally:
        print("\n[关闭] 回零 + 安全退出...")
        try:
            z1.home()
        except Exception:
            print("[Z1] 回零失败,继续退出")
        try:
            z1.quit()
        except Exception:
            pass
        print("✅ 完成。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
