#!/usr/bin/env python3
"""
Z1 位姿 IPC 共享模块 — Unix Domain Socket 协议。

Move_Key.py 运行服务端 (PoseServer), 其他脚本 (如 20_eye_in_hand_verify.py、
data_collect_manual.py) 运行客户端 (PoseClient)。

协议:
  Client → Server:  "GET\n"
  Server → Client:  '[0.075, 1.057, 0.073, 0.327, -0.08, 0.35]\n'  (JSON 数组, 6 个 float)
"""

import json
import os
import socket
import threading
from typing import Optional

import numpy as np

DEFAULT_SOCKET_PATH = "/tmp/z1_pose_server.sock"
RECV_TIMEOUT = 2.0  # 客户端接收超时 (秒)


# ══════════════════════════════════════════════════════════
# 服务端 — 嵌入 Move_Key.py, daemon 线程运行
# ══════════════════════════════════════════════════════════

class PoseServer:
    """Unix domain socket 服务端，提供当前 Z1 TCP 位姿。
    在独立 daemon 线程中运行，每次连接返回一次最新位姿。"""

    def __init__(self, socket_path: str = DEFAULT_SOCKET_PATH):
        self.socket_path = socket_path
        self._latest_posture: np.ndarray = np.zeros(6, dtype=float)
        self._lock = threading.Lock()
        self._server: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None

    # ── 生命周期 ──────────────────────────────────────────

    def start(self):
        """启动 daemon 线程，开始监听 socket。"""
        # 清理上次异常退出留下的陈旧 socket 文件
        try:
            os.unlink(self.socket_path)
        except OSError:
            if os.path.exists(self.socket_path):
                raise RuntimeError(
                    f"无法删除残留 socket 文件: {self.socket_path}\n"
                    f"请手动删除后重试。"
                )

        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self.socket_path)
        self._server.listen(5)
        self._server.settimeout(1.0)  # 允许定期检查退出信号
        self._thread = threading.Thread(
            target=self._serve_loop, daemon=True, name="pose-server"
        )
        self._thread.start()

    def stop(self):
        """关闭 socket 并删除文件。"""
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None
        try:
            os.unlink(self.socket_path)
        except OSError:
            pass

    # ── 线程安全更新 (由主线程调用) ────────────────────────

    def update_posture(self, posture: np.ndarray):
        """主循环调用，更新当前位姿缓存 (拷贝数组以保证线程安全)。"""
        with self._lock:
            self._latest_posture = np.copy(posture)

    # ── 内部 ──────────────────────────────────────────────

    def _serve_loop(self):
        """服务主循环: accept → 读取请求 → 返回 JSON → 关闭连接。"""
        while self._server is not None:
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue  # timeout 用于让线程有机会检查退出
            except OSError:
                break     # socket 已关闭

            with conn:
                try:
                    conn.settimeout(1.0)
                    data = conn.recv(256).decode("utf-8", errors="replace").strip()
                except (OSError, UnicodeDecodeError):
                    continue

                if data == "GET":
                    with self._lock:
                        p = self._latest_posture
                        response = json.dumps([
                            float(p[0]), float(p[1]), float(p[2]),
                            float(p[3]), float(p[4]), float(p[5]),
                        ])
                    try:
                        conn.sendall((response + "\n").encode())
                    except OSError:
                        pass


# ══════════════════════════════════════════════════════════
# 客户端 — 供 verify / data_collect 等脚本导入使用
# ══════════════════════════════════════════════════════════

class PoseClient:
    """Unix domain socket 客户端，向 Move_Key.py 的 PoseServer 请求当前 TCP 位姿。"""

    def __init__(self, socket_path: str = DEFAULT_SOCKET_PATH):
        self.socket_path = socket_path

    def get_posture(self, timeout: float = RECV_TIMEOUT) -> np.ndarray:
        """连接 socket → 发送 GET → 解析 JSON → 返回 [r, p, yaw, x, y, z] 数组。
        失败时抛出 RuntimeError, 附带中文错误信息。"""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect(self.socket_path)
            sock.sendall(b"GET\n")
            response = b""
            while b"\n" not in response:
                chunk = sock.recv(256)
                if not chunk:
                    break
                response += chunk
        except FileNotFoundError:
            raise RuntimeError(
                f"❌ 找不到 Pose 服务器 ({self.socket_path})\n"
                f"   请先在另一个终端启动 Move_Key.py"
            )
        except (ConnectionRefusedError, socket.timeout) as e:
            raise RuntimeError(
                f"❌ 无法连接到 Pose 服务器 ({self.socket_path}): {e}\n"
                f"   请确保 Move_Key.py 正在运行"
            )
        finally:
            sock.close()

        try:
            arr = json.loads(response.decode("utf-8").strip())
            if len(arr) != 6:
                raise ValueError(f"期望 6 个值，收到 {len(arr)} 个")
            return np.array(arr, dtype=float)
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"❌ 位姿数据解析失败: {e}\n原始数据: {response!r}")

    @staticmethod
    def check_available(socket_path: str = DEFAULT_SOCKET_PATH) -> bool:
        """快速检测 Pose 服务器是否可达 (不做完整 GET 请求)。"""
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect(socket_path)
            sock.close()
            return True
        except OSError:
            return False
