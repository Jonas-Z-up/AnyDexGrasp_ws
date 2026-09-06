#!/usr/bin/env python3
"""自研手控制桥接模块。

预塑形动作库 PRESHAPE_LIBRARY 以数据驱动方式定义了 8 种抓取类型：
  - broadcast: 6 电缸预塑形位姿 [ID1..5(手指0~2000), ID6(拇指旋转0~2000)]
  - close:      逐指闭合指令列表 [(motor_id, force, speed, pos), ...]

GraspHand 类提供 open / preshape / close / release 接口，与 anydex_eih_grasp.py / anydex_eih_grasp_calib.py 闭环适配。
preshape() 在空中(抓取点上方)广播预塑形位姿(含拇指内旋),close() 到位后只做逐指闭合。
"""

import os, sys, time

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from hand_control.controller_ub import HandController

# ── 预塑形动作库 ──────────────────────────────────────────────
# 电机映射:
#   ID1: 大拇指屈伸   ID2: 食指    ID3: 中指
#   ID4: 无名指        ID5: 小指    ID6: 大拇指内旋 (0=外展/张开, 2000=贴掌)
#   [2026-09-06 实机核对修正: ID6 p增大→内旋加深; 旧注释"0=贴掌心向外,2000=垂直掌心"方向写反, 数值不受影响]
# 手指位置: 0=完全闭合,  2000=完全张开
# close 指令: (motor_id, force_g, speed, target_position)

PRESHAPE_LIBRARY = {
    1: {  # Tip Pinch — 拇指+食指尖端捏 (来自 tip_Pinch)
        "name": "Tip Pinch",
        "broadcast": [800, 2000, 2000, 2000, 2000, 1600],
        "wait": 1.5,
        "close": [
            (1, 4000, 1000, 300),    # 拇指
            (2, 4000, 1000, 1100),   # 食指
        ],
    },
    2: {  # Ring — 拇指+食指+中指形成环 (来自 finger_2)
        "name": "Ring",
        "broadcast": [800, 2000, 2000, 2000, 2000, 1800],
        "wait": 1.5,
        "close": [
            (1, 4000, 1000, 300),    # 拇指
            (2, 4000, 1000, 1100),   # 食指
            (3, 4000, 1000, 900),    # 中指
        ],
    },
    3: {  # Prismatic 3-Finger — 四指捏 (来自 finger_3)
        "name": "Prismatic 3-Finger",
        "broadcast": [800, 2000, 2000, 2000, 2000, 1800],
        "wait": 1.5,
        "close": [
            (1, 4000, 1000, 300),    # 拇指
            (2, 4000, 1000, 1100),   # 食指
            (3, 4000, 1000, 900),    # 中指
            (4, 4000, 1000, 1100),   # 无名指
        ],
    },
    4: {  # Large Diameter — 五指全合包络大直径物体
        "name": "Large Diameter",
        "broadcast": [800, 2000, 2000, 2000, 2000, 2000],
        "wait": 1.5,
        "close": [
            (1, 4000, 1000, 300),
            (2, 4000, 1000, 1100),
            (3, 4000, 1000, 900),
            (4, 4000, 1000, 1100),
            (5, 4000, 1000, 1100),
        ],
    },
    5: {  # Medium Wrap — 包络抓
        "name": "Medium Wrap",
        "broadcast": [800, 2000, 2000, 2000, 2000, 1200],  # 拇指内旋半握 1200
        "wait": 1.5,
        "close": [
            (1, 4000, 1000, 300),
            (2, 4000, 1000, 300),
            (3, 4000, 1000, 300),
            (4, 4000, 1000, 300),
            (5, 4000, 1000, 300),
        ],
    },
    6: {  # Tripod — 三脚支撑抓
        "name": "Tripod",
        "broadcast": [800, 2000, 2000, 2000, 2000, 100],
        "wait": 1.5,
        "close": [
            (1, 4000, 1000, 300),
            (2, 4000, 1000, 1800),
            (3, 4000, 1000, 300),
            (4, 4000, 1000, 300),
            (5, 4000, 1000, 300),
        ],
    },
    7: {  # Sphere 3-Finger — 球形三指抓
        "name": "Sphere 3-Finger",
        "broadcast": [800, 2000, 2000, 2000, 2000, 300],   # 拇指内旋深 800
        "wait": 1.5,
        "close": [
            (1, 4000, 1000, 300),
            (2, 4000, 1000, 1100),
            (3, 4000, 1000, 900),
        ],
    },
    8: {  # Distal Type — 远端指节捏
        "name": "Distal Type",
        "broadcast": [800, 2000, 2000, 2000, 2000, 2000],
        "wait": 1.5,
        "close": [
            (1, 4000, 1000, 400),
            (3, 4000, 1000, 900),    # 中指
            (4, 4000, 1000, 1100),   # 无名指
        ],
    },
}

# 人类可读名称
GRASP_TYPE_NAMES = {t: v["name"] for t, v in PRESHAPE_LIBRARY.items()}


# ── 手控制接口 ────────────────────────────────────────────────

class GraspHand:
    """自研五指欠驱灵巧手的抓取控制接口。"""

    def __init__(self, port=None):
        if port is None:
            port = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
        self.ctrl = HandController(port=port)
        self._connected = False

    def connect(self):
        ok = self.ctrl.connect()
        if ok:
            self._connected = True
            print("[手] 串口连接成功")
        else:
            print("[手] ⚠ 串口连接失败，手控制不可用")
        return ok

    @property
    def connected(self):
        return self._connected

    def open(self):
        """抓取前张开所有手指到全开位姿 (pos=2000)。"""
        if not self._connected:
            return
        print("[手] 张开所有手指...")
        self.ctrl.action_reset()
        print("[手] 已张开")

    def preshape(self, grasp_type):
        """空中预塑形：广播到该抓取类型的预塑形位姿（含 ID6 拇指内旋）。

        应在高处（抓取点上方安全距离）先执行，避免拇指内旋在低空扫到桌面。
        之后下降到位，再调用 close() 做逐指闭合。
        """
        if not self._connected:
            return

        preset = PRESHAPE_LIBRARY.get(grasp_type)
        if preset is None:
            print(f"[手] ⚠ 未知抓取类型 {grasp_type}，使用 Type 4 兜底")
            preset = PRESHAPE_LIBRARY[4]

        name = preset["name"]
        bc = preset["broadcast"]
        wait_t = preset["wait"]

        print(f"[手] 预塑形 Type {grasp_type} ({name})")
        print(f"     预塑形 broadcast: {bc}")
        self.ctrl.send_broadcast(bc)
        time.sleep(wait_t)
        print(f"[手] 预塑形到位")

    def close(self, grasp_type):
        """根据网络输出的抓取类型逐指闭合。

        调用前需先 preshape()（空中已完成预塑形广播），这里只做逐指闭合。
        """
        if not self._connected:
            print(f"[手] 未连接，跳过闭合 (type={grasp_type})")
            return

        preset = PRESHAPE_LIBRARY.get(grasp_type)
        if preset is None:
            print(f"[手] ⚠ 未知抓取类型 {grasp_type}，使用 Type 4 兜底")
            preset = PRESHAPE_LIBRARY[4]

        name = preset["name"]
        close_seq = preset["close"]

        print(f"[手] 闭合 Type {grasp_type} ({name})")
        for motor_id, force, speed, pos in close_seq:
            print(f"     motor {motor_id} → force={force} speed={speed} pos={pos}")
            self.ctrl.send_single_motor(motor_id, force, speed, pos)

        print(f"[手] 抓取完成")

    def release(self):
        """松开物体。"""
        if not self._connected:
            return
        print("[手] 松手...")
        self.ctrl.action_reset()
        print("[手] 已松开")

    def close_connection(self):
        if self._connected:
            self.ctrl.close()
            self._connected = False
            print("[手] 串口已关闭")
