# controller.py
# 负责串口通信、协议封装、电机状态管理以及动作库逻辑。不包含任何界面代码。

import serial
import time
import struct
import threading

class HandController:
    # Windows 平台下使用 COM 端口:
    # def __init__(self, port='COM7', baudrate=921600):
    # Linux 平台下使用 /dev/serial/by-id/ 端口:
    def __init__(self, port='/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0', baudrate=921600):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.serial_lock = threading.Lock()

        # 记录ID1-5的最新目标值
        self.motor_current_pos = [2000] * 5
        self.log_callback = None

    def set_log_callback(self, func):
        self.log_callback = func

    def log(self, msg):
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(f"[Controller] {msg}")

    def connect(self):
        try:
            self.ser = serial.Serial(port=self.port, baudrate=self.baudrate,
                                     timeout=0.02, write_timeout=0.02)
            self.log(f"串口 {self.port} 连接成功")
            return True
        except Exception as e:
            self.log(f"串口打开失败: {e}")
            return False

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def _send_frame(self, data):
        """底层发送帧"""
        checksum = sum(data[2:]) & 0xFF
        data.append(checksum)

        if not self.ser: return

        self.ser.reset_input_buffer()
        self.ser.write(bytes(data))
        time.sleep(0.002)
        self.ser.read_all()

    def get_force_value(self, motor_id):
        """获取单个电机的力反馈值"""
        if not self.ser: return 0
        try:
            cmd_type = 0x30
            length = 0x03
            checksum = (length + motor_id + cmd_type) & 0xFF
            cmd_frame = bytes([0x55, 0xAA, length, motor_id, cmd_type, 0x00, 0x00, checksum])

            self.ser.reset_input_buffer()
            self.ser.write(cmd_frame)
            response = self.ser.read(19)

            if len(response) == 19 and response[0] == 0xAA and response[1] == 0x55:
                force_val = struct.unpack('<h', response[13:15])[0]
                return force_val
            return 0
        except Exception:
            return 0

    # === 获取所有手指力值 ===
    def get_all_forces(self):
        """
        依次读取ID1-5的力值并返回列表
        """
        forces = []
        with self.serial_lock:  # 加锁防止与绘图线程冲突
            for i in range(1, 6):
                forces.append(self.get_force_value(i))
        return forces

    def send_single_motor(self, _id, force, speed, pos):
        with self.serial_lock:
            raw_data = [0x55, 0xAA, 0x0D, _id, 0x32, 0x25, 0x00,
                        0x05, 0x00, 0x00, 0x00,
                        force & 0xff, (force >> 8) & 0xff,
                        speed & 0xff, (speed >> 8) & 0xff,
                        pos & 0xff, (pos >> 8) & 0xff]
            self._send_frame(raw_data)
        if 1 <= _id <= 5:
            self.motor_current_pos[_id - 1] = pos

    def send_broadcast(self, v_list):
        with self.serial_lock:
            num = 6
            data = [0x55, 0xAA, 1 + num * 3, 0xFF, 0xF2]
            for i, val in enumerate(v_list):
                data.append((i + 1) & 0xff)
                data.append(val & 0xff)
                data.append((val >> 8) & 0xff)
            self._send_frame(data)

    # ========= 动作库 =========
    # 0. 复位 RESET
    def action_reset(self):
        self.log("执行: 复位")
        self.send_single_motor(1, 4000, 2000, 500)
        time.sleep(1)
        for i in range(2, 6):
            self.send_single_motor(i, 1500, 2000, 2000)
        self.send_broadcast([500, 2000, 2000, 2000, 2000, 1000])

    # 1. Tip_Pinch
    def tip_Pinch(self):
        self.send_broadcast([2000, 2000, 2000, 2000, 2000, 1600])
        time.sleep(1.5)
        self.send_single_motor(1, 2500, 1000, 550)
        self.send_single_motor(2, 2500, 1000, 1100)

    # 2.Finger_2
    def finger_2(self):
        self.send_broadcast([2000, 2000, 2000, 2000, 2000, 1800])
        time.sleep(1.5)
        self.send_single_motor(1, 2500, 1000, 550)
        self.send_single_motor(2, 2500, 1000, 1100)
        self.send_single_motor(3, 2500, 1000, 900)

    # 3. Finger_3
    def finger_3(self):
        self.send_broadcast([2000, 2000, 2000, 2000, 2000, 1800])
        time.sleep(1.5)
        self.send_single_motor(1, 2500, 1000, 550)
        self.send_single_motor(2, 2500, 1000, 1100)
        self.send_single_motor(3, 2500, 1000, 900)
        self.send_single_motor(4, 2500, 1000, 1100)

    # 4. Finger_4
    def finger_4(self):
        self.send_broadcast([2000, 2000, 2000, 2000, 2000, 2000])
        time.sleep(1.5)
        self.send_single_motor(1, 2500, 1000, 550)
        self.send_single_motor(2, 2500, 1000, 1100)
        self.send_single_motor(3, 2500, 1000, 900)
        self.send_single_motor(4, 2500, 1000, 1100)
        self.send_single_motor(5, 2500, 1000, 1100)

    # 5. Medium_Wrap
    def medium_wrap(self):
        self.send_broadcast([2000, 2000, 2000, 2000, 2000, 1200])
        time.sleep(1.5)
        self.send_single_motor(1, 2500, 1000, 500)
        self.send_single_motor(2, 2500, 1000, 910)
        self.send_single_motor(3, 2500, 1000, 700)
        self.send_single_motor(4, 2500, 1000, 910)
        self.send_single_motor(5, 2500, 1000, 910)

    # 6. Tripod
    def tripod(self):
        self.send_broadcast([2000, 2000, 2000, 2000, 2000, 1200])
        time.sleep(1.5)
        self.send_single_motor(1, 2500, 1000, 550)
        self.send_single_motor(2, 2500, 1000, 1100)
        self.send_single_motor(3, 2500, 1000, 900)

    # 7. Sphere_3_Finger
    def sphere_3_finger(self):
        self.send_broadcast([2000, 2000, 2000, 2000, 2000, 800])
        time.sleep(1.5)
        self.send_single_motor(1, 2500, 1000, 550)
        self.send_single_motor(2, 2500, 1000, 1100)
        self.send_single_motor(3, 2500, 1000, 900)

    # 8. Distal_Type
    def distal_type(self):
        self.send_broadcast([2000, 2000, 2000, 2000, 2000, 2000])
        time.sleep(1.5)
        self.send_single_motor(1, 2500, 1000, 550)
        self.send_single_motor(3, 2500, 1000, 900)
        self.send_single_motor(4, 2500, 1000, 1100)

    # 数字手势 1-10
    def action_number_gesture(self, num):
        self.log(f"执行数字手势: {num}")
        if num == 1:
            self.send_single_motor(1, 2500, 2000, 200)
            self.send_single_motor(2, 2500, 2000, 2000)
            self.send_single_motor(3, 2500, 2000, 0)
            self.send_single_motor(4, 2500, 2000, 0)
            self.send_single_motor(5, 2500, 2000, 0)
            time.sleep(1)
        elif num == 2:
            self.send_single_motor(1, 2500, 2000, 200)
            self.send_single_motor(2, 2500, 2000, 2000)
            self.send_single_motor(3, 2500, 2000, 2000)
            self.send_single_motor(4, 2500, 2000, 0)
            self.send_single_motor(5, 2500, 2000, 0)
            time.sleep(1)
        elif num == 3:
            self.send_single_motor(1, 2500, 2000, 200)
            self.send_single_motor(2, 2500, 2000, 0)
            self.send_single_motor(3, 2500, 2000, 2000)
            self.send_single_motor(4, 2500, 2000, 2000)
            self.send_single_motor(5, 2500, 2000, 2000)
            time.sleep(1)
        elif num == 4:
            self.send_broadcast([2000, 2000, 2000, 2000, 2000, 1700])
            time.sleep(1)
            self.send_single_motor(1, 2500, 2000, 200)
            self.send_single_motor(2, 2500, 2000, 2000)
            self.send_single_motor(3, 2500, 2000, 2000)
            self.send_single_motor(4, 2500, 2000, 2000)
            self.send_single_motor(5, 2500, 2000, 2000)
        elif num == 5:
            self.send_broadcast([2000, 2000, 2000, 2000, 2000, 400])
            time.sleep(1)
        elif num == 6:
            self.send_single_motor(1, 2500, 2000, 2000)
            self.send_single_motor(2, 3000, 2000, 0)
            self.send_single_motor(3, 3000, 2000, 0)
            self.send_single_motor(4, 3000, 2000, 0)
            self.send_single_motor(5, 2500, 2000, 2000)
        elif num == 7:
            self.send_single_motor(1, 500, 2000, 2000)
            self.send_single_motor(2, 500, 2000, 2000)
            self.send_single_motor(3, 500, 2000, 2000)
            self.send_single_motor(4, 2500, 2000, 0)
            self.send_single_motor(5, 2500, 2000, 0)
        elif num == 8:
            self.send_single_motor(1, 500, 2000, 2000)
            self.send_single_motor(2, 500, 2000, 2000)
            self.send_single_motor(3, 2500, 2000, 0)
            self.send_single_motor(4, 2500, 2000, 0)
            self.send_single_motor(5, 2500, 2000, 0)
        elif num == 9:
            self.send_broadcast([2000, 2000, 2000, 2000, 2000, 1700])
            time.sleep(3)
            self.send_single_motor(1, 4000, 1800, 100)
            self.send_single_motor(2, 2500, 1800, 100)
            self.send_single_motor(3, 500, 2000, 2000)
            self.send_single_motor(4, 500, 2000, 2000)
            self.send_single_motor(5, 500, 2000, 2000)
            time.sleep(0.5)
        elif num == 10:
            self.send_broadcast([2000, 2000, 2000, 2000, 2000, 1000])
            time.sleep(1.5)
            self.send_single_motor(1, 2500, 800, 0)
            self.send_single_motor(2, 2300, 1500, 0)
            self.send_single_motor(3, 2300, 1500, 0)
            # self.send_single_motor(4, 12500, 2000, 0)
            self.send_single_motor(4, 2300, 1500, 0)
            self.send_single_motor(5, 2300, 1500, 0)
        else:
            self.log("无效的手势编号")