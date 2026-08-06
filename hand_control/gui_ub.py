# -*- coding: utf-8 -*-
# gui.py - 主程序入口: main()
# 负责组装界面、初始化 Controller 和 Plotter，并处理用户交互事件

import tkinter as tk
from tkinter import ttk
import threading
import ast
import re
import os
import time  # 新增 time 模块用于连播间隔
from PIL import Image, ImageTk

# 导入拆分后的模块
from controller_ub import HandController
from plot_ub import ForcePlotter


class GraspGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("灵巧手控制台 & 物体识别系统")
        self.root.geometry("1800x1200")

        # 节流控制标志位，用于滑块实时发送指令
        self.slider_pending = False

        # 1. 初始化控制器
        # Windows 平台下使用 COM 端口:
        # self.controller = HandController(port='COM7')
        # Linux 平台下使用 /dev/serial/by-id/ 端口:
        self.controller = HandController(port='/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0')
        self.controller.set_log_callback(self.log_msg)

        # 连接串口
        if not self.controller.connect():
            self.log_msg("错误: 串口连接失败，请检查配置")

        # 2. 初始化绘图器
        self.plotter = ForcePlotter()

        # 3. 加载图片资源 (仅加载，不显示在主界面)
        self.images = {}
        self.load_images()

        # 4. 构建界面 (恢复 Checkbox 多选布局 + 大字体 + 新增滑块)
        self.setup_ui()

        # 5. 启动绘图刷新
        self.plotter.start_animation(self.controller)

    def load_images(self):
        """加载图片到内存"""
        img_files = {
            "Tennis": "Tennis.png",
            "MarkPen": "MarkPen.png",
            "Unknown": None
        }
        target_size = (350, 300)

        # === 优化寻找 figure 目录的逻辑 ===
        current_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(current_dir)
        figure_dir = os.path.join(parent_dir, "figure")

        for key, filename in img_files.items():
            if not filename: continue
            img_path = os.path.join(figure_dir, filename)
            if os.path.exists(img_path):
                try:
                    pil_img = Image.open(img_path)
                    pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
                    self.images[key] = ImageTk.PhotoImage(pil_img)
                    print(f"成功加载图片: {img_path}")
                except Exception as e:
                    print(f"加载图片 {img_path} 失败: {e}")
            else:
                print(f"警告：图片文件缺失: {img_path}")

    def setup_ui(self):
        # === 布局划分 ===
        left_panel = tk.Frame(self.root)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 右侧面板加宽以适应大字体
        right_panel = tk.Frame(self.root, width=380, bg="#f0f0f0", bd=2, relief=tk.GROOVE)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        right_panel.pack_propagate(False)

        # --- 左侧内容 ---
        plot_widget = self.plotter.create_widget(left_panel)
        plot_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(left_panel, pady=15)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X)

        tk.Label(control_frame, text="原始指令:", font=("Arial", 16)).pack(side=tk.LEFT, padx=10)
        self.entry_input = tk.Entry(control_frame, font=("Arial", 16))
        self.entry_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.entry_input.bind('<Return>', self.on_send_raw_click)
        tk.Button(control_frame, text="发送", font=("Arial", 14), command=self.on_send_raw_click).pack(side=tk.LEFT,
                                                                                                       padx=10)

        self.label_status = tk.Label(left_panel, text="状态: 就绪", font=("Arial", 12), bd=1, relief=tk.SUNKEN,
                                     anchor=tk.W)
        self.label_status.pack(side=tk.BOTTOM, fill=tk.X)

        # --- 右侧内容 ---
        tk.Label(right_panel, text="快速调参控制", font=("Arial", 22, "bold"), bg="#f0f0f0").pack(pady=20)

        # 1. ID 选择
        group_id = tk.LabelFrame(right_panel, text="选择 ID (可多选)", font=("Arial", 14, "bold"), bg="#f0f0f0",
                                 padx=10, pady=10)
        group_id.pack(fill=tk.X, padx=10, pady=5)

        self.selected_id_vars = {}
        for i in range(1, 7):
            var = tk.BooleanVar(value=False)
            self.selected_id_vars[i] = var
            tk.Checkbutton(group_id, text=f"M{i}", variable=var,
                           bg="#f0f0f0", font=("Arial", 18),
                           height=1, width=4, cursor="hand2"
                           ).grid(row=(i - 1) // 3, column=(i - 1) % 3, padx=5, pady=5)

        # 2. 参数设置 (包含滑块区)
        group_params = tk.LabelFrame(right_panel, text="参数设置", font=("Arial", 14, "bold"), bg="#f0f0f0", padx=5,
                                     pady=5)
        group_params.pack(fill=tk.X, padx=10, pady=5)

        self.entry_force = self.mk_entry(group_params, "力 (g):", 2300)
        self.entry_speed = self.mk_entry(group_params, "速度:", 1000)
        self.entry_pos = self.mk_entry(group_params, "位移:", 1000)

        # === 位移控制滑块 ===
        self.slider_pos = tk.Scale(group_params, from_=0, to=2000, orient=tk.HORIZONTAL,
                                   label=">> 滑块实时控制位移", font=("Arial", 12, "bold"),
                                   bg="#e1f5fe", fg="#0056b3", length=250, cursor="hand2",
                                   command=self.on_slider_change)
        self.slider_pos.set(2000)
        self.slider_pos.pack(fill=tk.X, padx=10, pady=15)

        tk.Button(right_panel, text="执行多/单电机指令", command=self.on_quick_control_send,
                  bg="#4CAF50", fg="white", font=("Arial", 18, "bold"),
                  height=2, width=20, relief=tk.RAISED, bd=2, cursor="hand2"
                  ).pack(fill=tk.X, padx=10, pady=5)

        # 3. 动作库
        group_actions = tk.LabelFrame(right_panel, text="固定动作库", font=("Arial", 14, "bold"), bg="#f0f0f0", padx=5,
                                      pady=10)
        group_actions.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        actions = [
            ("0. Reset", self.controller.action_reset),
            ("1. Tip_Pinch", self.controller.tip_Pinch),
            ("2. finger_2", self.controller.finger_2),
            ("3. finger_3", self.controller.finger_3),
            ("4. finger_4", self.controller.finger_4),
            ("5. Medium_Wrap", self.controller.medium_wrap),
            ("6. Tripod", self.controller.tripod),
            ("7. Sphere_3_Finger", self.controller.sphere_3_finger),
            ("8. Distal_Type", self.controller.distal_type)
        ]

        # 生成 0-7 基础按钮
        for text, func in actions:
            self.create_action_btn(group_actions, text, func)

        # === 8. 数字手势 区域 (更新按钮提示信息) ===
        num_gesture_frame = tk.Frame(group_actions, bg="#f0f0f0")
        num_gesture_frame.pack(fill=tk.X, pady=4, padx=5)

        # 右侧的输入框 (供用户输入0-10)
        self.entry_num_gesture = tk.Entry(num_gesture_frame, width=4, font=("Arial", 16), justify='center')
        self.entry_num_gesture.pack(side=tk.RIGHT, padx=(5, 0))
        self.entry_num_gesture.insert(0, "0")  # 默认填入0

        # 左侧的触发按钮
        btn_num = tk.Button(num_gesture_frame, text="9. 数字手势(0连播/1-10)", command=self.on_num_gesture_click,
                            bg="#e1f5fe", font=("Arial", 14), height=1)
        btn_num.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 4. 识别按钮
        tk.Button(right_panel, text="[识别] 识别物体&记录数据", command=self.identify_object_thread,
                  bg="#FF9800", fg="white", font=("Arial", 18, "bold"), height=2).pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=20)

    def mk_entry(self, parent, txt, val):
        f = tk.Frame(parent, bg="#f0f0f0")
        f.pack(fill=tk.X, pady=4)
        tk.Label(f, text=txt, width=10, bg="#f0f0f0", anchor="w", font=("Arial", 14)).pack(side=tk.LEFT)
        e = tk.Entry(f, width=10, font=("Arial", 14))
        e.insert(0, str(val))
        e.pack(side=tk.RIGHT, expand=True, fill=tk.X)
        return e

    def create_action_btn(self, parent, text, func):
        def task():
            self.log_msg(f"开始执行: {text}...")
            try:
                func()
                self.log_msg(f"执行完成: {text}")
            except Exception as e:
                self.log_msg(f"执行出错: {e}")

        btn = tk.Button(parent, text=text, command=lambda: threading.Thread(target=task, daemon=True).start(),
                        bg="#e1f5fe", font=("Arial", 14), height=1)
        btn.pack(fill=tk.X, pady=4, padx=5)

    def on_num_gesture_click(self):
        """数字手势点击事件处理 (支持 0 为自动连播功能)"""
        try:
            num = int(self.entry_num_gesture.get())
            if num == 0:
                # 自动连播任务
                def task_seq():
                    self.log_msg("开始执行: 数字手势自动连播 (1-10)...")
                    try:
                        for i in range(1, 11):
                            self.log_msg(f"自动连播: 正在展示手势 {i} / 10")
                            self.controller.action_number_gesture(i)
                            # 在这里加入 1.5 秒停顿，让动作展示更清晰且平稳过渡
                            time.sleep(1.5)
                        self.log_msg("执行完成: 数字手势自动连播")
                    except Exception as e:
                        self.log_msg(f"连播执行出错: {e}")

                # 放入后台线程防止界面卡死
                threading.Thread(target=task_seq, daemon=True).start()

            elif 1 <= num <= 10:
                # 单个手势任务
                def task():
                    self.log_msg(f"开始执行: 数字手势 {num}...")
                    try:
                        self.controller.action_number_gesture(num)
                        self.log_msg(f"执行完成: 数字手势 {num}")
                    except Exception as e:
                        self.log_msg(f"执行出错: {e}")

                threading.Thread(target=task, daemon=True).start()
            else:
                self.log_msg("错误: 数字手势范围必须是 0 到 10 (0为自动连播)")
        except ValueError:
            self.log_msg("错误: 请在框内输入有效的数字")

    # ==================== 滑块实时控制核心逻辑 ====================
    def on_slider_change(self, val):
        """当滑块被拖动时触发"""
        pos = int(val)
        # 1. 实时更新上方的"位移"输入框，保持界面数值同步
        self.entry_pos.delete(0, tk.END)
        self.entry_pos.insert(0, str(pos))

        # 2. 节流机制 (Throttle): 防止串口被瞬间发送的几百条指令卡死
        if not self.slider_pending:
            self.slider_pending = True
            # 等待 50 毫秒后统一发送最新的滑块位置 (20Hz的丝滑刷新率)
            self.root.after(50, self.apply_slider_pos)

    def apply_slider_pos(self):
        """执行发送最新的滑块位置"""
        self.slider_pending = False

        selected_ids = self.get_selected_ids()
        if not selected_ids:
            return  # 如果没有勾选任何ID，则滑块只改变数值，不发送指令

        try:
            force = int(self.entry_force.get())
            speed = int(self.entry_speed.get())
            pos = int(self.slider_pos.get())
        except ValueError:
            return

        def task():
            if 6 in selected_ids:
                broadcast_data = self.controller.motor_current_pos.copy()
                broadcast_data.append(pos)
                self.controller.send_broadcast(broadcast_data)

            for tid in selected_ids:
                if 1 <= tid <= 5:
                    self.controller.send_single_motor(tid, force, speed, pos)

        # 放到后台线程发送，彻底保证主界面不掉帧
        threading.Thread(target=task, daemon=True).start()

    # ==============================================================

    def log_msg(self, msg):
        self.root.after(0, lambda: self.label_status.config(text=f"状态: {msg}"))

    def get_selected_ids(self):
        """获取当前勾选的所有ID列表"""
        return [i for i, var in self.selected_id_vars.items() if var.get()]

    def on_quick_control_send(self):
        try:
            selected_ids = self.get_selected_ids()
            if not selected_ids:
                self.log_msg("错误: 请至少选择一个 ID")
                return

            force = int(self.entry_force.get())
            speed = int(self.entry_speed.get())
            pos = int(self.entry_pos.get())

            if not (0 <= pos <= 2000):
                self.log_msg(f"错误: 位移 {pos} 超出范围 0-2000")
                return

            # 手动点击发送时，同步更新一下滑块的位置
            self.slider_pos.set(pos)

            if 6 in selected_ids:
                broadcast_data = self.controller.motor_current_pos.copy()
                broadcast_data.append(pos)
                self.controller.send_broadcast(broadcast_data)
                self.log_msg(f"多选控制: ID6 -> Pos {pos}")

            for tid in selected_ids:
                if 1 <= tid <= 5:
                    self.controller.send_single_motor(tid, force, speed, pos)
                    self.log_msg(f"多选控制: ID{tid} F:{force} S:{speed} P:{pos}")

        except ValueError:
            self.log_msg("错误: 请输入有效的整数参数")
        except Exception as e:
            self.log_msg(f"执行错误: {e}")

    def identify_object_thread(self):
        def task():
            self.log_msg("正在读取传感器数据...")
            forces = self.controller.get_all_forces()
            self.root.after(0, lambda: self.process_and_popup(forces))

        threading.Thread(target=task, daemon=True).start()

    def process_and_popup(self, forces):
        print(f"DEBUG: Forces: {forces}")
        f1, f2, f3, f4, f5 = forces

        result_name = "Unknown"
        display_text = "未知物体"

        def is_close(val, target, tol=1000):
            return abs(val - target) <= tol

        # Tennis: ID4=-9000~-11000, ID2=-3000
        if (-12000 <= f4 <= -8000) and is_close(f2, -2600):
            result_name = "Tennis"
            display_text = "网球 (Tennis)"
        # MarkPen: ID4=-4000, ID2=-1700
        elif is_close(f4, -4000, 1500) and is_close(f2, -2500, 1200):
            result_name = "MarkPen"
            display_text = "记号笔 (Marker)"

        self.show_popup_result(result_name, display_text, forces)
        self.log_msg(f"识别完成: {display_text}")

    def show_popup_result(self, key_name, display_text, forces):
        top = tk.Toplevel(self.root)
        top.title("传感器数据")
        top.geometry("500x600")
        top.attributes('-topmost', True)

        tk.Label(top, text="识别结果:", font=("Arial", 16)).pack(pady=(15, 0))
        tk.Label(top, text=display_text, font=("Arial", 28, "bold"), fg="blue").pack(pady=5)

        if key_name in self.images and self.images[key_name]:
            tk.Label(top, image=self.images[key_name], bd=2, relief="ridge").pack(padx=20, pady=10)
        else:
            tk.Label(top, text="[ 未找到对应图片 ]", font=("Arial", 14), fg="gray", height=5).pack(pady=20)

        frame_data = tk.LabelFrame(top, text="传感器原始数据 (ID1-5)", font=("Arial", 12, "bold"), padx=10, pady=10)
        frame_data.pack(fill=tk.X, padx=20, pady=10, side=tk.BOTTOM)

        force_str = ",  ".join([str(x) for x in forces])
        tk.Label(frame_data, text=f"[{force_str}]", font=("Consolas", 15)).pack()

        tk.Button(top, text="关闭", command=top.destroy, bg="#ddd", font=("Arial", 12), width=10).pack(pady=5,
                                                                                                       side=tk.BOTTOM)

    def on_send_raw_click(self, event=None):
        cmd_text = self.entry_input.get().strip()
        if not cmd_text: return
        self.entry_input.delete(0, tk.END)
        try:
            blocks = re.findall(r'\[[^\]]+\]', cmd_text)
            for blk in blocks:
                data = ast.literal_eval(blk)
                if len(data) == 2 and data[0] == 6:
                    broadcast_data = self.controller.motor_current_pos + [data[1]]
                    self.controller.send_broadcast(broadcast_data)
                elif len(data) == 4 and 1 <= data[0] <= 5:
                    self.controller.send_single_motor(*data)
        except Exception as e:
            self.log_msg(f"指令错误: {e}")

    def on_close(self):
        self.log_msg("正在退出程序...")
        if hasattr(self, 'plotter'):
            self.plotter.stop()
        if hasattr(self, 'controller'):
            self.controller.close()
        self.root.destroy()
        import os
        os._exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = GraspGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()