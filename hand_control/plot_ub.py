# plot.py
# 封装 Matplotlib 的绘图逻辑和数据缓存。通过调用 Controller来获取数据，但不涉及底层串口细节

import collections
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

class ForcePlotter:
    def __init__(self, history_len=200):
        self.history_len = history_len
        self.data_queues = {
            i: collections.deque([0] * history_len, maxlen=history_len)
            for i in range(1, 6)
        }

        # 初始化 Matplotlib 图表
        self.fig, self.ax = plt.subplots(figsize=(8, 6), dpi=100)
        self.ax.set_xlim(0, history_len)
        self.ax.set_ylim(-10000, 5000)

        # === 字体优化部分 ===
        self.ax.set_title("Real-time Force Feedback", fontsize=20)  # 标题放大 14->20
        self.ax.set_ylabel("Force (g)", fontsize=16)  # 轴标签放大 12->16
        self.ax.tick_params(axis='both', which='major', labelsize=12)  # 刻度数字放大
        self.ax.grid(True, alpha=0.3)

        self.colors = ['r', 'g', 'b', 'orange', 'purple']
        self.lines = []
        for i in range(5):
            line, = self.ax.plot([], [], color=self.colors[i], label=f'ID {i + 1}=0', lw=2)
            self.lines.append(line)

        # 图例字体放大
        self.leg = self.ax.legend(loc='upper left', prop={'size': 14})  # 图例 10->14
        self.controller = None

        # ... (其余方法 create_widget, update_plot, start_animation, stop 保持不变) ...

    def create_widget(self, parent_frame):
        canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        widget = canvas.get_tk_widget()
        return widget

    def update_plot(self, frame):
        if not self.controller:
            return self.lines
        with self.controller.serial_lock:
            for i in range(1, 6):
                val = self.controller.get_force_value(i)
                self.data_queues[i].append(val)
        for i in range(1, 6):
            self.lines[i - 1].set_data(range(self.history_len), self.data_queues[i])
        if self.leg:
            texts = self.leg.get_texts()
            for i, text_obj in enumerate(texts):
                current_val = self.data_queues[i + 1][-1]
                text_obj.set_text(f"ID {i + 1} = {current_val}")
        return self.lines + [self.leg]

    def start_animation(self, controller_instance):
        self.controller = controller_instance
        self.ani = FuncAnimation(
            self.fig,
            self.update_plot,
            interval=50,
            blit=True,
            cache_frame_data=False
        )

    def stop(self):
        if hasattr(self, 'ani') and self.ani.event_source:
            self.ani.event_source.stop()