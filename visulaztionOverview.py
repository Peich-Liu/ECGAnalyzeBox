'''
Develop here later after having the signal
'''
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np

class InteractivePlot:
    def __init__(self, canvas_frame):
        self.canvas_frame = canvas_frame

        # 初始化交互属性
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212, sharex=self.ax1)
        self.rect1 = None
        self.rect2 = None
        self.is_drawing = False
        self.start_x = None
        self.selection_enabled = False  # 初始状态为禁用交互

        # 初始化图形和工具栏
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.add_toolbar()

        # 绑定事件
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("button_release_event", self.on_release)

    def initialize_plot(self):
        """初始化或重置图形和工具栏"""
        # 如果已经存在图表和工具栏，先清除
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
            if self.toolbar:
                self.toolbar.pack_forget()

        # 创建新的图形和子图
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212, sharex=self.ax1)

        # 设置canvas和工具栏
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        # 绑定事件
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_drag)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        
    def add_toolbar(self):
        """添加导航工具栏"""
        toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def plot_signals(self, ecg_data, ap_data):
        """更新绘图内容"""
        # 清除现有内容
        self.ax1.clear()
        self.ax2.clear()

        # 绘制新的ECG和AP信号
        x = np.arange(len(ecg_data))
        self.ax1.plot(x, ecg_data, label="ECG Signal")
        self.ax1.set_ylabel("ECG Amplitude")
        self.ax1.legend()

        self.ax2.plot(x, ap_data, label="AP Signal", color="orange")
        self.ax2.set_xlabel("Time (samples)")
        self.ax2.set_ylabel("AP Amplitude")
        self.ax2.legend()

        # 更新画布
        self.canvas.draw()

    def toggle_selection(self):
        """切换选择模式的启用/禁用"""
        self.selection_enabled = not self.selection_enabled
        return self.selection_enabled  # 返回当前交互状态

    def on_press(self, event):
        """鼠标按下事件处理"""
        if not self.selection_enabled or event.inaxes not in [self.ax1, self.ax2]:  # 确保事件在绘图区内且交互被启用
            return
        self.is_drawing = True
        self.start_x = event.xdata
        # 创建同步的矩形框
        self.rect1 = Rectangle((self.start_x, self.ax1.get_ylim()[0]), 0, np.diff(self.ax1.get_ylim())[0],
                               edgecolor='r', facecolor='none')
        self.rect2 = Rectangle((self.start_x, self.ax2.get_ylim()[0]), 0, np.diff(self.ax2.get_ylim())[0],
                               edgecolor='r', facecolor='none')
        self.ax1.add_patch(self.rect1)
        self.ax2.add_patch(self.rect2)

    def on_drag(self, event):
        """鼠标拖动事件处理"""
        if not self.is_drawing or not self.selection_enabled or event.inaxes not in [self.ax1, self.ax2]:
            return
        width = event.xdata - self.start_x
        self.rect1.set_width(width)
        self.rect2.set_width(width)
        self.canvas.draw()

    def on_release(self, event):
        """鼠标松开事件处理"""
        if not self.is_drawing or not self.selection_enabled:
            return
        self.is_drawing = False
        x_min = int(self.rect1.get_x())
        x_max = int(self.rect1.get_x() + self.rect1.get_width())

        return x_min, x_max
        
        # # 输出起点和终点
        # print(f"起点: {x_min}, 终点: {x_max}")

def vis_data(cba_instance, plot_instance):
    plot_instance.plot_signals(cba_instance.ecg_signal, cba_instance.ap_signal)

def updateInteract(plot_instance):
    plot_instance.toggle_selection()