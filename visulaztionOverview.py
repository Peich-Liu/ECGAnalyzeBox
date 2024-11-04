'''
Develop here later after having the signal
'''
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np
from utilities import *
import time

class InteractivePlot:
    def __init__(self, observer):
        # self.canvas_frame = canvas_frame

        # 初始化交互属性
        # self.fig = Figure(figsize=(10, 6), dpi=100)
        self.observer = observer
        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212, sharex=self.ax1)
        self.rect1 = None
        self.rect2 = None
        self.is_drawing = False
        self.is_dragging = False
        self.selected_index = None  # 当前被选中的矩形索引
        self.start_x = None
        self.create_mode = False  # 控制是否允许创建新矩形
        self.selection_ranges = {}  # 存储选择的范围
        self.current_index = 0  # 索引计数器
        self.last_update_time = 0  # 初始化最后一次更新的时间戳

        self.last_press_time = 0  # 上次鼠标按下事件的时间戳
        self.debounce_interval = 0.5  # 去抖时间间隔（秒）

    def add_toolbar(self, canvas, canvas_frame):
        """添加导航工具栏"""
        self.toolbar = NavigationToolbar2Tk(canvas, canvas_frame)
        self.toolbar.update()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def plot_signals(self, ecg_data, ap_data):
        """更新绘图内容"""
        self.ax1.clear()
        self.ax2.clear()
        x = np.arange(len(ecg_data))
        self.ax1.plot(x, ecg_data, label="ECG Signal")
        self.ax1.set_ylabel("ECG Amplitude")
        self.ax1.legend()
        self.ax2.plot(x, ap_data, label="AP Signal", color="orange")
        self.ax2.set_xlabel("Time (samples)")
        self.ax2.set_ylabel("AP Amplitude")
        self.ax2.legend()
        self.fig.canvas.draw()

    def toggle_create_mode(self):
        """切换创建模式"""
        self.create_mode = True
        return self.create_mode 
    
    def toggle_drag_mode(self):
        """切换拖动模式"""
        self.create_mode = False
        return self.create_mode  

    def on_press(self, event):
        """鼠标按下事件处理"""
        if event.inaxes not in [self.ax1, self.ax2]:  # 确保事件在绘图区内
            return
        current_time = time.time()
        if current_time - self.last_press_time < self.debounce_interval:
            return
        self.last_press_time = current_time

        if self.create_mode:
            self.is_drawing = True
            self.start_x = event.xdata
            rect1 = Rectangle((self.start_x, self.ax1.get_ylim()[0]), 0, np.diff(self.ax1.get_ylim())[0],
                              edgecolor='r', facecolor='none')
            rect2 = Rectangle((self.start_x, self.ax2.get_ylim()[0]), 0, np.diff(self.ax2.get_ylim())[0],
                              edgecolor='r', facecolor='none')
            self.ax1.add_patch(rect1)
            self.ax2.add_patch(rect2)
            self.selection_ranges[self.current_index] = (rect1, rect2)
            self.selected_index = self.current_index
            self.current_index += 1

        else:  
            for index, (rect1, rect2) in self.selection_ranges.items():
                print("self.selection_ranges = ", self.selection_ranges)
                print("rect1,rect2 = ", rect1, rect2)
                if rect1.contains(event)[0] or rect2.contains(event)[0]:  # 如果在框框内
                    self.is_dragging = True
                    self.selected_index = index
                    self.start_x = event.xdata - rect1.get_x()  # 记录点击位置相对于矩形的偏移
                    return

    def on_drag(self, event):
        """鼠标拖动事件处理"""
        if event.inaxes not in [self.ax1, self.ax2]:
            return

        if self.is_drawing and self.create_mode:
            # 正在创建新的矩形
            width = event.xdata - self.start_x
            rect1, rect2 = self.selection_ranges[self.selected_index]
            rect1.set_width(width)
            rect2.set_width(width)
            self.fig.canvas.draw()

        elif self.is_dragging and not self.create_mode:
            # 正在拖动已有的矩形
            rect1, rect2 = self.selection_ranges[self.selected_index]
            new_x = event.xdata - self.start_x  # 根据偏移计算新的位置
            rect1.set_x(new_x)
            rect2.set_x(new_x)
            self.fig.canvas.draw()

    def on_release(self, event):
        """鼠标松开事件处理"""
        if self.is_drawing:
            # 完成矩形创建
            self.is_drawing = False

        elif self.is_dragging:
            # 完成矩形拖动
            self.is_dragging = False
            self.selected_index = None
        
        # 获取更新的选择范围并通知所有订阅者
        updated_ranges = self.get_selection_ranges()
        # self.selection_ranges = updated_ranges
        self.observer.notify(updated_ranges)  # 通知观察者


    def get_selection_ranges(self):
        """返回所有选择的范围信息，确保 rect1 和 rect2 是 Rectangle 对象"""
        selection_ranges = {}
        for index, (rect1, rect2) in self.selection_ranges.items():
            if isinstance(rect1, Rectangle) and isinstance(rect2, Rectangle):
                selection_ranges[index] = (
                    int(rect1.get_x()), 
                    int(rect1.get_x() + rect1.get_width())
                )
        return selection_ranges

def vis_data(cba_instance, plot_instance):
    plot_instance.plot_signals(cba_instance.ecg_signal, cba_instance.ap_signal)

def updateInteract(plot_instance):
    plot_instance.toggle_create_mode()