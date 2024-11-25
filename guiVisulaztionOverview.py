import numpy as np
import time
import numpy as np
import matplotlib.dates as mdates
import tkinter as tk

from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
from matplotlib.widgets import Slider
from utilities import *


class InteractivePlot:
    def __init__(self, observer):
        # self.canvas_frame = canvas_frame

        # 初始化交互属性
        self.fig = Figure(figsize=(20, 6), dpi=100)
        self.observer = observer
        # self.fig = Figure()
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212, sharex=self.ax1)
        # self.ax3 = self.fig.add_subplot(313, sharex=self.ax1)
        
        self.rect1 = None
        self.rect2 = None
        self.is_drawing = False
        self.is_dragging = False
        self.selected_index = None  # 当前被选中的矩形索引
        self.analyze_window_index = 0
        self.start_x = None

        self.create_mode = False  # 控制是否允许创建新矩形
        self.drag_mode = False  # 控制是否允许拖动新矩形
        self.del_mode = False
        
        
        self.selected_start = {}
        self.selected_end = {}
        self.selection_ranges = {}  # 存储选择的范围
        self.current_index = 0  # 索引计数器
        self.last_update_time = 0  # 初始化最后一次更新的时间戳

        self.last_press_time = 0  # 上次鼠标按下事件的时间戳
        self.debounce_interval = 0.5  # 去抖时间间隔（秒）
        


    def plot_signals(self, ecg_data, ap_data, quality_data, start_time="00:00:00", sample_interval=0.004):
        """更新绘图内容"""
        self.ax1.clear()
        self.ax2.clear()
        # self.ax3.clear()
        
        # 将起始时间转化为 datetime 对象
        start_datetime = datetime.strptime(start_time, "%H:%M:%S")
        
        # 创建 x 轴的时间列表
        time_points = [start_datetime + timedelta(seconds=i * sample_interval) for i in range(len(ecg_data))]
        
        # 绘制 ECG 数据
        self.ax1.plot(time_points, ecg_data, label="ECG Signal")
        self.ax1.set_ylabel("ECG Amplitude(mV)")
        self.ax1.legend()
        
        # 绘制 AP 数据
        self.ax2.plot(time_points, ap_data, label="AP Signal", color="orange")
        self.ax2.set_xlabel("Time (HH:MM:SS)")
        self.ax2.set_ylabel("AP Amplitude (mmHg)")
        self.ax2.legend()

        # 如果提供了质量数据，标记 "bad" 区间
        if quality_data is not None:
            is_bad = False
            start_index = 0
            for i, quality in enumerate(quality_data):
                if quality == "Bad" and not is_bad:
                    is_bad = True
                    start_index = i
                elif quality == "Good" and is_bad:
                    is_bad = False
                    # 绘制 "bad" 区间
                    self.ax1.axvspan(time_points[start_index], time_points[i], color='grey', alpha=0.3)
                    self.ax2.axvspan(time_points[start_index], time_points[i], color='grey', alpha=0.3)
            
            # 如果最后一个点是 "Bad"，处理未关闭的区间
            if is_bad:
                self.ax1.axvspan(time_points[start_index], time_points[-1], color='grey', alpha=0.3)
                self.ax2.axvspan(time_points[start_index], time_points[-1], color='grey', alpha=0.3)
    
        
        # 设置 x 轴格式
        self.ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        self.ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        
        # 自动格式化 x 轴时间刻度
        self.ax1.xaxis.set_major_locator(mdates.SecondLocator(interval=1000))  # 设置为每 10 秒一个刻度
        self.ax2.xaxis.set_major_locator(mdates.SecondLocator(interval=1000))  # 设置为每 10 秒一个刻度

        # 绘图更新
        self.fig.autofmt_xdate()  # 自动旋转日期标签
        self.fig.canvas.draw()

    def toggle_create_mode(self):
        """切换创建模式"""
        self.create_mode = True
        self.drag_mode = False
        self.del_mode = False
        
        return self.create_mode
    
    def toggle_drag_mode(self):
        """切换拖动模式"""
        self.create_mode = False
        self.drag_mode = True
        self.del_mode = False
        
        return self.create_mode  
    def toggle_plot_mode(self):
        """切换plot模式"""
        self.create_mode = False
        self.drag_mode = False
        self.del_mode = False
        
        return
        # return self.create_mode  
    def toggle_del_mode(self):
        self.del_mode = True
        self.create_mode = False
        self.drag_mode = False
        
        return 

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

            # selected_time = mdates.num2date(event.xdata)
            # self.selected_start[self.current_index] = selected_time.strftime("%H:%M:%S")
            selected_time = event.xdata
            self.selected_start[self.current_index] = selected_time
            
            # print("self.selected_start[self.current_index]",self.selected_start[self.current_index])

            rect1 = Rectangle((self.start_x, self.ax1.get_ylim()[0]), 0, np.diff(self.ax1.get_ylim())[0],
                              edgecolor='r', facecolor='none')
            rect2 = Rectangle((self.start_x, self.ax2.get_ylim()[0]), 0, np.diff(self.ax2.get_ylim())[0],
                              edgecolor='r', facecolor='none')
            self.ax1.add_patch(rect1)
            self.ax2.add_patch(rect2)
            self.selection_ranges[self.current_index] = (rect1, rect2)
            self.selected_index = self.current_index

        elif self.drag_mode:  
            for index, (rect1, rect2) in self.selection_ranges.items():
                print("event",event.xdata)
                if rect1.contains(event)[0] or rect2.contains(event)[0]:  # 如果在框框内
                    self.is_dragging = True
                    self.selected_index = index
                    self.start_x = event.xdata - rect1.get_x()  # 记录点击位置相对于矩形的偏移
                    selected_time = mdates.num2date(event.xdata)
                    self.selected_start[self.current_index] = selected_time.strftime("%H:%M:%S")
                    # print("self.selected_start[self.current_index]",self.selected_start[self.current_index])
                    return
        
        elif self.del_mode:
            pass
        
        else:
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


        elif self.is_dragging and not self.create_mode and self.drag_mode:
            # 正在拖动已有的矩形
            rect1, rect2 = self.selection_ranges[self.selected_index]
            new_x = event.xdata - self.start_x  # 根据偏移计算新的位置
            rect1.set_x(new_x)
            rect2.set_x(new_x)
            self.fig.canvas.draw()

    def on_release(self, event):
        """鼠标松开事件处理"""

        if self.is_drawing and self.create_mode:
            # 完成矩形创建
            self.is_drawing = False
            # selected_time = mdates.num2date(event.xdata)# update end
            selected_time = event.xdata# update end
            # self.selected_end[self.current_index] = selected_time.strftime("%H:%M:%S")
            self.selected_end[self.current_index] = selected_time
            
            self.current_index += 1
            

        elif self.is_dragging and self.drag_mode:
            # 完成矩形拖动
            self.is_dragging = False
            self.selected_index = None
            # selected_time = mdates.num2date(event.xdata)# update end
            selected_time = event.xdata# update end
            
            # self.selected_end[self.current_index] = selected_time.strftime("%H:%M:%S")
            self.selected_end[self.current_index] = selected_time
            
        
        # 获取更新的选择范围并通知所有订阅者
        updated_ranges = self.get_selection_ranges()
        # self.selection_ranges = updated_ranges
        self.observer.notify(updated_ranges)  # 通知观察者

    def get_selection_ranges(self):
        # selection_ranges = {}
        # for index, (start, end) in zip(self.selected_start.items(), self.selected_end.items()):
        # for index, start in self.selected_start.items():
        #     # print("self.selected_start",self.selected_start)
        #     end = self.selected_end.get(index)
            # 将时间格式化为 HH:MM:SS
            # self.selection_ranges[index] = (start, end)
            # print("selection_ranges1",self.selection_ranges)=
        return self.selection_ranges
    # def mark_noise_regions(self, time_points, quality_data):
    def mark_noise_regions(self, quality_data):
        """根据 quality 列标记噪音区间"""
        time_points = range(len(quality_data))
        noise_intervals = []
        is_bad = False
        start_index = 0

        # 遍历 quality_data 列
        for i, quality in enumerate(quality_data):
            if quality == "Bad" and not is_bad:
                is_bad = True
                start_index = i
            elif quality == "Good" and is_bad:
                is_bad = False
                noise_intervals.append((start_index, i))
        
        # 如果文件以 "Bad" 结尾，处理最后的区间
        if is_bad:
            noise_intervals.append((start_index, len(quality_data)))

        # 在图表上标记噪音区间
        for start, end in noise_intervals:
            self.ax1.axvspan(time_points[start], time_points[end], color='grey', alpha=0.3)
            self.ax2.axvspan(time_points[start], time_points[end], color='grey', alpha=0.3)
        # 使用 blit 刷新
        self.fig.canvas.draw_idle()  # `draw_idle` 优化性能
        self.fig.canvas.flush_events()  # 确保更新发生

def vis_data(cba_instance, plot_instance):
    plot_instance.plot_signals(cba_instance.ecg_signal, cba_instance.ap_signal)

def updateInteract(plot_instance):
    plot_instance.toggle_create_mode()







# class AnalyzerPlot:
#     def __init__(self, gui_window):
#         self.guiWindow = gui_window
#         self.range_max = None
#         self.range_min = None
#         self.max_points = []  # 保存最大值的点
#         self.min_points = []  # 保存最小值的点
#         self.time_series = []  # 保存时间序列的 x 值
#         self.hrv_values = []   # 保存 HRV 值
#         self.parameters = None

#         # 初始化图形和Canvas
#         self.fig = Figure(figsize=(5, 4), dpi=100)
#         self.ax = self.fig.add_subplot(111)
#         self.canvas = FigureCanvasTkAgg(self.fig, master=self.guiWindow.canvas_hrv_frame)
#         self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

#     def hrv_analysis(self, range):
#         print("hrv_analysis, range[0][0]:",range)
#         # 获取采样点并计算相对时间
#         sample_point = range[0][0]
#         time_delta = timedelta(seconds=sample_point / 250)  # 将采样点转换为时间差

#         # 将起始时间设为 `datetime` 对象
#         self.start_time = datetime.strptime("00:00:00", "%H:%M:%S")

#         # 计算该采样点的实际时间
#         x = self.start_time + time_delta
#         hrv = float(self.parameters['0']['ECG']['Heart Rate'])

#         # 将新的时间点和 HRV 值添加到序列中
#         self.time_series.append(x)
#         self.hrv_values.append(hrv)

#         # 清除旧图像
#         self.ax.clear()

#         print("self.hrv_values",x)

#         # 绘制已经存在的时间序列数据
#         self.ax.plot(self.time_series, self.hrv_values, '-o', label="Heart Rate Time Series")

#         # 单独标记最新的点
#         self.ax.plot(x, hrv, 'ro', label="Current Point")

#         # 设置 x 轴格式为 HH:MM:SS
#         self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
#         self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())

#         # 设置轴标签和图例
#         self.ax.set_xlabel("Time (HH:MM:SS)")
#         self.ax.set_ylabel("Heart Rate (BPM)")
#         self.ax.legend()
        
#         self.canvas.draw()  # 刷新 Canvas 显示最新的图形

#         getFigure(self.fig, self.guiWindow.canvas_hrv_frame)

#     def update_range_maxmin(self, range):
#         # 获取 range[0] 的最大值和最小值
#         print("update_range_maxmin", range)
#         range_max = max(range[0])
#         range_min = min(range[0])

#         # 初始化最大值和最小值
#         if self.range_max is None or self.range_min is None:
#             self.range_max = range_max
#             self.range_min = range_min
#             self.hrv_analysis(range)  # 更新图形
#             return
#         # 更新最大值
#         if range_max > self.range_max:
#             print(f"最大值已更新：从 {self.range_max} 到 {range_max}")
#             self.range_max = range_max
#             self.max_points.append((range[0][1], range_max))  # 添加新的最大值点
#             self.hrv_analysis(range)  # 更新图形

#         # 更新最小值
#         if range_min < self.range_min:
#             print(f"最小值已更新：从 {self.range_min} 到 {range_min}")
#             self.range_min = range_min
#             self.min_points.append((range[0][0], range_min))  # 添加新的最小值点
#             self.hrv_analysis(range)  # 更新图形