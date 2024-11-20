import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
from datetime import datetime, timedelta
from utilities import *
import matplotlib.patches as mpatches

# 生成示例数据
filePath = r'c:\Document\sc2024\filtered_ecg_with_snr.csv'
ecg, ap, rr, quality = loadRtDatawithRR(filePath)
x = range(len(ecg))  # 采样点序列
sampling_rate = 250  # 假设采样率为 250 Hz

# 找到连续 "bad" 的区间
bad_intervals = []
is_bad = False
start_index = 0

for i in range(len(quality)):
    if quality[i] == "Bad" and not is_bad:
        is_bad = True
        start_index = i
    elif quality[i] != "Bad" and is_bad:
        is_bad = False
        bad_intervals.append((start_index, i))

if is_bad:
    bad_intervals.append((start_index, len(quality)))

# 将采样点转换为时间
start_time = datetime(2024, 11, 20, 0, 0, 0)  # 假设数据采集起始时间
time_stamps = [start_time + timedelta(seconds=i / sampling_rate) for i in x]

# 初始显示范围
initial_range = 1000  # 初始显示 1000 个采样点
start_idx = 0

# 创建图形和两个子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
plt.subplots_adjust(hspace=0.5, bottom=0.4)  # 调整子图间距和底部空间

# 绘制两条信号
line1, = ax1.plot(time_stamps, ecg, lw=2)
line2, = ax2.plot(time_stamps, ap, lw=2)

# 在图上标记 "bad" 区间
for start, end in bad_intervals:
    ax1.axvspan(time_stamps[start], time_stamps[end], color='red', alpha=0.3)
    ax2.axvspan(time_stamps[start], time_stamps[end], color='red', alpha=0.3)

bad_patch = mpatches.Patch(color='red', alpha=0.3, label='"Bad" Interval')
fig.legend(handles=[bad_patch], loc='upper right')

# 设置初始显示范围
ax1.set_xlim(time_stamps[start_idx], time_stamps[start_idx + initial_range])
ax2.set_xlim(time_stamps[start_idx], time_stamps[start_idx + initial_range])

# 设置时间格式化
from matplotlib.dates import DateFormatter
time_formatter = DateFormatter('%H:%M:%S')  # 设置时间格式为 HH:MM:SS
ax1.xaxis.set_major_formatter(time_formatter)
ax2.xaxis.set_major_formatter(time_formatter)

# 调整滑动条的位置
slider_height = 0.03  # 滑动条高度
ax_slider1 = plt.axes([0.15, 0.5, 0.7, slider_height], facecolor='lightgoldenrodyellow')  # 紧贴第一张图下方
slider1 = Slider(ax_slider1, 'Position 1', 0, len(time_stamps) - initial_range, valinit=0)

ax_slider2 = plt.axes([0.15, 0.1, 0.7, slider_height], facecolor='lightblue')  # 紧贴第二张图下方
slider2 = Slider(ax_slider2, 'Position 2', 0, len(time_stamps) - initial_range, valinit=0)

# 缩放按钮的位置
ax_zoom_in_x = plt.axes([0.7, 0.85, 0.1, 0.05])  # 放大X轴按钮
ax_zoom_out_x = plt.axes([0.7, 0.78, 0.1, 0.05])  # 缩小X轴按钮
ax_zoom_in_y1 = plt.axes([0.1, 0.85, 0.1, 0.05])  # 第一个子图放大Y轴按钮
ax_zoom_out_y1 = plt.axes([0.1, 0.78, 0.1, 0.05])  # 第一个子图缩小Y轴按钮
ax_zoom_in_y2 = plt.axes([0.1, 0.65, 0.1, 0.05])  # 第二个子图放大Y轴按钮
ax_zoom_out_y2 = plt.axes([0.1, 0.58, 0.1, 0.05])  # 第二个子图缩小Y轴按钮

button_zoom_in_x = Button(ax_zoom_in_x, 'Zoom In X')
button_zoom_out_x = Button(ax_zoom_out_x, 'Zoom Out X')
button_zoom_in_y1 = Button(ax_zoom_in_y1, 'Zoom In Y1')
button_zoom_out_y1 = Button(ax_zoom_out_y1, 'Zoom Out Y1')
button_zoom_in_y2 = Button(ax_zoom_in_y2, 'Zoom In Y2')
button_zoom_out_y2 = Button(ax_zoom_out_y2, 'Zoom Out Y2')

# 缩放比例
zoom_factor = 0.8


def zoom_in_x(event):
    global initial_range
    initial_range = int(initial_range * zoom_factor)
    if initial_range < 10:  # 防止范围太小
        initial_range = 10
    update_range()


def zoom_out_x(event):
    global initial_range
    initial_range = int(initial_range / zoom_factor)
    if initial_range > len(time_stamps):  # 防止范围太大
        initial_range = len(time_stamps)
    update_range()


def zoom_in_y(ax):
    ymin, ymax = ax.get_ylim()
    ycenter = (ymin + ymax) / 2
    yrange = (ymax - ymin) * zoom_factor / 2
    ax.set_ylim(ycenter - yrange, ycenter + yrange)
    fig.canvas.draw_idle()


def zoom_out_y(ax):
    ymin, ymax = ax.get_ylim()
    ycenter = (ymin + ymax) / 2
    yrange = (ymax - ymin) / zoom_factor / 2
    ax.set_ylim(ycenter - yrange, ycenter + yrange)
    fig.canvas.draw_idle()


def update_range():
    pos1 = int(slider1.val)
    pos2 = int(slider2.val)
    ax1.set_xlim(time_stamps[pos1], time_stamps[min(pos1 + initial_range, len(time_stamps) - 1)])
    ax2.set_xlim(time_stamps[pos2], time_stamps[min(pos2 + initial_range, len(time_stamps) - 1)])
    fig.canvas.draw_idle()


# 为按钮绑定事件
button_zoom_in_x.on_clicked(zoom_in_x)
button_zoom_out_x.on_clicked(zoom_out_x)
button_zoom_in_y1.on_clicked(lambda event: zoom_in_y(ax1))
button_zoom_out_y1.on_clicked(lambda event: zoom_out_y(ax1))
button_zoom_in_y2.on_clicked(lambda event: zoom_in_y(ax2))
button_zoom_out_y2.on_clicked(lambda event: zoom_out_y(ax2))


# 滑动条1的事件处理函数
def update1(val):
    pos1 = int(slider1.val)  # 获取滑动条1的值
    ax1.set_xlim(time_stamps[pos1], time_stamps[min(pos1 + initial_range, len(time_stamps) - 1)])
    slider1.valtext.set_text(time_stamps[pos1].strftime('%H:%M:%S'))  # 更新滑动条的显示值
    fig.canvas.draw_idle()


# 滑动条2的事件处理函数
def update2(val):
    pos2 = int(slider2.val)  # 获取滑动条2的值
    ax2.set_xlim(time_stamps[pos2], time_stamps[min(pos2 + initial_range, len(time_stamps) - 1)])
    slider2.valtext.set_text(time_stamps[pos2].strftime('%H:%M:%S'))  # 更新滑动条的显示值
    fig.canvas.draw_idle()


# 绑定滑动条事件
slider1.on_changed(update1)
slider2.on_changed(update2)

plt.show()
