import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# 创建 Tkinter 主窗口
root = tk.Tk()
root.title("Tkinter 与 Matplotlib 双子图同步选择")

# 创建坐标显示标签和平均值显示标签
coord_label = tk.Label(root, text="框的坐标将在这里显示")
coord_label.pack()
mean_label = tk.Label(root, text="选定时间区间内信号的平均值将在这里显示")
mean_label.pack()

# 创建 Matplotlib Figure 和双子图
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
x = np.linspace(0, 4, 100)
y1 = np.sin(x * np.pi) + np.random.normal(0, 0.1, size=x.shape)  # 示例数据1
y2 = np.cos(x * np.pi) + np.random.normal(0, 0.1, size=x.shape)  # 示例数据2
ax1.plot(x, y1, label='Signal 1')
ax2.plot(x, y2, label='Signal 2')
ax1.set_title("Signal 1")
ax2.set_title("Signal 2")
fig.suptitle("按住鼠标拖动以选择时间区间，双子图同步更新")

# 将 Matplotlib 图表嵌入 Tkinter Canvas
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# 添加 Matplotlib 的工具栏，以支持缩放和平移功能
toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas.get_tk_widget().pack()

# 全局变量
start_x = None                 # 矩形框的起始点
rect1 = None                   # 子图1的矩形对象
rect2 = None                   # 子图2的矩形对象
is_dragging = False            # 标记是否处于拖动状态
is_drawing = False             # 标记是否处于绘制矩形框状态
offset_x = 0                   # 拖动时的偏移量

# 更新坐标显示标签和计算时间区间内数据的平均值
def update_labels():
    if rect1 is not None and rect2 is not None:
        x_min, x_max = rect1.get_x(), rect1.get_x() + rect1.get_width()
        coord_label.config(text=f"时间区间: 开始={x_min:.2f}, 结束={x_max:.2f}")
        
        # 提取时间区间内的数据并计算平均值
        mask = (x >= x_min) & (x <= x_max)
        data_in_time_range_1 = y1[mask]
        data_in_time_range_2 = y2[mask]
        if data_in_time_range_1.size > 0 and data_in_time_range_2.size > 0:
            mean_value_1 = np.mean(data_in_time_range_1)
            mean_value_2 = np.mean(data_in_time_range_2)
            mean_label.config(text=f"选定时间区间内信号1的平均值: {mean_value_1:.2f}, 信号2的平均值: {mean_value_2:.2f}")
        else:
            mean_label.config(text="选定时间区间内无数据")

# 鼠标按下事件 - 创建或准备拖动 Rectangle 对象
def on_press(event):
    global start_x, rect1, rect2, is_dragging, is_drawing, offset_x
    if event.inaxes and toolbar.mode == '':
        # 如果已经存在矩形框，检查是否点击在框内以准备拖动
        if rect1 is not None and rect2 is not None:
            contains, _ = rect1.contains(event)
            if contains:
                is_dragging = True
                offset_x = event.xdata - rect1.get_x()
        else:
            # 如果没有矩形框，开始绘制新的矩形框
            is_drawing = True
            start_x = event.xdata
            # 创建两个矩形框并将 y 轴的范围设置为全局
            rect1 = Rectangle((start_x, ax1.get_ylim()[0]), 0, np.diff(ax1.get_ylim())[0],
                              linewidth=1, edgecolor='r', facecolor='none')
            rect2 = Rectangle((start_x, ax2.get_ylim()[0]), 0, np.diff(ax2.get_ylim())[0],
                              linewidth=1, edgecolor='r', facecolor='none')
            ax1.add_patch(rect1)
            ax2.add_patch(rect2)

# 鼠标拖动事件 - 更新矩形框的大小或位置
def on_drag(event):
    global rect1, rect2
    if event.inaxes and toolbar.mode == '':
        if is_drawing and rect1 is not None and rect2 is not None:
            # 调整矩形的宽度以覆盖选择的时间区间
            width = event.xdata - start_x
            rect1.set_width(width)
            rect2.set_width(width)
        elif is_dragging and rect1 is not None and rect2 is not None:
            # 拖动矩形框
            new_x = event.xdata - offset_x
            rect1.set_x(new_x)
            rect2.set_x(new_x)
        fig.canvas.draw()
        update_labels()  # 更新标签显示框的坐标和平均值

# 鼠标松开事件 - 完成拖动或创建矩形
def on_release(event):
    global is_dragging, is_drawing, rect1, rect2
    if is_dragging:
        is_dragging = False
        if rect1 is not None and rect2 is not None:
            update_labels()  # 更新标签显示框的坐标和平均值
    elif is_drawing:
        is_drawing = False
        if rect1 is not None and rect2 is not None:
            update_labels()  # 更新标签显示框的坐标和平均值

# 绑定鼠标事件
canvas.mpl_connect("button_press_event", on_press)
canvas.mpl_connect("motion_notify_event", on_drag)
canvas.mpl_connect("button_release_event", on_release)

# 运行 Tkinter 主循环
root.mainloop()
