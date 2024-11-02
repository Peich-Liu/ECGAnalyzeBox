# data_loader_handlers.py
from utilities import *
import tkinter as tk
from tkinter import messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np

# 用于存储矩形框状态
rect1 = None
rect2 = None
is_drawing = False
is_dragging = False
start_x = None
offset_x = 0

def load_ecg_data(cba_instance, patient_id_entry, start_entry, end_entry, canvas_frame_ecg):
    try:
        patient_id = patient_id_entry.get()
        start = int(start_entry.get())
        end = int(end_entry.get())
        data_directory = r'C:\Document\sc2024\fantasia-database-1.0.0/'

        # 加载ECG数据
        records, _ = readPatientRecords2(patient_id, data_directory)
        if not records:
            messagebox.showerror("Error", "Failed to read records")
            return
        concatenated_signal = concatenateECG(records, start, end)
        cba_instance.update_ecg_signal(concatenated_signal, 250)  # 设置采样率

        # 在两个子图中绘制ECG和AP信号
        plot_signals(cba_instance, canvas_frame_ecg, concatenated_signal, cba_instance.ap_signal)

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid start and end sample values")

def load_ap_data(cba_instance, patient_id_entry, start_entry, end_entry, canvas_frame_ecg):
    try:
        patient_id = patient_id_entry.get()
        start = int(start_entry.get())
        end = int(end_entry.get())
        data_directory = r'C:\Document\sc2024\fantasia-database-1.0.0/'

        # 加载AP数据
        records, _ = readPatientRecords2(patient_id, data_directory)
        if not records:
            messagebox.showerror("Error", "Failed to read records")
            return
        concatenated_signal = concatenateAP(records, start, end)
        cba_instance.update_ap_signal(concatenated_signal, 250)  # 设置采样率

        # 在两个子图中绘制ECG和AP信号
        plot_signals(cba_instance, canvas_frame_ecg, cba_instance.ecg_signal, concatenated_signal)

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid start and end sample values")

def plot_signals(cba_instance, canvas_frame, ecg_signal, ap_signal):
    figure = Figure(figsize=(10, 6), dpi=100)
    ax1 = figure.add_subplot(211)
    ax2 = figure.add_subplot(212, sharex=ax1)
    
    # 绘制ECG和AP信号
    ax1.plot(np.arange(len(ecg_signal)), ecg_signal, label="ECG Signal")
    ax1.set_ylabel("Amplitude")
    ax1.legend()

    ax2.plot(np.arange(len(ap_signal)), ap_signal, label="AP Signal", color="orange")
    ax2.set_xlabel("Time (samples)")
    ax2.set_ylabel("Amplitude")
    ax2.legend()
    
    # 将图表嵌入Tkinter Canvas
    canvas = FigureCanvasTkAgg(figure, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # 添加同步矩形框选择功能
    canvas.mpl_connect("button_press_event", lambda event: on_press(event, ax1, ax2))
    canvas.mpl_connect("motion_notify_event", lambda event: on_drag(event, ax1, ax2))
    canvas.mpl_connect("button_release_event", lambda event: on_release(event, ax1, ax2, ecg_signal, ap_signal))

# 鼠标按下事件
def on_press(event, ax1, ax2):
    global start_x, rect1, rect2, is_drawing, is_dragging, offset_x
    if event.inaxes and not is_drawing:
        start_x = event.xdata
        is_drawing = True
        rect1 = Rectangle((start_x, ax1.get_ylim()[0]), 0, np.diff(ax1.get_ylim())[0], edgecolor='r', facecolor='none')
        rect2 = Rectangle((start_x, ax2.get_ylim()[0]), 0, np.diff(ax2.get_ylim())[0], edgecolor='r', facecolor='none')
        ax1.add_patch(rect1)
        ax2.add_patch(rect2)

# 鼠标拖动事件
def on_drag(event, ax1, ax2):
    global rect1, rect2
    if is_drawing and event.inaxes and rect1 is not None and rect2 is not None:
        width = event.xdata - start_x
        rect1.set_width(width)
        rect2.set_width(width)
        ax1.figure.canvas.draw()
        ax2.figure.canvas.draw()

# 鼠标松开事件
def on_release(event, ax1, ax2, ecg_signal, ap_signal):
    global is_drawing, rect1, rect2
    if is_drawing and rect1 is not None and rect2 is not None:
        is_drawing = False
        x_min = int(rect1.get_x())
        x_max = int(rect1.get_x() + rect1.get_width())
        
        # 计算时间区间内ECG和AP信号的平均值
        ecg_mean = np.mean(ecg_signal[x_min:x_max]) if x_max > x_min else 0
        ap_mean = np.mean(ap_signal[x_min:x_max]) if x_max > x_min else 0
        
        # 弹出消息框显示平均值
        messagebox.showinfo("平均值", f"时间区间内ECG信号平均值: {ecg_mean:.2f}\n时间区间内AP信号平均值: {ap_mean:.2f}")

# 在 Load Data Page 上集成
def create_load_data_page(notebook, load_ecg_command, load_ap_command,  load_eeg_command, vis_data):
    load_data_page = ttk.Frame(notebook)
    notebook.add(load_data_page, text="Load Data Page")

    ecg_frame = ttk.Frame(load_data_page)
    ecg_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    ttk.Label(ecg_frame, text="Patient ID:").grid(row=0, column=0, padx=5, pady=5)
    ecg_patient_id_entry = ttk.Entry(ecg_frame)
    ecg_patient_id_entry.grid(row=0, column=1, padx=5, pady=5)
    ecg_patient_id_entry.insert(0, "f2o01")

    ttk.Label(ecg_frame, text="Start Sample:").grid(row=0, column=2, padx=5, pady=5)
    ecg_start_entry = ttk.Entry(ecg_frame)
    ecg_start_entry.grid(row=0, column=3, padx=5, pady=5)
    ecg_start_entry.insert(0, "0")

    ttk.Label(ecg_frame, text="End Sample:").grid(row=0, column=4, padx=5, pady=5)
    ecg_end_entry = ttk.Entry(ecg_frame)
    ecg_end_entry.grid(row=0, column=5, padx=5, pady=5)
    ecg_end_entry.insert(0, "50000")

    load_ecg_button = ttk.Button(ecg_frame, text="Load ECG Data", command=load_ecg_command)
    load_ecg_button.grid(row=0, column=6, padx=5, pady=5)

    load_ap_button = ttk.Button(ecg_frame, text="Load AP Data", command=load_ap_command)
    load_ap_button.grid(row=0, column=7, padx=5, pady=5)

    canvas_frame_ecg = ttk.Frame(load_data_page)
    canvas_frame_ecg.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    return load_data_page
