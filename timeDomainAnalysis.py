# signal_processing_handlers.py
import tkinter as tk
from tkinter import messagebox
from utilities import *

def calculateEcgTimeDomainValue(cba_instance, fs,properties_frame, sd_canvas_frame):

    if cba_instance.ecg_signal is None:
        messagebox.showerror("Error", "No signal loaded to filter. Please load data first.")
        return
    try:
        # Calculate and display signal properties
        properties, sdFigure = calculateEcgSignalProperties(cba_instance.ecg_signal, fs) # The algorithm of the time domain anaylsis is here
        displaySignalProperties(properties, properties_frame)
        getFigure(sdFigure, sd_canvas_frame)
    except ValueError:
        messagebox.showerror("Input Error", "No signal loaded to filter. Please load data first.")

def calculateApTimeDomainValue(cba_instance, fs, properties_frame):
    if cba_instance.ap_signal is None:
        messagebox.showerror("Error", "No signal loaded to filter. Please load data first.")
        return
    try:
        # Calculate and display signal properties
        properties = calculateApSignalProperties(cba_instance.ap_signal, fs) # The algorithm of the time domain anaylsis is here
        displaySignalProperties(properties, properties_frame)
    except ValueError:
        messagebox.showerror("Input Error", "No signal loaded to filter. Please load data first.")


def displaySignalProperties(properties, properties_frame):
    # Clear previous properties if they exist
    for widget in properties_frame.winfo_children():
        widget.destroy()

    # Display signal properties
    row = 0
    for key, value in properties.items():
        ttk.Label(properties_frame, text=f"{key}: {value}").grid(row=row, column=0, padx=10, pady=5, sticky=tk.W)
        row += 1
# ===============================================================
# ================= Time Domain Page Layout =====================
# ===============================================================
def create_time_domain_page(notebook, ecg_time_domain_command, ap_time_domain_command):
    # 创建 Time Domain Analysis 页面
    time_domain_page = ttk.Frame(notebook)
    notebook.add(time_domain_page, text="Time Domain Analysis")

    # 按钮框架，位于顶部
    button_frame = ttk.Frame(time_domain_page)
    button_frame.place(x=10, y=10, width=780, height=50)

    # 添加按钮到按钮框架
    ecg_time_analysis_button = ttk.Button(button_frame, text="Analyze ECG Time Domain", command=ecg_time_domain_command)
    ecg_time_analysis_button.place(x=10, y=10, width=180, height=30)

    ap_time_analysis_button = ttk.Button(button_frame, text="Analyze AP Time Domain", command=ap_time_domain_command)
    ap_time_analysis_button.place(x=200, y=10, width=180, height=30)

    # Frame for displaying ECG signal properties (在左侧)
    properties_frame_ecg = ttk.Frame(time_domain_page)
    properties_frame_ecg.place(x=10, y=70, width=350, height=400)

    # Frame for frequency analysis visualization, 放置在 time_domain_page 中
    sd_canvas_frame = ttk.Frame(time_domain_page)
    sd_canvas_frame.config(width=150, height=100)
    sd_canvas_frame.place(x=10, y=280, width=350, height=400)

    # Add a vertical separator between the left and right frames
    separator = ttk.Separator(time_domain_page, orient="vertical")
    separator.place(x=370, y=70, width=10, height=400)

    # Frame for displaying AP signal properties 
    properties_frame_ap = ttk.Frame(time_domain_page)
    properties_frame_ap.place(x=390, y=70, width=350, height=450)

    return {
        "properties_frame_ecg": properties_frame_ecg,
        "properties_frame_ap": properties_frame_ap,
        "sd_canvas_frame": sd_canvas_frame
    }
    # time_domain_page = ttk.Frame(notebook)
    # notebook.add(time_domain_page, text="Time Domain Analysis")

    # button_frame = ttk.Frame(time_domain_page)
    # button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    # ecg_time_analysis_button = ttk.Button(button_frame, text="Analyze ECG Time Domain", command=ecg_time_domain_command)
    # ecg_time_analysis_button.pack(side=tk.LEFT, padx=10, pady=10)

    # ap_time_analysis_button = ttk.Button(button_frame, text="Analyze AP Time Domain", command=ap_time_domain_command)
    # ap_time_analysis_button.pack(side=tk.LEFT, padx=10, pady=10)

    # # Frame for displaying signal properties
    # properties_frame_ecg = ttk.Frame(time_domain_page)
    # properties_frame_ecg.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    # # Frame for frequency analysis visualization, placed below the properties_frame_ecg
    # sd_canvas_frame = ttk.Frame(time_domain_page)
    # sd_canvas_frame.config(width=150, height=100)
    # sd_canvas_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
    # # sd_canvas_frame.place(x=20, y=100)

    # # Add a vertical separator between the two frames
    # separator = ttk.Separator(time_domain_page, orient="vertical")
    # separator.pack(side=tk.TOP, fill=tk.X, padx=5)

    # # Frame for displaying signal properties
    # properties_frame_ap = ttk.Frame(time_domain_page)
    # properties_frame_ap.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)





