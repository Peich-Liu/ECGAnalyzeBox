import numpy as np
import guiVisulaztionOverview as vo
import tkinter as tk

from tkinter import messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from utilities import *

def load_data(cba_instance):
    try:
        # data_directory = r'/Users/liu/Documents/SC2024fall/250 kun HR.csv'
                # 创建一个隐藏的根窗口，用于弹出文件选择对话框
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口

        # 弹出文件选择对话框
        data_directory = filedialog.askopenfilename(
            title="Select a Data File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        # 检查是否选择了文件
        if not data_directory:
            print("No file selected.")
            return

        # 加载ECG数据
        ecg, ap = loadData(data_directory)
        if ecg is None:
            messagebox.showerror("Error", "Failed to read records")
            return

        cba_instance.update_ecg_signal(ecg, 250)  # 设置采样率
        cba_instance.update_ap_signal(ap, 250)

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid start and end sample values")

def load_rt_data(cba_instance):
    try:
        # data_directory = r'C:\Document\sc2024\filtered_ecg_with_quality.csv'
        # data_directory = r'c:\Document\sc2024\filtered_ecg_with_snr.csv'
        # data_directory = r'/Users/liu/Documents/SC2024fall/filtered_ecg_with_snr.csv'
        # 创建一个隐藏的根窗口，用于弹出文件选择对话框
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口

        # 弹出文件选择对话框
        data_directory = filedialog.askopenfilename(
            title="Select a Data File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        # 检查是否选择了文件
        if not data_directory:
            print("No file selected.")
            return
        # 加载ECG数据
        ecg, ap, quality = loadRtData(data_directory)
        if ecg is None:
            messagebox.showerror("Error", "Failed to read records")
            return
        cba_instance.update_quality(quality)
        print(cba_instance.quality)
        cba_instance.update_ecg_signal(ecg, 250)  # 设置采样率
        cba_instance.update_ap_signal(ap, 250)

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid start and end sample values")







def load_ecg_data(cba_instance, patient_id_entry, start_entry, end_entry):

    try:
        print("test")
        patient_id = patient_id_entry.get()
        start = int(start_entry.get())
        end = int(end_entry.get())
        # data_directory = filedialog.askdirectory(title="Select Data Directory")#modify here after gain data
        data_directory = r'C:\Document\sc2024\fantasia-database-1.0.0/'

        ##Here is the data load function
        records, _ = readPatientRecords2(patient_id, data_directory)
        
        if not records:
            messagebox.showerror("Error", "Failed to read records")
            return
        
        concatenated_signal = concatenateECG(records, start, end)
        cba_instance.update_ecg_signal(concatenated_signal, 250)#fs need  change after gain data

        # Plot the loaded signal
        # plot_signals(cba_instance, canvas_frame)

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid start and end sample values")

def load_ap_data(cba_instance, patient_id_entry, start_entry, end_entry):
    try:
        print("test")
        patient_id = patient_id_entry.get()
        start = int(start_entry.get())
        end = int(end_entry.get())
        # data_directory = filedialog.askdirectory(title="Select Data Directory")#modify here after gain data
        data_directory = r'C:\Document\sc2024\fantasia-database-1.0.0/'

        ##Here is the data load function
        records, _ = readPatientRecords2(patient_id, data_directory)
        
        if not records:
            messagebox.showerror("Error", "Failed to read records")
            return
        
        concatenated_signal = concatenateAP(records, start, end)
        cba_instance.update_ap_signal(concatenated_signal, 250)#fs need  change after gain data

        # Plot the loaded signal
        # plot_signals(cba_instance, canvas_frame)


    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid start and end sample values")