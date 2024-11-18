import numpy as np
import guiVisulaztionOverview as vo
import tkinter as tk

from tkinter import messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from utilities import *

def load_data(cba_instance):
    try:
        data_directory = r'C:\Document\sc2024\250 kun HR.csv'

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
        cba_instance.update_ecg_signal(cba_instance.guiWindow.ecg_window, 250)  # 设置采样率
        cba_instance.update_ap_signal(cba_instance.guiWindow.ap_window, 250)
        print(len(cba_instance.guiWindow.ecg_window), len(cba_instance.guiWindow.ap_window))

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid start and end sample values")
# def show_signal_qulity(filePath, ):






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