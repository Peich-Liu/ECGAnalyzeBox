# signal_processing_handlers.py
import tkinter as tk
from tkinter import messagebox
from utilities import *

def calculateEcgTimeDomainValue(cba_instance, fs,properties_frame, selected_ranges):

    if cba_instance.ecg_signal is None:
        messagebox.showerror("Error", "No signal loaded to filter. Please load data first.")
        return
    try:
        # Calculate and display signal properties
        # properties, sdFigure = calculateEcgSignalProperties(cba_instance.ecg_signal, fs) # The algorithm of the time domain anaylsis is here
        properties, sdFigure = calculateEcgSignalRangeProperties(cba_instance.ecg_signal, fs, selected_ranges) # The algorithm of the time domain anaylsis is here
        displaySignalProperties(properties, properties_frame)
        # getFigure(sdFigure, sd_canvas_frame)
    except ValueError:
        messagebox.showerror("Input Error", "No signal loaded to filter. Please load data first.")

def calculateApTimeDomainValue(cba_instance, fs, properties_frame, selected_ranges):
    if cba_instance.ap_signal is None:
        messagebox.showerror("Error", "No signal loaded to filter. Please load data first.")
        return
    try:
        # Calculate and display signal properties
        # properties = calculateApSignalProperties(cba_instance.ap_signal, fs) # The algorithm of the time domain anaylsis is here
        properties_per_segment = calculateApSignalRangeProperties(cba_instance.ap_signal, fs, selected_ranges)
        displaySignalProperties(properties_per_segment, properties_frame)
    except ValueError:
        messagebox.showerror("Input Error", "No signal loaded to filter. Please load data first.")

def calculateApandEcgTimeDomainValue(cba_instance, fs, properties_frame, selected_ranges):
    if cba_instance.ecg_signal is None:
        messagebox.showerror("Error", "No signal loaded to filter. Please load data first.")
        return
    try:
        # Calculate and display signal properties
        # properties, sdFigure = calculateEcgSignalProperties(cba_instance.ecg_signal, fs) # The algorithm of the time domain anaylsis is here
        properties_ecg, sdFigure = calculateEcgSignalRangeProperties(cba_instance.ecg_signal, fs, selected_ranges)
        properties_ap = calculateApSignalRangeProperties(cba_instance.ap_signal, fs, selected_ranges) # The algorithm of the time domain anaylsis is here
        # 假设 properties_ecg 和 properties_ap 的格式是类似上面的字典结构
        combined_results = {}

        # 遍历每个范围的结果并将 ECG 和 AP 的结果合并
        for index in properties_ecg.keys():
            combined_results[f"{index}"] = {
                "ECG": properties_ecg[index],
                "AP": properties_ap[index],
            }
        cba_instance.hrv_plot.parameters = combined_results
        displaySignalProperties(combined_results, properties_frame)
        # getFigure(sdFigure, sd_canvas_frame)
    except ValueError:
        messagebox.showerror("Input Error", "No signal loaded to filter. Please load data first.")
