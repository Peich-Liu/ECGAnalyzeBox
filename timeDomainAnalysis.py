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
