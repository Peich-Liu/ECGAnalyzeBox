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