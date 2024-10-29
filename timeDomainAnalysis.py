# signal_processing_handlers.py
import tkinter as tk
from tkinter import messagebox
from utilities import *

def calculateEcgTimeDomainValue(cba_instance, fs,properties_frame):

    if cba_instance.ecg_signal is None:
        messagebox.showerror("Error", "No signal loaded to filter. Please load data first.")
        return
    try:
        # Calculate and display signal properties
        properties = calculateSignalProperties(cba_instance.ecg_signal, fs) # The algorithm of the time domain anaylsis is here
        displaySignalProperties(properties, properties_frame)
    except ValueError:
        messagebox.showerror("Input Error", "No signal loaded to filter. Please load data first.")
def calculateApTimeDomainValue(cba_instance, fs,properties_frame):
    pass
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
    time_domain_page = ttk.Frame(notebook)
    notebook.add(time_domain_page, text="Time Domain Analysis")

    button_frame = ttk.Frame(time_domain_page)
    button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    ecg_time_analysis_button = ttk.Button(button_frame, text="Analyze ECG Time Domain", command=ecg_time_domain_command)
    ecg_time_analysis_button.pack(side=tk.LEFT, padx=10, pady=10)

    ap_time_analysis_button = ttk.Button(button_frame, text="Analyze AP Time Domain", command=ap_time_domain_command)
    ap_time_analysis_button.pack(side=tk.LEFT, padx=10, pady=10)

    # Frame for displaying signal properties
    properties_frame = ttk.Frame(time_domain_page)
    properties_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)



    return {
        "properties_frame": properties_frame
    }