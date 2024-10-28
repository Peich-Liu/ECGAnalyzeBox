# signal_processing_handlers.py
import tkinter as tk
from tkinter import messagebox
from utilities import *

# ===============================================================
# ================= Time Domain Page Layout =====================
# ===============================================================
def create_time_domain_page(notebook, time_domain_command):
    time_domain_page = ttk.Frame(notebook)
    notebook.add(time_domain_page, text="Time Domain Analysis")

    button_frame = ttk.Frame(time_domain_page)
    button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    time_analysis_button = ttk.Button(button_frame, text="Analyze Time Domain", command=time_domain_command)
    time_analysis_button.pack(side=tk.LEFT, padx=10, pady=10)

    # Frame for displaying signal properties
    properties_frame = ttk.Frame(time_domain_page)
    properties_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

    return {
        "properties_frame": properties_frame
    }