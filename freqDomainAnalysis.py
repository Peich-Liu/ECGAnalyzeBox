import tkinter as tk
from tkinter import ttk

# ===============================================================
# ================= Frequency Domain Analysis =====================
# ===============================================================
def create_frequency_domain_page(notebook, frequency_analysis_command):
    frequency_domain_page = ttk.Frame(notebook)
    notebook.add(frequency_domain_page, text="Frequency Domain Analysis")

    # Frame for frequency analysis controls and visualization
    freq_control_frame = ttk.Frame(frequency_domain_page)
    freq_control_frame.place(x=20, y=20, width=300, height=50)

    # PSD按钮
    analyze_button = ttk.Button(freq_control_frame, text="PSD Visualize", command=frequency_analysis_command)
    analyze_button.place(x=10, y=10)

    # Frame for frequency analysis visualization
    freq_canvas_frame = ttk.Frame(frequency_domain_page)
    freq_canvas_frame.config(width=150, height=100)
    freq_canvas_frame.place(x=20, y=100)

    return {
        "freq_canvas_frame": freq_canvas_frame
    }