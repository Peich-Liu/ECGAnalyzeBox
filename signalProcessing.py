# signal_processing_handlers.py
import tkinter as tk
from tkinter import messagebox
from utilities import *

def filter_signal(cba_instance, lowcut_entry, highcut_entry):
    if ((cba_instance.ecg_signal is None) & (cba_instance.ap_signal is None)):
        messagebox.showerror("Error", "No signal loaded to filter. Please load data first.")
        return
    try:
        lowcut = float(lowcut_entry.get())
        highcut = float(highcut_entry.get())
        filtered_ecg_signal = bandpass_filter(cba_instance.ecg_signal, lowcut, highcut, sampling_rate=cba_instance.fs)
        filtered_ap_signal = bandpass_filter(cba_instance.ap_signal, lowcut, highcut, sampling_rate=cba_instance.fs)

        cba_instance.update_ecg_signal(filtered_ecg_signal, 250)#fs need  change after gain data
        cba_instance.update_ap_signal(filtered_ap_signal, 250)#fs need  change after gain data
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid lowcut and highcut filter values")


    # figure = visualizeSignalinGUI(filtered_signal)
    # getFigure(figure, canvas_frame)

    # try:
    #     lowcut = float(lowcut_entry.get())
    #     highcut = float(highcut_entry.get())
    #     filtered_signal = bandpass_filter(cba_instance.selected_channel_signal, lowcut, highcut, sampling_rate=4)
    #     figure = visualizeSignalinGUI(filtered_signal)
    #     getFigure(figure, canvas_frame)

    # except ValueError:
    #     messagebox.showerror("Input Error", "Please enter valid lowcut and highcut filter values")


# ====================================================================== 
# =================== Signal Processing Page Layout ====================
# ====================================================================== 
def create_signal_processing_page(notebook, filter_command, artifact_command, vis_data):
    signal_processing_page = ttk.Frame(notebook)
    notebook.add(signal_processing_page, text="Signal Processing")

    # Input frame for filter
    input_frame = ttk.Frame(signal_processing_page)
    input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    ttk.Label(input_frame, text="Lowcut Frequency:").grid(row=1, column=0, padx=5, pady=5)
    lowcut_entry = ttk.Entry(input_frame)
    lowcut_entry.grid(row=1, column=1, padx=5, pady=5)
    lowcut_entry.insert(0, "0.5")

    ttk.Label(input_frame, text="Highcut Frequency:").grid(row=1, column=2, padx=5, pady=5)
    highcut_entry = ttk.Entry(input_frame)
    highcut_entry.grid(row=1, column=3, padx=5, pady=5)
    highcut_entry.insert(0, "50.0")

    # Frame for buttons
    button_frame = ttk.Frame(signal_processing_page)
    button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    filter_button = ttk.Button(button_frame, text="Filter the ECG Signal", command=filter_command)
    filter_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

    artifact_button = ttk.Button(button_frame, text="Artifact Remove", command=artifact_command)
    artifact_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    visual_button = ttk.Button(button_frame, text="Visualize Filtered Data", command=vis_data)
    visual_button.grid(row=0, column=8, padx=5, pady=5)
    # Frame for signal visualization
    canvas_frame = ttk.Frame(signal_processing_page)
    canvas_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)

    # # Frame for displaying signal properties
    # properties_frame = ttk.Frame(signal_processing_page)
    # properties_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

    # Return the components needed for interaction
    return {
        "lowcut_entry": lowcut_entry,
        "highcut_entry": highcut_entry,
        "canvas_frame": canvas_frame,
    }
