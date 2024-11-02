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