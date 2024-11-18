import tkinter as tk

from tkinter import ttk
from tkinter import messagebox
from utilities import *


def perform_frequency_analysis_psd(cba_instance, fs, freq_canvas_frame):
    if cba_instance.ecg_signal is None or fs is None:
        messagebox.showerror("Error", "No signal loaded for frequency analysis. Please load data first.")
        return
    freq, psd = PSDAnalyze(cba_instance.ecg_signal, fs)
    figure = visualPSD(freq, psd)
    getFigure(figure, freq_canvas_frame)