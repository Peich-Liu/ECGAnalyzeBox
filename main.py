from scipy.signal import butter, filtfilt
import numpy as np
import os
import wfdb
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from methodLibrary import *


def getFigure(figure, canvas_frame):
    # Clear previous plot if it exists
    for widget in canvas_frame.winfo_children():
        widget.destroy()
    # Embed the plot in the Tkinter canvas
    canvas = FigureCanvasTkAgg(figure, master=canvas_frame)
    canvas.draw()
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)

    # Add navigation toolbar for better interactivity
    toolbar = NavigationToolbar2Tk(canvas, canvas_frame)
    toolbar.update()
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)


def displaySignalProperties(properties, properties_frame):
    # Clear previous properties if they exist
    for widget in properties_frame.winfo_children():
        widget.destroy()

    # Display signal properties
    row = 0
    for key, value in properties.items():
        ttk.Label(properties_frame, text=f"{key}: {value}").grid(row=row, column=0, padx=10, pady=5, sticky=tk.W)
        row += 1

def performFrequencyDomainAnalysis():
    pass

def main():
    concatenated_signal = None
    fs = None
    def loadData():
        nonlocal concatenated_signal, fs
        patient_id = patient_id_entry.get()
        # data_directory = filedialog.askdirectory(title="Select Data Directory")
        data_directory = r'C:\Document\sc2024\mit-bih-arrhythmia-database-1.0.0/' 
        records, fs, _ = readPatientRecords(patient_id, data_directory)#after gian the data, modify here

        if not records:
            messagebox.showerror("Error", "Failed to read records")
            return
        try:
            start = int(start_entry.get())
            end = int(end_entry.get())
            concatenated_signal = concatenateandProcessSignals(records, start, end)
            figure = visualizeSignalinGUI(concatenated_signal)
            getFigure(figure, canvas_frame)
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid start and end sample values")
            
    def calculateTimeDomainValue():
        nonlocal concatenated_signal, fs

        if concatenated_signal is None:
            messagebox.showerror("Error", "No signal loaded to filter. Please load data first.")
            return
        try:
            # Calculate and display signal properties
            properties = calculateSignalProperties(concatenated_signal, fs) # The algorithm of the time domain anaylsis is here
            displaySignalProperties(properties, properties_frame)
        except ValueError:
            messagebox.showerror("Input Error", "No signal loaded to filter. Please load data first.")

    def filterSignal():
        nonlocal concatenated_signal,fs
        if concatenated_signal is None:
            messagebox.showerror("Error", "No signal loaded to filter. Please load data first.")
            return
        # lowcut = 0.5 
        # highcut = 1.0
        try:
            lowcut = float(lowcut_entry.get())
            highcut = float(highcut_entry.get())
            # order = int(order_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid lowcut, highcut, and filter order values")
            return
        # The algorithm of filter is here(bandpass filter, maybe change later)
        filtered_signal = bandpass_filter(concatenated_signal, lowcut, highcut, sampling_rate=fs) #, order=4 add order later maybe
        concatenated_signal = filtered_signal
        figure = visualizeSignalinGUI(filtered_signal)
        getFigure(figure, canvas_frame)

    def artifiactProcess():
        nonlocal concatenated_signal,fs
        if concatenated_signal is None:
            messagebox.showerror("Error", "No signal loaded to remover. Please load data first.")
            return
        artificat_removed_signal = remove_artifacts(concatenated_signal)
        concatenated_signal = artificat_removed_signal
        figure = visualizeSignalinGUI(artificat_removed_signal)
        getFigure(figure, canvas_frame)

    def performFrequencyAnalysis():
        if concatenated_signal is None or fs is None:
            messagebox.showerror("Error", "No signal loaded for frequency analysis. Please load data first.")
            return
        figure = performFrequencyDomainAnalysis(concatenated_signal, fs)
        getFigure(figure, freq_canvas_frame)

    # Set up the main GUI window
    root = tk.Tk()
    root.title("ECG Signal Processing GUI")
    root.geometry("1000x800")

    # Create a notebook widget for multiple tabs
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    # Page 0: Time Domain Analysis
    load_data_page = ttk.Frame(notebook)
    notebook.add(load_data_page, text="Load Data Page")

    # Page 1: Signal Processing
    signal_processing = ttk.Frame(notebook)
    notebook.add(signal_processing, text="Signal Processing")

    # Page 2: Time Domain Analysis
    time_domain_page = ttk.Frame(notebook)
    notebook.add(time_domain_page, text="Time Domain Analysis")

    # Page 3: Frequency Domain Analysis
    frequency_domain_page = ttk.Frame(notebook)
    notebook.add(frequency_domain_page, text="Frequency Domain Analysis")

    # =================== Load Data Page Layout =================== #
    ############################
    ##Frame for base information
    ############################
    input_frame = ttk.Frame(load_data_page)
    input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    ttk.Label(input_frame, text="Patient ID:").grid(row=0, column=0, padx=5, pady=5)
    patient_id_entry = ttk.Entry(input_frame)
    patient_id_entry.grid(row=0, column=1, padx=5, pady=5)
    patient_id_entry.insert(0, "10")

    ttk.Label(input_frame, text="Start Sample:").grid(row=0, column=2, padx=5, pady=5)
    start_entry = ttk.Entry(input_frame)
    start_entry.grid(row=0, column=3, padx=5, pady=5)
    start_entry.insert(0, "0")

    ttk.Label(input_frame, text="End Sample:").grid(row=0, column=4, padx=5, pady=5)
    end_entry = ttk.Entry(input_frame)
    end_entry.grid(row=0, column=5, padx=5, pady=5)
    end_entry.insert(0, "5000")

    ############################
    ##Load data button
    ############################
    button_frame = ttk.Frame(load_data_page)
    button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    load_button = ttk.Button(button_frame, text="Load and Process Data", command=loadData)
    load_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

    # ====================================================================== 
    # =================== Signal Processing Page Layout ====================
    # ====================================================================== 

    #########################
    ##Input Frame for filter
    input_frame1 = ttk.Frame(signal_processing)
    input_frame1.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    ttk.Label(input_frame1, text="Lowcut Frequency:").grid(row=1, column=0, padx=5, pady=5)
    lowcut_entry = ttk.Entry(input_frame1)
    lowcut_entry.grid(row=1, column=1, padx=5, pady=5)
    lowcut_entry.insert(0, "0.5")

    ttk.Label(input_frame1, text="Highcut Frequency:").grid(row=1, column=2, padx=5, pady=5)
    highcut_entry = ttk.Entry(input_frame1)
    highcut_entry.grid(row=1, column=3, padx=5, pady=5)
    highcut_entry.insert(0, "50.0")

    #########################
    ##Frame for buttons
    button_frame = ttk.Frame(signal_processing)
    button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
    # align button
    button_frame.grid_columnconfigure(0, weight=1)
    button_frame.grid_columnconfigure(1, weight=1)

    filter_button = ttk.Button(button_frame, text="Filter the ECG Signal", command=filterSignal)
    filter_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    artificat_button = ttk.Button(button_frame, text="Artifiact Remove", command=artifiactProcess)
    artificat_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    #########################
    ##Frame for signal visualization
    canvas_frame = ttk.Frame(root)
    canvas_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)
    ########################################
    ##Frame for displaying signal properties
    properties_frame = ttk.Frame(root)
    properties_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

    # ===============================================================
    # ================= Time Domain Page Layout =====================
    # ===============================================================

    button_frame = ttk.Frame(time_domain_page)
    button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    time_anaylsis_button = ttk.Button(button_frame, text="Analyze Time Domain", command=calculateTimeDomainValue)
    # time_anaylsis_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
    time_anaylsis_button.pack(side=tk.LEFT, padx=10, pady=10)

    # Frame for displaying signal properties
    properties_frame = ttk.Frame(time_domain_page)
    properties_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

    # ====================================================================
    # =================== Frequency Domain Page Layout =================== 
    # ====================================================================

    # Frame for frequency analysis controls
    freq_control_frame = ttk.Frame(frequency_domain_page)
    freq_control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    analyze_button = ttk.Button(freq_control_frame, text="Analyze Frequency Domain", command=performFrequencyAnalysis)
    analyze_button.pack(side=tk.LEFT, padx=10, pady=10)

    # Frame for frequency analysis visualization
    freq_canvas_frame = ttk.Frame(frequency_domain_page)
    freq_canvas_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
