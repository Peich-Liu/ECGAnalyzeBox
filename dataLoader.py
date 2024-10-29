'''
After get the data, the whole data loader maybe need to re-write
'''

# data_loader_handlers.py
from utilities import *
import tkinter as tk
from tkinter import messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np


# def plot_signal(cba_instance, canvas_frame):
#     figure = Figure(figsize=(8, 4), dpi=100)
#     ax = figure.add_subplot(111)

#     # Plot the entire loaded signal
#     ax.plot(np.arange(len(cba_instance.whole_signal)), cba_instance.whole_signal, label="Signal")
#     ax.set_title("Loaded Signal")
#     ax.set_xlabel("Time (samples)")
#     ax.set_ylabel("Amplitude")
#     ax.legend()

#     # Use the getFigure function to display the figure
#     getFigure(figure, canvas_frame)

def load_ecg_data(cba_instance, patient_id_entry, start_entry, end_entry):

    try:
        print("test")
        patient_id = patient_id_entry.get()
        start = int(start_entry.get())
        end = int(end_entry.get())
        # data_directory = filedialog.askdirectory(title="Select Data Directory")#modify here after gain data
        data_directory = r'C:\Document\sc2024\fantasia-database-1.0.0/'

        ##Here is the data load function
        records, _ = readPatientRecords2(patient_id, data_directory)
        
        if not records:
            messagebox.showerror("Error", "Failed to read records")
            return
        
        concatenated_signal = concatenateECG(records, start, end)
        cba_instance.update_ecg_signal(concatenated_signal, 250)#fs need  change after gain data

        # Plot the loaded signal
        # plot_signals(cba_instance, canvas_frame)

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid start and end sample values")

def load_ap_data(cba_instance, patient_id_entry, start_entry, end_entry):
    try:
        print("test")
        patient_id = patient_id_entry.get()
        start = int(start_entry.get())
        end = int(end_entry.get())
        # data_directory = filedialog.askdirectory(title="Select Data Directory")#modify here after gain data
        data_directory = r'C:\Document\sc2024\fantasia-database-1.0.0/'

        ##Here is the data load function
        records, _ = readPatientRecords2(patient_id, data_directory)
        
        if not records:
            messagebox.showerror("Error", "Failed to read records")
            return
        
        concatenated_signal = concatenateAP(records, start, end)
        cba_instance.update_ap_signal(concatenated_signal, 250)#fs need  change after gain data

        # Plot the loaded signal
        # plot_signals(cba_instance, canvas_frame)


    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid start and end sample values")

# =================== Load Data Page Layout =================== #
def create_load_data_page(notebook, load_ecg_command, load_ap_command, load_eeg_command, vis_data):
    load_data_page = ttk.Frame(notebook)
    notebook.add(load_data_page, text="Load Data Page")

    # =================== Load ECG =================== #
    # Create the first frame for Load ECG Button
    ecg_frame = ttk.Frame(load_data_page)
    ecg_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    ttk.Label(ecg_frame, text="Patient ID:").grid(row=0, column=0, padx=5, pady=5)
    ecg_patient_id_entry = ttk.Entry(ecg_frame)
    ecg_patient_id_entry.grid(row=0, column=1, padx=5, pady=5)
    ecg_patient_id_entry.insert(0, "f2o01")

    ttk.Label(ecg_frame, text="Start Sample:").grid(row=0, column=2, padx=5, pady=5)
    ecg_start_entry = ttk.Entry(ecg_frame)
    ecg_start_entry.grid(row=0, column=3, padx=5, pady=5)
    ecg_start_entry.insert(0, "0")

    ttk.Label(ecg_frame, text="End Sample:").grid(row=0, column=4, padx=5, pady=5)
    ecg_end_entry = ttk.Entry(ecg_frame)
    ecg_end_entry.grid(row=0, column=5, padx=5, pady=5)
    ecg_end_entry.insert(0, "50000")

    load_ecg_button = ttk.Button(ecg_frame, text="Load ECG Data", command=load_ecg_command)
    load_ecg_button.grid(row=0, column=6, padx=5, pady=5)

    load_ap_button = ttk.Button(ecg_frame, text="Load AP Data", command=load_ap_command)
    load_ap_button.grid(row=0, column=7, padx=5, pady=5)

    visual_button = ttk.Button(ecg_frame, text="Visualize Data", command=vis_data)
    visual_button.grid(row=0, column=8, padx=5, pady=5)
    # Separator Line
    separator1 = ttk.Separator(load_data_page, orient='horizontal')
    separator1.pack(fill='x', padx=10, pady=5)
    # Frame for plotting the loaded signal
    canvas_frame_ecg = ttk.Frame(load_data_page)
    canvas_frame_ecg.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)


    return {
        "patient_id_entry": ecg_patient_id_entry,
        "start_entry": ecg_start_entry,
        "end_entry": ecg_end_entry,
        "canvas_frame_ecg": canvas_frame_ecg,
        "load_ecg_button": load_ecg_button,  # Add button to the return dictionary
        "load_ap_button": load_ap_button     # Add button to the return dictionary
    }







    #old function
# def load_ecg_data(cba_instance, patient_id_entry, start_entry, end_entry):
#     try:
#         patient_id = patient_id_entry.get()
#         start = int(start_entry.get())
#         end = int(end_entry.get())
#         # data_directory = filedialog.askdirectory(title="Select Data Directory")#modify here after gain data
#         data_directory = r'C:\Document\sc2024\mit-bih-arrhythmia-database-1.0.0/'

#         ##Here is the data load function
#         records, fs, _ = readPatientRecords(patient_id, data_directory)
        
#         if not records:
#             messagebox.showerror("Error", "Failed to read records")
#             return
        
#         concatenated_signal = concatenateSignals(records, start, end)
#         cba_instance.update_whole_signal(concatenated_signal, fs)
#         # cba_instance.update_whole_signal(concatenated_signal, fs)

#         # channel_options = [f"Channel {i+1}" for i in range(concatenated_signal.shape[1])]
#         # channel_selector['values'] = channel_options
#         # channel_selector.current(0)

#         # # cba_instance.selected_channel_signal = concatenated_signal

#         # # cba_instance.update_signal(concatenated_signal, fs)
#         # channel_selector.bind("<<ComboboxSelected>>", lambda e: cba_instance.update_signal(concatenated_signal, fs))

#         # # cba_instance.fs = fs

#         # figure = visualizeSignalinGUISelectChannel(cba_instance.selected_channel_signal, channel_selector.current())
#         # getFigure(figure, load_data_canvas_frame)

#     except ValueError:
#         messagebox.showerror("Input Error", "Please enter valid start and end sample values")



    # return load_data_page
    # load_data_page = ttk.Frame(notebook)
    # notebook.add(load_data_page, text="Load Data Page")

    # # Frame for base information
    # input_frame = ttk.Frame(load_data_page)
    # input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    # ttk.Label(input_frame, text="Patient ID:").grid(row=0, column=0, padx=5, pady=5)
    # patient_id_entry = ttk.Entry(input_frame)
    # patient_id_entry.grid(row=0, column=1, padx=5, pady=5)
    # patient_id_entry.insert(0, "10")

    # ttk.Label(input_frame, text="Start Sample:").grid(row=0, column=2, padx=5, pady=5)
    # start_entry = ttk.Entry(input_frame)
    # start_entry.grid(row=0, column=3, padx=5, pady=5)
    # start_entry.insert(0, "0")

    # ttk.Label(input_frame, text="End Sample:").grid(row=0, column=4, padx=5, pady=5)
    # end_entry = ttk.Entry(input_frame)
    # end_entry.grid(row=0, column=5, padx=5, pady=5)
    # end_entry.insert(0, "50000")

    # # Frame for Load data button
    # button_frame = ttk.Frame(load_data_page)
    # button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    # load_button = ttk.Button(button_frame, text="Load and Process Data", command=load_data_command)
    # load_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

    # # Frame for channel selection
    # channel_frame = ttk.Frame(load_data_page)
    # channel_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    # ttk.Label(channel_frame, text="Select Channel:").grid(row=0, column=0, padx=5, pady=5)
    # channel_selector = ttk.Combobox(channel_frame, state="readonly")
    # channel_selector.grid(row=0, column=1, padx=5, pady=5)

    
    # # Frame for signal visualization
    # load_data_canvas_frame = ttk.Frame(load_data_page)
    # load_data_canvas_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)

    # # Frame for displaying signal properties
    # properties_frame = ttk.Frame(load_data_page)
    # properties_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

    # # Return the components needed for interaction
    # return {
    #     "patient_id_entry": patient_id_entry,
    #     "start_entry": start_entry,
    #     "end_entry": end_entry,
    #     "channel_selector": channel_selector,
    #     "load_data_canvas_frame": load_data_canvas_frame,
    #     "properties_frame": properties_frame
    # }