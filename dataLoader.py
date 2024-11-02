'''
After get the data, the whole data loader maybe need to re-write
'''

# data_loader_handlers.py
from utilities import *
import tkinter as tk
from tkinter import messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import visulaztionOverview as vo

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