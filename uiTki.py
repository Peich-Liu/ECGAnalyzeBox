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
        ttk.Label(properties_frame, text=f"{key}: {value}").grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
        row += 1


def main():
    def load_data():
        patient_id = patient_id_entry.get()
        # data_directory = filedialog.askdirectory(title="Select Data Directory")
        data_directory = r'C:\Document\sc2024\mit-bih-arrhythmia-database-1.0.0/'  # Replace with your data directory

        records, annotations = readPatientRecords(patient_id, data_directory)

        if not records:
            messagebox.showerror("Error", "Failed to read records")
            return

        try:
            start = int(start_entry.get())
            end = int(end_entry.get())
            concatenated_signal = concatenateandProcessSignals(records, start, end)
            figure = visualizeSignalinGUI(concatenated_signal)
            getFigure(figure, canvas_frame)
            
            # Calculate and display signal properties
            properties = calculateSignalProperties(concatenated_signal)
            displaySignalProperties(properties, properties_frame)
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid start and end sample values")

    # Set up the main GUI window
    root = tk.Tk()
    root.title("ECG Signal Processing GUI")
    root.geometry("1000x800")

    # Frame for user inputs
    input_frame = ttk.Frame(root)
    input_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    ttk.Label(input_frame, text="Patient ID:").grid(row=0, column=0, padx=5, pady=5)
    patient_id_entry = ttk.Entry(input_frame)
    patient_id_entry.grid(row=0, column=1, padx=5, pady=5)

    ttk.Label(input_frame, text="Start Sample:").grid(row=1, column=0, padx=5, pady=5)
    start_entry = ttk.Entry(input_frame)
    start_entry.grid(row=1, column=1, padx=5, pady=5)

    ttk.Label(input_frame, text="End Sample:").grid(row=2, column=0, padx=5, pady=5)
    end_entry = ttk.Entry(input_frame)
    end_entry.grid(row=2, column=1, padx=5, pady=5)

    load_button = ttk.Button(input_frame, text="Load and Process Data", command=load_data)
    load_button.grid(row=3, column=0, columnspan=2, pady=10)

    # Frame for signal visualization
    canvas_frame = ttk.Frame(root)
    canvas_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Frame for displaying signal properties
    properties_frame = ttk.Frame(root)
    properties_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
