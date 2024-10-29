'''
Develop here later after having the signal
'''

import tkinter as tk
from tkinter import messagebox
from utilities import *

def create_visual_page(notebook):
    visual_overview_page = ttk.Frame(notebook)
    notebook.add(visual_overview_page, text="Visualaztion")

    button_frame = ttk.Frame(visual_overview_page)
    button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    time_analysis_button = ttk.Button(button_frame, text="Visual All the Signals", command=visual_overview_page)
    time_analysis_button.pack(side=tk.LEFT, padx=10, pady=10)

    # Frame for displaying signal properties
    properties_frame = ttk.Frame(visual_overview_page)
    properties_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)