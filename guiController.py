import tkinter as tk
from tkinter import ttk
import visulaztionOverview as vo
from functools import partial
import observer as ob
from utilities import *

class guiWindow:
    def __init__(self, root, observer, interactive_plot):
        self.root = root
        self.root.title("Data Visualization Interface")
        self.root.geometry("1200x1000")
        
        # 创建一个Notebook容器
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=False)
        
        self.interactive_plot = None

        self.plotObersever = observer
        self.interactive_plot = interactive_plot

        # self.interactive_plot = vo.InteractivePlot(self.plotObersever)


        # self.plotObersever.subscribe(calculate_mean)
        # self.plotObersever.subscribe(calculate_variance)

    def create_canvas_for_page(self, page):
        """为指定的页面创建一个Canvas并绑定共享的Figure"""
        canvas = FigureCanvasTkAgg(self.interactive_plot.fig, master=page)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # 绑定事件
        canvas.mpl_connect("button_press_event", self.interactive_plot.on_press)
        canvas.mpl_connect("motion_notify_event", self.interactive_plot.on_drag)
        canvas.mpl_connect("button_release_event", self.interactive_plot.on_release)
        return canvas
    
    def mode_changed_load_page(self, interactive_plot, event):
        selected_mode = self.mode_var.get()
        if selected_mode == "Create Mode":
            interactive_plot.toggle_create_mode()
            print("Create Mode Selected")

        elif selected_mode == "Drag Mode":
            interactive_plot.toggle_drag_mode()
            print("Edit Mode Selected")

        elif selected_mode == "Delete Mode":
            print("View Mode Selected")

    '''=========================================================
    The Data Loading Page,
    There are 3 buttons, one mode select box, 3 text inputers
    ========================================================='''

    def create_load_data_page(self, load_ecg_command, load_ap_command, load_eeg_command, vis_data, interact_update):
        '''data loader interface'''
        load_data_page = ttk.Frame(self.notebook)
        self.notebook.add(load_data_page, text="Load Data Page")

        # =================== Load ECG =================== #
        # Create the first frame for Load ECG Button
        ecg_frame = ttk.Frame(load_data_page)
        ecg_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        ttk.Label(ecg_frame, text="Select Mode:").grid(row=0, column=0, padx=5, pady=5)
        
        self.mode_var = tk.StringVar(value="Select Mode")
        modes = ["Create Mode", "Drag Mode", "Delete Mode"]  # 模式选项，可以根据需要扩展
        self.mode_combobox = ttk.Combobox(ecg_frame, textvariable=self.mode_var, values=modes, state="readonly")
        self.mode_combobox.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(ecg_frame, text="Patient ID:").grid(row=1, column=0, padx=5, pady=5)
        ecg_patient_id_entry = ttk.Entry(ecg_frame)
        ecg_patient_id_entry.grid(row=1, column=1, padx=5, pady=5)
        ecg_patient_id_entry.insert(0, "f2o01")

        ttk.Label(ecg_frame, text="Start Sample:").grid(row=1, column=2, padx=5, pady=5)
        ecg_start_entry = ttk.Entry(ecg_frame)
        ecg_start_entry.grid(row=1, column=3, padx=5, pady=5)
        ecg_start_entry.insert(0, "0")

        ttk.Label(ecg_frame, text="End Sample:").grid(row=1, column=4, padx=5, pady=5)
        ecg_end_entry = ttk.Entry(ecg_frame)
        ecg_end_entry.grid(row=1, column=5, padx=5, pady=5)
        ecg_end_entry.insert(0, "50000")

        load_ecg_button = ttk.Button(ecg_frame, text="Load ECG Data", command=load_ecg_command)
        load_ecg_button.grid(row=1, column=6, padx=5, pady=5)

        load_ap_button = ttk.Button(ecg_frame, text="Load AP Data", command=load_ap_command)
        load_ap_button.grid(row=1, column=7, padx=5, pady=5)

        visual_button = ttk.Button(ecg_frame, text="Visualize Data", command=vis_data)
        visual_button.grid(row=1, column=8, padx=5, pady=5)

        self.create_canvas_for_page(load_data_page)
        self.mode_combobox.bind("<<ComboboxSelected>>", partial(self.mode_changed_load_page, self.interactive_plot))


        return {
            "patient_id_entry": ecg_patient_id_entry,
            "start_entry": ecg_start_entry,
            "end_entry": ecg_end_entry,
            # "canvas_frame_ecg": self.canvas_frame_root,
            "load_ecg_button": load_ecg_button,  # Add button to the return dictionary
            "load_ap_button": load_ap_button,     # Add button to the return dictionary
            "interactive_plot": self.interactive_plot,
            # "parameter_info":properties_frame_window
        }
    
    '''=========================================================
    The Signal Processing Page,
    There are 2 buttons
    ========================================================='''

    def create_signal_processing_page(self, filter_command, artifact_command, vis_data):
        signal_processing_page = ttk.Frame(self.notebook)
        self.notebook.add(signal_processing_page, text="Signal Processing")

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
        return {
            "lowcut_entry": lowcut_entry,
            "highcut_entry": highcut_entry,
            "canvas_frame": canvas_frame,
        }
    

    '''=========================================================
    The Time Domain Analysis Page,
    There are 2 buttons
   ========================================================='''
    def create_time_domain_page(self, ecg_time_domain_command, ap_time_domain_command):
        time_domain_page = ttk.Frame(self.notebook)
        self.notebook.add(time_domain_page, text="Time Domain Analysis")

        button_frame = ttk.Frame(time_domain_page)
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        ecg_time_analysis_button = ttk.Button(button_frame, text="Analyze ECG Time Domain", command=ecg_time_domain_command)
        ecg_time_analysis_button.grid(row=0, column=0, padx=5, pady=5)

        ap_time_analysis_button = ttk.Button(button_frame, text="Analyze AP Time Domain", command=ap_time_domain_command)
        ap_time_analysis_button.grid(row=0, column=1, padx=5, pady=5)

        properties_frame_ecg = ttk.Frame(time_domain_page)
        properties_frame_ecg.place(x=10, y=70, width=350, height=400)

        sd_canvas_frame = ttk.Frame(time_domain_page)
        sd_canvas_frame.config(width=150, height=100)
        sd_canvas_frame.place(x=10, y=280, width=350, height=400)

        separator = ttk.Separator(time_domain_page, orient="vertical")
        separator.place(x=370, y=70, width=10, height=400)

        properties_frame_ap = ttk.Frame(time_domain_page)
        properties_frame_ap.place(x=390, y=70, width=500, height=500)

        return {
            "properties_frame_ecg": properties_frame_ecg,
            "properties_frame_ap": properties_frame_ap,
            "sd_canvas_frame": sd_canvas_frame
        }
    
    '''=========================================================
    The Frequency Domain Analysis Page,
    There are 2 buttons
   ========================================================='''

    def create_frequency_domain_page(self, frequency_analysis_command):
        frequency_domain_page = ttk.Frame(self.notebook)
        self.notebook.add(frequency_domain_page, text="Frequency Domain Analysis")

        # Frame for frequency analysis controls and visualization
        freq_control_frame = ttk.Frame(frequency_domain_page)
        freq_control_frame.place(x=20, y=20, width=300, height=50)

        # PSD按钮
        analyze_button = ttk.Button(freq_control_frame, text="ECG PSD Visualize", command=frequency_analysis_command)
        analyze_button.place(x=10, y=10)

        # Frame for frequency analysis visualization
        freq_canvas_frame = ttk.Frame(frequency_domain_page)
        freq_canvas_frame.config(width=150, height=100)
        freq_canvas_frame.place(x=20, y=100)

        return {
            "freq_canvas_frame": freq_canvas_frame
        }