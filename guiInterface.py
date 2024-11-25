import numpy as np
import guiVisulaztionOverview as vo
import guiObserver as ob
import matplotlib.pyplot as plt
import tkinter as tk

from tkinter import ttk
from collections import deque
from functools import partial
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# from pylsl import StreamInlet, resolve_stream
# from utilities import *

class guiWindow:
    def __init__(self, root, observer, interactive_plot, data_analyzer):
        self.root = root
        self.root.title("Data Visualization Interface")
        self.root.geometry("1200x1000")
        
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)
        self.data_analyzer = data_analyzer

        self.interactive_plot = None
        self.dataLoader_properties_frame = None
        self.ecg_result_label = None
        self.ap_result_label = None
    
        self.plotObersever = observer
        self.interactive_plot = interactive_plot


    # def resolve_lsl_stream(self, stream_name):
    #     streams = resolve_stream('name', stream_name)
    #     return StreamInlet(streams[0])

    '''
    =========================================================
    The Data Loading Page,
    There are 3 buttons, one mode select box, 3 text inputers
    =========================================================
    '''
    def create_load_data_page(self, load_data, load_rt_data, interact_update):
        '''data loader interface'''
        load_data_page = ttk.Frame(self.notebook)
        self.notebook.add(load_data_page, text="Load Data Page")

        # 顶部控制区域
        ecg_frame = ttk.Frame(load_data_page)
        ecg_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        ttk.Label(ecg_frame, text="Select Mode:").grid(row=0, column=0, padx=5, pady=5)

        self.mode_var = tk.StringVar(value="Select Mode")
        modes = ["Window Create Mode", "Window Drag Mode", "Window Delete Mode", "Plot Adoption Mode"]
        self.mode_combobox = ttk.Combobox(ecg_frame, textvariable=self.mode_var, values=modes, state="readonly")
        self.mode_combobox.grid(row=0, column=1, padx=5, pady=5)

        load_ecg_button = ttk.Button(ecg_frame, text="Load Offline Data", command=load_data)
        load_ecg_button.grid(row=0, column=3, padx=5, pady=5)

        load_rt_data_button = ttk.Button(ecg_frame, text="Load Real-time Data", command=load_rt_data)
        load_rt_data_button.grid(row=0, column=4, padx=5, pady=5)

        # 主内容容器
        main_container = ttk.Frame(load_data_page)
        main_container.grid(row=1, column=0, sticky="nsew")

        # 设置行列权重，确保主内容容器可以动态调整大小
        load_data_page.grid_rowconfigure(1, weight=1)
        load_data_page.grid_columnconfigure(0, weight=1)

        # 创建 canvas 并放置在 main_container 中
        canvas = FigureCanvasTkAgg(self.interactive_plot.fig, master=main_container)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky="nsew")

        # 添加工具栏
        toolbar_frame = ttk.Frame(main_container)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        self.toolbar.update()

        canvas.draw()

        # 绑定事件
        canvas.mpl_connect("button_press_event", self.interactive_plot.on_press)
        canvas.mpl_connect("motion_notify_event", self.interactive_plot.on_drag)
        canvas.mpl_connect("button_release_event", self.interactive_plot.on_release)

        # 属性框架
        self.dataLoader_properties_frame = ttk.Frame(main_container)
        self.dataLoader_properties_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        self.ecg_result_label = ttk.Label(self.dataLoader_properties_frame, text="Window Information: ")
        self.ecg_result_label.grid(row=0, column=0)

        self.mode_combobox.bind("<<ComboboxSelected>>", partial(self.mode_changed_load_page, self.interactive_plot))

        return {
            "load_ecg_button": load_ecg_button,
            "interactive_plot": self.interactive_plot,
            "properties_frame": self.dataLoader_properties_frame,
            "ecg_result_label": self.ecg_result_label,
        }

    def mode_changed_load_page(self, interactive_plot, event):
        selected_mode = self.mode_var.get()
        if selected_mode == "Window Create Mode":
            interactive_plot.toggle_create_mode()
            print("Create Mode Selected")

        elif selected_mode == "Window Drag Mode":
            interactive_plot.toggle_drag_mode()
            print("Edit Mode Selected")

        elif selected_mode == "Window Delete Mode":
            interactive_plot.toggle_del_mode()
            print("View Mode Selected")
            
        elif selected_mode == "Plot Adoption Mode":
            interactive_plot.toggle_plot_mode()
            print("Plot Adoption Mode")

    '''
    =========================================================
    The Signal Processing Page,
    There are 2 buttons
    =========================================================
    '''

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
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        return {
            "lowcut_entry": lowcut_entry,
            "highcut_entry": highcut_entry,
            "canvas_frame": canvas_frame,
        }
    

    '''
    =========================================================
    The Time Domain Analysis Page,
    There are 2 buttons
    =========================================================
    '''
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
    
    '''
    =========================================================
    The Frequency Domain Analysis Page,
    There are 2 buttons
    =========================================================
    '''
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
    
    '''
    =========================================================
    Real-time Analysis Page
    =========================================================
    '''

    def create_real_time_page(self, loadingRtSignal_command):
        real_time_page = ttk.Frame(self.notebook)
        self.notebook.add(real_time_page, text="Real Time Analysis")

        rt_button = ttk.Button(real_time_page, text="Start Collect RT Signal", command=loadingRtSignal_command)
        rt_button.pack(fill='both', expand=True)
    
        self.rt_canvas = FigureCanvasTkAgg(self.data_analyzer.fig, master=real_time_page)
        self.rt_canvas.get_tk_widget().pack(fill='both', expand=True)
        # if self.data_analyzer.isLoading:
        self.data_analyzer.update_plot()

        # return {
        #     "rt_button":rt_button,
        #     "rt_canvas":self.rt_canvas
        # }