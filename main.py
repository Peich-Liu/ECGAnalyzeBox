import tkinter as tk
from tkinter import ttk
import dataLoader as dl
import signalProcessing as sp
import timeDomainAnalysis as tda
import freqDomainAnalysis as fda
import visulaztionOverview as vo
import guiController as gui
import observer as ob
from utilities import *


class CBATools:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG Signal Processing GUI")
        self.root.geometry("1200x1000")

        self.observer = ob.Observer()
        self.interactive_plot = vo.InteractivePlot(self.observer)
        self.guiWindow = gui.guiWindow(self.root, self.observer, self.interactive_plot)

        self.whole_signal = None
        self.ecg_signal = None
        self.ap_signal = None
        self.selected_channel_signal = None 
        self.range = None

        self.fs = 250 #Change it after get fs 

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)

        self.observer.subscribe(self.return_range)
        self.observer.subscribe(self.update_labels_on_change)

        # self.observer.subscribe(calculate_variance)

        load_data_page_components = self.guiWindow.create_load_data_page(
            # notebook,
            lambda: dl.load_ecg_data(
            self,
            load_data_page_components["patient_id_entry"],
            load_data_page_components["start_entry"],
            load_data_page_components["end_entry"],
            ),
            lambda: dl.load_ap_data(
            self,
            load_data_page_components["patient_id_entry"],
            load_data_page_components["start_entry"],
            load_data_page_components["end_entry"],
            ),
            lambda: dl.load_ecg_data(
            self,
            load_data_page_components["patient_id_entry"],
            load_data_page_components["start_entry"],
            load_data_page_components["end_entry"],
            ),
            lambda: vo.vis_data(
            self,
            load_data_page_components["interactive_plot"],
            ),
            lambda: vo.updateInteract(
            load_data_page_components["interactive_plot"],
            )
        )

        signal_processing_page_components = self.guiWindow.create_signal_processing_page(
            # notebook,
            lambda: sp.filter_signal(
                self,
                signal_processing_page_components["lowcut_entry"],
                signal_processing_page_components["highcut_entry"],
            ),
            lambda: sp.artifact_process(
                self.selected_channel_signal,
                signal_processing_page_components["canvas_frame"]
            ),
            lambda: plot_signals(
            self,
            signal_processing_page_components["canvas_frame"]
            )
            
        )

        time_domain_page_components = self.guiWindow.create_time_domain_page(
            lambda: tda.calculateEcgTimeDomainValue(
                self,
                self.fs,
                time_domain_page_components["properties_frame_ecg"],
                time_domain_page_components["sd_canvas_frame"],
                self.range
            ),
            lambda: tda.calculateApTimeDomainValue(
                self,
                self.fs,
                time_domain_page_components["properties_frame_ap"],
                self.range
            )
        )


        frequency_domain_page_components = self.guiWindow.create_frequency_domain_page(
            lambda: fda.perform_frequency_analysis_psd(
                self,
                self.fs,
                frequency_domain_page_components["freq_canvas_frame"]
            )
        )
    @property
    def ecg_signal(self):
        return self._ecg_signal

    @ecg_signal.setter
    def ecg_signal(self, value):
        self._ecg_signal = value
        self.on_signal_change("ecg")

    @property
    def ap_signal(self):
        return self._ap_signal

    @ap_signal.setter
    def ap_signal(self, value):
        self._ap_signal = value
        self.on_signal_change("ap")

    def return_range(self, range):
        self.range = range
        # self.range = selection_ranges
        print("ranges:", self.range)
        # return selection_ranges

    def on_signal_change(self, signal_type):
        if signal_type == "ecg" or signal_type == "ap":
            if self._ecg_signal is not None and self._ap_signal is not None:
                # 如果两个信号都有数据，则更新绘图
                self.guiWindow.interactive_plot.plot_signals(self._ecg_signal, self._ap_signal)

    # Update labels when new data is available
    def update_labels_on_change(self, range):
        for key, (start, end) in range.items():
            print(f"Channel {key}: Start = {start}, End = {end}")

        ecg_result = range
        # ap_result = data.get("ap_result", "No Data")
        self.guiWindow.ecg_result_label.config(text=f"Window Information: {ecg_result}")
        # self.guiWindow.ap_result_label.config(text=f"AP Time Domain Result: {ap_result}")

    def update_ecg_signal(self, signal, sampling_rate):
        self.ecg_signal = signal
        self.fs = sampling_rate

    def update_ap_signal(self, signal, sampling_rate):
        self.ap_signal = signal
        self.fs = sampling_rate

if __name__ == "__main__":
    root = tk.Tk()
    app = CBATools(root)
    root.mainloop()