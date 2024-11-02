import tkinter as tk
from tkinter import ttk
import dataLoader as dl
import signalProcessing as sp
import timeDomainAnalysis as tda
import freqDomainAnalysis as fda
import visulaztionOverview as vo
import guiController as gui
from utilities import *


class CBATools:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG Signal Processing GUI")
        self.root.geometry("1200x1000")

        self.guiWindow = gui.guiWindow(self.root)

        self.whole_signal = None
        self.ecg_signal = None
        self.ap_signal = None
        self.selected_channel_signal = None 
        self.fs = 250 #Change it after get fs 

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)

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
                time_domain_page_components["sd_canvas_frame"]

            ),
            lambda: tda.calculateApTimeDomainValue(
                self,
                self.fs,
                time_domain_page_components["properties_frame_ap"]
            )
        )


        frequency_domain_page_components = self.guiWindow.create_frequency_domain_page(
            lambda: fda.perform_frequency_analysis_psd(
                self,
                self.fs,
                frequency_domain_page_components["freq_canvas_frame"]
            )
        )
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