import tkinter as tk
from tkinter import ttk
# import main_guiLayout as gl
import dataLoader as dl
import signalProcessing as sp
import timeDomainAnalysis as tda
import freqDomainAnalysis as fda


class CBATools:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG Signal Processing GUI")
        self.root.geometry("1000x800")

        self.whole_signal = None
        self.selected_channel_signal = None 
        self.fs = None 

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)

        load_data_page_components = dl.create_load_data_page(
            notebook,
            lambda: dl.load_ecg_data(
            self,
            load_data_page_components["patient_id_entry"],
            load_data_page_components["start_entry"],
            load_data_page_components["end_entry"],
            load_data_page_components["channel_selector"],
            load_data_page_components["load_data_canvas_frame"]
            ),
            lambda: dl.load_ecg_data(
            self,
            load_data_page_components["patient_id_entry"],
            load_data_page_components["start_entry"],
            load_data_page_components["end_entry"],
            load_data_page_components["channel_selector"],
            load_data_page_components["load_data_canvas_frame"]
            ),
            lambda: dl.load_ecg_data(
            self,
            load_data_page_components["patient_id_entry"],
            load_data_page_components["start_entry"],
            load_data_page_components["end_entry"],
            load_data_page_components["channel_selector"],
            load_data_page_components["load_data_canvas_frame"]
            )
        )

        visualization_page_components = tda.create_time_domain_page(
            notebook,
            lambda: fda.perform_frequency_analysis(
                self.selected_channel_signal,
                self.fs,
                time_domain_page_components["freq_canvas_frame"]
            )
        )

        signal_processing_page_components = sp.create_signal_processing_page(
            notebook,
            lambda: sp.filter_signal(
                self,
                signal_processing_page_components["lowcut_entry"],
                signal_processing_page_components["highcut_entry"],
                signal_processing_page_components["canvas_frame"]
            ),
            lambda: sp.artifact_process(
                self.selected_channel_signal,
                signal_processing_page_components["canvas_frame"]
            )
        )

        time_domain_page_components = tda.create_time_domain_page(
            notebook,
            lambda: fda.perform_frequency_analysis(
                self.selected_channel_signal,
                self.fs,
                time_domain_page_components["freq_canvas_frame"]
            )
        )


        frequency_domain_page_components = fda.create_frequency_domain_page(
            notebook,
            lambda: fda.perform_frequency_analysis(
                self.selected_channel_signal,
                self.fs,
                frequency_domain_page_components["freq_canvas_frame"]
            )
        )

    def update_whole_signal(self, signal, sampling_rate):
        self.selected_channel_signal = signal
        self.fs = sampling_rate
    def update_selected_signal(self, signal, sampling_rate):
        self.selected_channel_signal = signal
        self.fs = sampling_rate

if __name__ == "__main__":
    root = tk.Tk()
    app = CBATools(root)
    root.mainloop()