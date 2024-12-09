import guiDataLoader as dl
import guiSignalProcessing as sp
import guiTimeDomainAnalysis as tda
import guiFreqDomainAnalysis as fda
import guiVisulaztionOverview as vo
import guiInterface as gui
import guiObserver as ob
import guiBrsAnalyze as ba
import guiRealtimeDataLoader as rtdl
import tkinter as tk

from tkinter import ttk
# from pylsl import StreamInfo, StreamOutlet
from utilities import *



class CBATools:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG Signal Processing GUI")
        self.root.geometry("1200x1000")

        self.whole_signal = None
        self.ecg_signal = None
        self.ap_signal = None
        self.selected_channel_signal = None 
        self.range = None
        self.filepath = r'C:\Document\sc2024/250 kun HR.csv'
        self.rtWindowLength = 1000
        self.quality = None

        self.range_min = None
        self.range_max = None

        self.fs = 250 #The fs is locked in 250Hz

        self.observer = ob.Observer(self.fs)
        self.interactive_plot = vo.InteractivePlot(self.observer)
        self.rtDataLoder = rtdl.rtPlot(root)
        self.guiWindow = gui.guiWindow(self.root, self.observer, self.interactive_plot, self.rtDataLoder)

        self.observer.subscribe(self.return_range)
        self.observer.subscribe(self.update_labels_on_change)
        # self.observer.subscribe(show_windowInfo)

        load_data_page_components = self.guiWindow.create_load_data_page(
            lambda: dl.load_data(
            self,
            ),
            lambda: dl.load_rt_data(
            self,
            ),
            lambda: vo.store_new_file(
            self,
            self.fs
            )
            # lambda: vo.updateInteract(
            # load_data_page_components["interactive_plot"],
            # )
        )

        brs_anaylze = self.guiWindow.create_brs_page(
            lambda: ba.calculate_brs(self.ecg_signal, self.ap_signal, brs_anaylze['brs_canvas_frame'], fs=self.fs),
        )

        real_time_page_components = self.guiWindow.create_real_time_page(
            lambda: self.rtDataLoder.openLoadData(),
            lambda: rtdl.simulateRtSignal(self.ecg_signal, self.ap_signal),
            # self.rtDataLoder
        )

        # signal_processing_page_components = self.guiWindow.create_signal_processing_page(
        #     # notebook,
        #     lambda: sp.filter_signal(
        #         self,
        #         signal_processing_page_components["lowcut_entry"],
        #         signal_processing_page_components["highcut_entry"],
        #     ),
        #     lambda: sp.artifact_process(
        #         self.selected_channel_signal,
        #         signal_processing_page_components["canvas_frame"]
        #     ),
        #     lambda: plot_signals(
        #     self,
        #     signal_processing_page_components["canvas_frame"]
        #     )
            
        # )

        # time_domain_page_components = self.guiWindow.create_time_domain_page(
        #     lambda: tda.calculateEcgTimeDomainValue(
        #         self,
        #         self.fs,
        #         time_domain_page_components["properties_frame_ecg"],
        #         time_domain_page_components["sd_canvas_frame"],
        #         self.range
        #     ),
        #     lambda: tda.calculateApTimeDomainValue(
        #         self,
        #         self.fs,
        #         time_domain_page_components["properties_frame_ap"],
        #         self.range
        #     )
        # )


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
        # print("return_range observer:",range)
        self.range = range

    def on_signal_change(self, signal_type):
        if signal_type == "ecg" or signal_type == "ap":
            if self._ecg_signal is not None and self._ap_signal is not None:
                # 如果两个信号都有数据，则更新绘图
                self.guiWindow.interactive_plot.plot_signals(self._ecg_signal, self._ap_signal, self.quality)

    # Update labels when new data is available
    def update_labels_on_change(self, range):
        tda.calculateApandEcgTimeDomainValue(self, self.fs, self.guiWindow.dataLoader_properties_frame, range)

    def update_ecg_signal(self, signal, sampling_rate):
        self.ecg_signal = signal
        self.fs = sampling_rate

    def update_ap_signal(self, signal, sampling_rate):
        self.ap_signal = signal
        self.fs = sampling_rate

    def update_quality(self, quality):
        # print(len(quality))
        self.quality = quality
    
    def get_signal(self):
        return self.ecg_signal, self.ap_signal


if __name__ == "__main__":
    root = tk.Tk()
    app = CBATools(root)
    root.mainloop()