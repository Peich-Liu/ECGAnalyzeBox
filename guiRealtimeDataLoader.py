import pandas as pd
import numpy as np
from scipy import signal
from collections import deque

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from collections import deque
from pylsl import StreamInfo, StreamOutlet

class dataAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.window_length = 1000
        self.rt_ecg_data = []
        self.rt_ap_data = []
        self.filtered_data = deque(maxlen=self.window_length)
        self.rr_intervals = deque([0] * self.window_length, maxlen=self.window_length)

        self.low = 0.5
        self.high = 45
        self.fs = 1000
        self.sos = self.filter2Sos(self.low, self.high, self.fs)
        self.zi = signal.sosfilt_zi(self.sos)
        self.index = 0

        self.load_data()

    def filter2Sos(self, low, high, fs=1000, order=4):
        nyquist = fs / 2
        sos = signal.butter(order, [low / nyquist, high / nyquist], btype='band', output='sos')
        return sos

    def load_data(self):
        data = pd.read_csv(self.file_path, sep=';')
        data['ecg'] = data['ecg'].str.replace(',', '.').astype(float)
        data['ecg'] = data['ecg'].fillna(0)
        data['abp[mmHg]'] = data['abp[mmHg]'].str.replace(',', '.').astype(float)
        data['abp[mmHg]'] = data['abp[mmHg]'].fillna(0)

        self.rt_ecg_data = data['ecg'].values[375000:1799788]
        self.rt_ap_data = data['abp[mmHg]'].values[375000:1799788]

    def get_next_data_point(self):
        if self.index < len(self.rt_ecg_data):
            ecg_point = self.rt_ecg_data[self.index]
            self.index += 1
            filtered_ecg, self.zi = signal.sosfilt(self.sos, [ecg_point], zi=self.zi)
            self.filtered_data.append(filtered_ecg[0])
            return filtered_ecg[0]
        return None

    def get_rr_interval(self):
        if len(self.filtered_data) == self.window_length:
            ecg_window_data = np.array(self.filtered_data)
            peaks, _ = signal.find_peaks(ecg_window_data, distance=200, height=np.mean(ecg_window_data) * 1.2)
            if len(peaks) > 1:
                rr_interval = peaks[-1] - peaks[-2]
                self.rr_intervals.append(rr_interval)
                return rr_interval
        return 0