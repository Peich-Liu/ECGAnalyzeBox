import csv
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk

from tkinter import ttk
from datetime import datetime, timedelta
from scipy import signal
from collections import deque
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
from scipy.stats import kurtosis as calc_kurtosis, skew as calc_skew
from utilities import *
from tkinter import Tk
from tkinter.filedialog import askopenfilename


# class dataAnalyzer:
class rtPlot:
    def __init__(self, root):
        self.root = root
        self.start_time = "00:00:00"
        self.window_length = 1000
        self.rt_ecg_data = []
        self.rt_ap_data = []
        self.filtered_data = deque(maxlen=self.window_length)
        self.rr_intervals = deque([0] * self.window_length, maxlen=self.window_length)
        self.fig, self.ax = plt.subplots(2, 1, figsize=(10, 8))
        self.isLoading = False
        self.ecg_threshold = 1.5
        

        #filter Parameters
        self.low = 0.5
        self.high = 45
        self.fs = 1000
        self.sos = self.filter2Sos(self.low, self.high, self.fs)
        self.zi = signal.sosfilt_zi(self.sos)
        self.index = 0

        #initial the deques
        self.filtered_ecg_window = deque(maxlen=self.window_length)
        self.filtered_rr_window = deque(maxlen=self.window_length)
        self.filtered_ap_window = deque([0] * self.window_length, maxlen=self.window_length)
        self.ecgWindow = deque(maxlen=self.window_length)
        self.apWindow = deque(maxlen=self.window_length)
        self.ecgFilteredWindow = deque(maxlen=self.window_length)
        self.rrInterval = deque([0] * self.window_length, maxlen=self.window_length)

        # 在初始化中创建绘图对象，只需要创建一次
        self.line1, = self.ax[0].plot([], [], label='Filtered ECG Signal', color='b')
        self.line2, = self.ax[1].plot([], [], label='RR Intervals', color='g')
        self.ax[0].legend()
        self.ax[1].legend()

        self.index = 0

        # 初始化 CSV 文件
        self.csv_file_path = "ecg_data_records.csv"
        self.csv_file = open(self.csv_file_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # 写入 CSV 标题行
        self.csv_writer.writerow(["Timestamp", "ECG", "RR Interval", "Quality"])

    def __del__(self):
        # 关闭 CSV 文件
        if self.csv_file:
            self.csv_file.close()

    def append_to_csv(self, timestamp, ecg, rr_interval, quality):
        # 追加新记录到 CSV 文件
        self.csv_writer.writerow([timestamp, ecg, rr_interval, quality])
        # 立即刷新缓冲区，以确保数据被及时写入文件
        self.csv_file.flush()

    # def resolve_lsl_stream(self, stream_name):
    #     streams = resolve_stream('name', stream_name)
    #     return StreamInlet(streams[0])
    def filter2Sos(self, low, high, fs=1000, order=4):
        nyquist = fs / 2
        sos = signal.butter(order, [low / nyquist, high / nyquist], btype='band', output='sos')
        return sos

    def load_data(self):
        print(self.isLoading)
        if self.isLoading:
            print("rtSignalCollecting")
                # 打开文件浏览器选择文件
            Tk().withdraw()  # 隐藏主窗口
            file_path = askopenfilename(title="Please chosse the simulate file", 
                                        filetypes=[("CSV file", "*.csv"), ("all file", "*.*")])
    
            data = pd.read_csv(file_path, sep=';')
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
    
    def update_plot(self):

        filtered_ecg = self.get_next_data_point()
        filtered_ap = self.get_next_data_point()

        rr_interval = self.get_rr_interval()

        if filtered_ecg is not None:
            self.filtered_ecg_window.append(filtered_ecg)
            self.filtered_rr_window.append(rr_interval)
            self.filtered_ap_window.append(filtered_ap)

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # quality check
            quality = "Good" if abs(filtered_ecg) <= self.ecg_threshold else "Bad"

            self.append_to_csv(timestamp, filtered_ecg, rr_interval, quality)


            if len(self.filtered_ecg_window) == self.window_length:
                x = np.arange(len(self.filtered_ecg_window))
                # x = [self.start_time + timedelta(seconds=i) for i in range(len(self.filtered_ecg_window))]

                self.line1.set_data(x, np.array(self.filtered_ecg_window))
                self.line2.set_data(x, np.array(self.filtered_rr_window))

                self.ax[0].set_xlim(0, len(self.filtered_ecg_window) - 1)
                self.ax[0].relim()
                self.ax[0].autoscale_view()

                self.ax[1].set_xlim(0, len(self.filtered_rr_window) - 1)
                self.ax[1].relim()
                self.ax[1].autoscale_view()

                self.fig.canvas.draw()

        self.index += 1
        # print("self.index:", self.index)
        self.root.after(4, self.update_plot)
        

    def openLoadData(self):
        self.isLoading = True
        print("RTDataLoadingStart...")
        self.load_data()

    def closeLoadData(self):
        self.isLoading = False




# PQRST波形计算和SNR估计
def calculate_average_pqrst(pqrst_list):
    """
    计算平均PQRST波形
    """

    pqrst_array = np.array(pqrst_list)
    # print("pqrst_list",pqrst_array)
    return np.mean(pqrst_array, axis=0)

def calculate_snr(pqrst_list, average_pqrst):
    """
    计算信噪比SNR
    """
    snr_values = []
    for beat in pqrst_list:
        noise = beat - average_pqrst
        signal_power = np.mean(average_pqrst**2)
        noise_power = np.mean(noise**2)
        snr = 10 * np.log10(signal_power / noise_power)  # SNR单位为dB
        snr_values.append(snr)
    return min(snr_values)  # 返回最小SNR

def filter2Sos(low, high, fs=1000, order=4):
    nyquist = fs / 2
    sos = signal.butter(order, [low / nyquist, high / nyquist], btype='band', output='sos')
    return sos

def ziFilter(sos, data_point, zi):
    filtered_point, zi = signal.sosfilt(sos, [data_point], zi=zi)
    return filtered_point, zi

def compute_z_score(value, mean, std):
    # Compute the z-score 
    return (value - mean) / std if std > 0 else 0

def normalize_signal(signal):
    if np.max(signal) == np.min(signal):
        return np.zeros_like(signal)
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

def signalQualityEva(window, 
                        threshold_amplitude_range, 
                        zero_cross_min, zero_cross_max, 
                        peak_height, beat_length, 
                        kur_min, kur_max, 
                        ske_min, ske_max,
                        snr_min, snr_max):
    # Normalize the window to [0, 1]
    quality = "Good" 
    window = normalize_signal(window)
    # quality = "Good" if (snr > snr_min) else "Bad"  # 设定10 dB为良好信号的门限
    

    # flat line check
    amplitude_range = np.max(window) - np.min(window)
    if amplitude_range < threshold_amplitude_range:
        # return "Bad"
        print("flat line check")
        quality = "Bad"

    # pure noise check（Zero Crossing Rate (零交叉率)）
    zero_crossings = np.sum(np.diff(window > np.mean(window)) != 0)
    if zero_crossings < zero_cross_min or zero_crossings > zero_cross_max:
        quality = "Bad"
        print("Zero Crossing")
        
        # return "Bad"

    # QRS detection
    peaks, _ = signal.find_peaks(window, height=peak_height, distance=beat_length)
    if len(peaks) < 2:
        print("QRS detection")
        quality = "Bad"
        # return "Bad"
    
    # Kurtosis (峰度)
    kurtosis = calc_kurtosis(window)
    # all_kurtosis.append(kurtosis)  # 动态记录
    if kurtosis < kur_min or kurtosis > kur_max:
        print("kurtosis")
        quality = "Bad"
        # return "Bad"

    #Skewness (偏度)
    skewness = calc_skew(window)
    # all_skewness.append(skewness)  
    if skewness < ske_min or skewness > ske_max:
        print("skewness")
        quality = "Bad"

    return quality, kurtosis, skewness

def fixThreshold(window):
    threshold_amplitude_range=0.1     
    zero_cross_min=5
    zero_cross_max= 50
    peak_height=0.6
    beat_length=100

    quality = "Good" 
    window = normalize_signal(window)
    
    # flat line check
    amplitude_range = np.max(window) - np.min(window)
    if amplitude_range < threshold_amplitude_range:
        print("flat line check")
        quality = "Bad"

    # pure noise check（Zero Crossing Rate (零交叉率)）
    zero_crossings = np.sum(np.diff(window > np.mean(window)) != 0)
    if zero_crossings < zero_cross_min or zero_crossings > zero_cross_max:
        quality = "Bad"
        print("Zero Crossing")

    # QRS detection
    peaks, _ = signal.find_peaks(window, height=peak_height, distance=beat_length)
    if len(peaks) < 2:
        print("QRS detection")
        quality = "Bad"

    return quality

def dynamicThreshold(window,
                    kur_min, kur_max, 
                    ske_min, ske_max,
                    snr_min, snr_max):
    
    quality = "Good"
    #SNR calculation
    peaks_snr, _ = signal.find_peaks(window, distance=200, height=np.mean(window) * 1.2)
    pqrst_list = [list(window)[max(0, peak-50):min(len(window), peak+50)] for peak in peaks_snr]
    pqrst_list = [wave for wave in pqrst_list if len(wave) == 100]
    
    if len(pqrst_list) > 1:
        average_pqrst = calculate_average_pqrst(pqrst_list)
        snr = calculate_snr(pqrst_list, average_pqrst)
    else:
        snr = 0  # 若PQRST提取失败

    if snr < snr_min:
        print("snr")
        quality = "Consider"

    # Kurtosis (峰度) calculation
    kurtosis = calc_kurtosis(window)

    # all_kurtosis.append(kurtosis)
    if kurtosis < kur_min or kurtosis > kur_max:
        print("kurtosis")
        quality = "Consider"


    #Skewness (偏度)
    skewness = calc_skew(window)
    # all_skewness.append(skewness)  
    if skewness < ske_min or skewness > ske_max:
        print("skewness")
        quality = "Consider"

    return quality, snr, kurtosis, skewness

def simulateRtSignal(ecg, ap):
    all_kurtosis = []  
    all_skewness = []  
    all_snr = []
    low_ecg = 0.5
    high_ecg = 40
    low_abp = 0.5
    high_abp = 20
    sos_ecg = filter2Sos(low_ecg, high_ecg)
    sos_abp = filter2Sos(low_abp, high_abp)

    zi_ecg = signal.sosfilt_zi(sos_ecg)
    zi_abp = signal.sosfilt_zi(sos_abp)

    #thresholds
    kur_min=2
    kur_max= 4
    ske_min=-1
    ske_max=1

    window_length = 1000
    overlap_length = 500  
    ecgFilteredWindow = deque(maxlen=window_length)
    abpFilteredWindow = deque(maxlen=window_length)
    rrInterval = deque([0] * window_length, maxlen=window_length)
    qualityResult = "Good"

    # output_file = r"C:\Document\sc2024/filtered_ecg_with_qualitynew.csv"
    output_file = filedialog.asksaveasfilename(
    title="Choose store position",
    defaultextension=".csv",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["sample_index", "ecg", "ap", "filtered_ecg", "filtered_abp","rr_interval", "quality"])

    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        for i in range(len(ecg)):
            # print("i",i)
            # # Pre-processing
            filtered_ecg, zi_ecg = ziFilter(sos_ecg, ecg[i], zi_ecg)
            ecgFilteredWindow.append(filtered_ecg[0])
            filtered_abp, zi_abp = ziFilter(sos_abp, ap[i], zi_abp)
            abpFilteredWindow.append(filtered_abp[0])
            if(i % overlap_length == 0):
                #fix threshold
                qualityResult = fixThreshold(list(ecgFilteredWindow))
                if qualityResult == "Good":
                    #动态阈值, [mu-2sigma, mu+2sigma], 95%
                    mean_kurtosis = np.mean(all_kurtosis)
                    std_kurtosis = np.std(all_kurtosis)
                    kur_min = mean_kurtosis - 2 * std_kurtosis
                    kur_max = mean_kurtosis + 2 * std_kurtosis

                    mean_skewness = np.mean(all_skewness)
                    std_skewness = np.std(all_skewness)
                    ske_min = mean_skewness - 2 * std_skewness
                    ske_max = mean_skewness + 2 * std_skewness
                    
                    mean_snr = np.mean(all_snr)
                    std_snr = np.std(all_snr)
                    # snr_min = mean_snr - 2 * std_snr
                    snr_min = max(mean_snr - 2 * std_snr, 0)
                    snr_max = mean_snr + 2 * std_snr

                    qualityResult, snr, kurtosis, skewness = dynamicThreshold(list(ecgFilteredWindow),
                                                                    kur_min, kur_max, 
                                                                    ske_min, ske_max,
                                                                    snr_min, snr_max)
                    all_kurtosis.append(kurtosis)  # 动态记录
                    all_skewness.append(skewness)  
                    all_snr.append(snr)
            ecg_window_data = np.array(ecgFilteredWindow)
            peaks, _ = signal.find_peaks(ecg_window_data, distance=200, height=np.mean(ecg_window_data) * 1.2)

            if len(peaks) > 1:
                rr_interval = np.diff(peaks)  # 计算所有相邻 R 波之间的间隔
            else:
                rr_interval = [] 

            writer.writerow([i, ecg[i], ap[i], filtered_ecg[0],filtered_abp[0], rr_interval, qualityResult])