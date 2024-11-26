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
# from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream

# class dataAnalyzer:
class rtPlot:
    def __init__(self, file_path, root):
        self.file_path = file_path
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

                # This is the version which the axis can move

                # # 从 LSL 接收 ECG 数据
                # ecg_sample, _ = self.ecg_inlet.pull_sample(timeout=0.0)
                # ap_sample, _ = self.ap_inlet.pull_sample(timeout=0.0)

                # print("ecg_sample,",ecg_sample)

                # if ecg_sample:
                #     self.ecgWindow.append(ecg_sample[0])
                # if ap_sample:
                #     self.apWindow.append(ap_sample[0])


                # x = np.arange(self.index - self.window_length + 1, self.index + 1)
                # x = [self.start_time + timedelta(seconds=i) for i in range(len(self.filtered_ecg_window))]

                # # 更新 line 数据
                # self.line1.set_data(x, np.array(self.filtered_ecg_window))
                # self.line2.set_data(x, np.array(self.filtered_rr_window))

                # # 更新坐标轴的范围以实现移动效果
                # self.ax[0].set_xlim(x[0], x[-1])
                # self.ax[1].set_xlim(x[0], x[-1])

                # # 设置坐标轴标签
                # self.ax[0].set_xlabel("time (sample)")
                # self.ax[1].set_xlabel("time (sample)")
                # self.ax[0].set_ylabel("ecg (mv)")
                # self.ax[1].set_ylabel("rr interval (ms)")

                # # 调整坐标轴以适应数据
                # self.ax[0].relim()
                # self.ax[0].autoscale_view()
                # self.ax[1].relim()
                # self.ax[1].autoscale_view()


                # self.line1.set_data(x, np.array(self.filtered_ecg_window))
                # self.line2.set_data(x, np.array(self.filtered_rr_window))

                # self.ax[0].set_xlim(0, len(self.filtered_ecg_window) - 1)
                # self.ax[0].relim()
                # self.ax[0].autoscale_view()

                # self.ax[1].set_xlim(0, len(self.filtered_rr_window) - 1)
                # self.ax[1].relim()
                # self.ax[1].autoscale_view()


        self.index += 1
        # print("self.index:", self.index)
        self.root.after(4, self.update_plot)
        

    def openLoadData(self):
        self.isLoading = True
        print("RTDataLoadingStart...")
        self.load_data()

    def closeLoadData(self):
        self.isLoading = False