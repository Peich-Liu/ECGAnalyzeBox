import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from scipy import signal
import csv

def filter2Sos(low, high, fs=1000, order=4):
    nyquist = fs / 2
    sos = signal.butter(order, [low / nyquist, high / nyquist], btype='band', output='sos')
    return sos

def ziFilter(sos, data_point, zi):
    filtered_point, zi = signal.sosfilt(sos, [data_point], zi=zi)
    return filtered_point, zi

def signalQualityEva(signalSample, signal_threshold = 1.5):
    '''
    @Gao: you can change the parameter in this function
    '''
    quality = "Good" if abs(filtered_ecg[0]) < signal_threshold else "Bad"
    return quality

low = 0.5
high = 45
sos = filter2Sos(low, high)

zi = signal.sosfilt_zi(sos)

filePath = r"C:\Document\sc2024\250 kun HR.csv"
data = pd.read_csv(filePath, sep=';')

data['ecg'] = data['ecg'].str.replace(',', '.').astype(float)
data['ecg'] = data['ecg'].fillna(0)

# 提取 ECG 信号
ecg = data['ecg'].values
ecg = ecg

# 设置窗口长度
window_length = 1000
# 初始化滑动窗口和数据队列
ecgWindow = deque(maxlen=window_length)
ecgFilteredWindow = deque(maxlen=window_length)
rrInterval = deque([0] * window_length, maxlen=window_length)

output_file = r"C:\Document\sc2024\filtered_ecg_with_quality.csv"
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["sample_index", "ecg", "filtered_ecg", "rr_interval", "quality"])


r_peaks = []

with open(output_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    
    for i in range(len(ecg)):
        #simulate the rt signal
        ecgWindow.append(ecg[i])

        #pre-processing
        filtered_ecg, zi = ziFilter(sos, ecg[i], zi)
        ecgFilteredWindow.append(filtered_ecg[0])

        # here is the signal quality check
        quality = signalQualityEva(filtered_ecg[0])

        # rr interval calculation
        ecg_window_data = np.array(ecgFilteredWindow)
        peaks, _ = signal.find_peaks(ecg_window_data, distance=200, height=np.mean(ecg_window_data) * 1.2)

        if len(peaks) > 1:
            rr_interval = peaks[-1] - peaks[-2]
        else:
            rr_interval = 0
        rrInterval.append(rr_interval)#maybe delete


        writer.writerow([i, ecg[i], filtered_ecg[0], rr_interval, quality])