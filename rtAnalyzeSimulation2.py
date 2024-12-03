import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from scipy import signal
from scipy.stats import kurtosis as calc_kurtosis, skew as calc_skew
import csv
from utilities import *

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
    # 避免分母为零
    if np.max(signal) == np.min(signal):
        return np.zeros_like(signal)
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

def signalQualityEva(window,threshold_amplitude_range, zero_cross_min,zero_cross_max,peak_height,beat_length,kur_min,kur_max,ske_min,ske_max):
    # Normalize the window to [0, 1]
    window = normalize_signal(window)

    # flat line check
    amplitude_range = np.max(window) - np.min(window)
    if amplitude_range < threshold_amplitude_range:
        return "Bad"

    # pure noise check（Zero Crossing Rate (零交叉率)）
    zero_crossings = np.sum(np.diff(window > np.mean(window)) != 0)
    if zero_crossings < zero_cross_min or zero_crossings > zero_cross_max:
        return "Bad"

    # QRS detection
    peaks, _ = signal.find_peaks(window, height=peak_height, distance=beat_length)
    if len(peaks) < 2:
        return "Bad"
    
    # Kurtosis (峰度)
    kurtosis = calc_kurtosis(window)
    all_kurtosis.append(kurtosis)  # 动态记录
    if kurtosis < kur_min or kurtosis > kur_max:
        return "Bad"

    #Skewness (偏度)
    skewness = calc_skew(window)
    all_skewness.append(skewness)  
    if skewness < ske_min or skewness > ske_max:
        return "Bad"
    return "Good"

all_kurtosis = []  
all_skewness = []  
low_ecg = 0.5
high_ecg = 40
low_abp = 0.5
high_abp = 20
sos_ecg = filter2Sos(low_ecg, high_ecg)
sos_abp = filter2Sos(low_abp, high_abp)

zi_ecg = signal.sosfilt_zi(sos_ecg)
zi_abp = signal.sosfilt_zi(sos_abp)

#thresholds
threshold_amplitude_range=0.1     
zero_cross_min=5
zero_cross_max= 50
peak_height=0.6
beat_length=100

kur_min=2
kur_max= 4
ske_min=-1
ske_max=1


filePath = r"C:\Users\60427\Desktop\250 kun HR.csv"
data = pd.read_csv(filePath, sep=';')

data['ecg'] = data['ecg'].str.replace(',', '.').astype(float)
data['ecg'] = data['ecg'].fillna(0)
ecg = data['ecg'].values
ecg = ecg

data['abp[mmHg]'] = data['abp[mmHg]'].str.replace(',', '.').astype(float)
data['abp[mmHg]'] = data['abp[mmHg]'].fillna(0)
abp = data['abp[mmHg]'].values
abp = abp

window_length = 60000
overlap_length = 30000  
ecgFilteredWindow = deque(maxlen=window_length)
abpFilteredWindow = deque(maxlen=window_length)
rrInterval = deque([0] * window_length, maxlen=window_length)

output_file = r"C:\Users\60427\Desktop\filtered_ecg_with_quality333.csv"
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["sample_index", "ecg", "filtered_ecg", "filtered_abp","rr_interval", "quality"])

bad_windows = []
r_peaks = []

# filtered_ecg = bandpass_filter(ecg, low_ecg,high_ecg, 250)
# filtered_abp = bandpass_filter(abp, low_abp,high_abp, 250)
with open(output_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    
    for i in range(len(ecg)):
        print("i",i)
        # # Pre-processing
        filtered_ecg, zi_ecg = ziFilter(sos_ecg, ecg[i], zi_ecg)
        ecgFilteredWindow.append(filtered_ecg[0])
        filtered_abp, zi_abp = ziFilter(sos_abp, abp[i], zi_abp)
        abpFilteredWindow.append(filtered_abp[0])
        if(i % overlap_length == 0):
            #动态阈值, [mu-2sigma, mu+2sigma], 95%
            mean_kurtosis = np.mean(all_kurtosis)
            std_kurtosis = np.std(all_kurtosis)
            kur_min = mean_kurtosis - 2 * std_kurtosis
            kur_max = mean_kurtosis + 2 * std_kurtosis

            mean_skewness = np.mean(all_skewness)
            std_skewness = np.std(all_skewness)
            ske_min = mean_skewness - 2 * std_skewness
            ske_max = mean_skewness + 2 * std_skewness

            quality = signalQualityEva(list(ecgFilteredWindow), threshold_amplitude_range, zero_cross_min,zero_cross_max,
                                       peak_height,beat_length,kur_min,kur_max,ske_min,ske_max)

        # RR interval calculation
        ecg_window_data = np.array(ecgFilteredWindow)
        peaks, _ = signal.find_peaks(ecg_window_data, distance=200, height=np.mean(ecg_window_data) * 1.2)

        if len(peaks) > 1:
            rr_interval = np.diff(peaks)  # 计算所有相邻 R 波之间的间隔
        else:
            rr_interval = [] 

        writer.writerow([i, ecg[i], filtered_ecg[0],filtered_abp[0], rr_interval, quality])

        # 记录“Bad”窗口
        if quality == "Bad":
            bad_windows.append(i)

'''
plt.figure(figsize=(15, 6))
plt.plot(ecg, label="ECG Signal", alpha=0.7)
for start in bad_windows:
    plt.axvspan(start, start + window_length, color='red', alpha=0.3, label="Bad Window" if start == bad_windows[0] else "")

plt.title("ECG Signal with Bad Windows Highlighted")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()
'''