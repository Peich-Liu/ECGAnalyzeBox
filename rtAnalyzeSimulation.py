import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from scipy import signal
from scipy.stats import kurtosis as calc_kurtosis, skew as calc_skew
import csv

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

def signalQualityEva(window, thresholds):
    # Normalize the window to [0, 1]
    window = normalize_signal(window)  

    # flat line check
    amplitude_range = np.max(window) - np.min(window)
    if amplitude_range < thresholds["amplitude_range"]:
        return "Bad"  
    # pure noise check
    zero_crossings = np.sum(np.diff(window > np.mean(window)) != 0)
    if zero_crossings < thresholds["zero_cross_min"] or zero_crossings > thresholds["zero_cross_max"]:
        return "Bad"
    # QRS detection
    peaks, _ = signal.find_peaks(window, height=thresholds["peak_height"], distance=thresholds["beat_length"])
    if len(peaks) < 2:
        return "Bad" 

    return "Good"

low = 0.5
high = 45
sos = filter2Sos(low, high)

zi = signal.sosfilt_zi(sos)

thresholds = {
    "amplitude_range": 0.1,      
    "zero_cross_min": 5,         
    "zero_cross_max": 50,       
    "peak_height": 0.6,          
    "beat_length": 100,          
}

filePath = r"c:\Document\sc2024\250 kun HR.csv"
data = pd.read_csv(filePath, sep=';')

data['ecg'] = data['ecg'].str.replace(',', '.').astype(float)
data['ecg'] = data['ecg'].fillna(0)

ecg = data['ecg'].values
ecg = ecg

window_length = 1000
overlap_length = 500  
ecgFilteredWindow = deque(maxlen=window_length)
rrInterval = deque([0] * window_length, maxlen=window_length)

output_file = r"c:\Document\sc2024\filtered_ecg_with_quality.csv"
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["sample_index", "ecg", "filtered_ecg", "rr_interval", "quality"])


bad_windows = []
r_peaks = []

with open(output_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    
    for i in range(len(ecg)):
        print("i",i)
        # # Pre-processing
        filtered_ecg, zi = ziFilter(sos, ecg[i], zi)
        ecgFilteredWindow.append(filtered_ecg[0])
        if(i % overlap_length == 0):
            quality = signalQualityEva(list(ecgFilteredWindow), thresholds)

        # RR interval calculation
        ecg_window_data = np.array(ecgFilteredWindow)
        peaks, _ = signal.find_peaks(ecg_window_data, distance=200, height=np.mean(ecg_window_data) * 1.2)

        if len(peaks) > 1:
            rr_interval = peaks[-1] - peaks[-2]
        else:
            rr_interval = 0
        rrInterval.append(rr_interval)

        writer.writerow([i, ecg[i], filtered_ecg[0], rr_interval, quality])

        # 记录“Bad”窗口
        if quality == "Bad":
            bad_windows.append(i)

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