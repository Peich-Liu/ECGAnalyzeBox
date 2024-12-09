import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from utilities import *
from scipy.signal import savgol_filter
from scipy.signal import detrend
from datetime import timedelta, datetime


# filePath = r"/Users/liu/Documents/SC2024fall/filtered_ecg_with_quality333.csv"
# filePath = r"/Users/liu/Documents/SC2024fall/filtered_ecg_with_quality888.csv"
filePath = r"C:\Document\sc2024/filtered_ecg_with_quality_hospital.csv"



data = pd.read_csv(filePath)
ecg = data['ecg'].values
quality = data['quality'].values

#x-axis setting
sampling_rate = 1000
total_samples = len(ecg)
start_time = datetime.strptime("00:00:00", "%H:%M:%S")
# time_stamps = [start_time + timedelta(seconds=i / sampling_rate) for i in range(total_samples)]
time_stamps = [i / sampling_rate for i in range(total_samples)]


# 找到连续 "bad" 的区间
bad_intervals = []
is_bad = False
start_index = 0

for i in range(len(quality)):
    if quality[i] == "Bad" and not is_bad:
        is_bad = True
        start_index = i
    elif quality[i] != "Bad" and is_bad:
        is_bad = False
        bad_intervals.append((start_index, i))

if is_bad:
    bad_intervals.append((start_index, min(len(quality) - 1, len(time_stamps) - 1)))

# 找到连续 "consider" 的区间
consider_intervals = []
is_consider = False
start_consider_index = 0

for i in range(len(quality)):
    if quality[i] == "Consider" and not is_consider:
        is_consider = True
        start_consider_index = i
    elif quality[i] != "Consider" and is_consider:
        is_consider = False
        consider_intervals.append((start_consider_index, i))

if is_consider:
    consider_intervals.append((start_consider_index, min(len(quality) - 1, len(time_stamps) - 1)))


plt.figure(figsize=(12, 6))
plt.plot(time_stamps, ecg, label='Filtered ECG Signal', color='black')

# 在图上标记 "consider" 区间
for start, end in consider_intervals:
    plt.axvspan(time_stamps[start], time_stamps[end], color='blue', alpha=0.3, label='Consider Interval')
    # plt.axvspan(time_stamps[start], time_stamps[end], color='red', alpha=0.3, label='Consider Interval')


# 在图上标记 "bad" 区间
for start, end in bad_intervals:
    plt.axvspan(time_stamps[start], time_stamps[end], color='red', alpha=0.3, label='Bad Interval')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # 去重
plt.legend(by_label.values(), by_label.keys())
plt.title("Filtered ECG Signal with Quality Annotations")
plt.xlabel("Time (s)")
plt.ylabel("ECG Amplitude (mV)")
plt.show()
