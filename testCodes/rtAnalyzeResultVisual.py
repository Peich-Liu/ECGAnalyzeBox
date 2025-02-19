import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import FastICA
from utilities import *
from scipy.signal import savgol_filter
from scipy.signal import detrend
from datetime import timedelta, datetime


filePath = r"c:\Document\sc2024\filtered_ecg_with_snr.csv"
data = pd.read_csv(filePath)
ecg = data['filtered_ecg'].values
quality = data['quality'].values

#x-axis setting
sampling_rate = 250 
total_samples = len(ecg)
start_time = datetime.strptime("00:00:00", "%H:%M:%S")
time_stamps = [start_time + timedelta(seconds=i / sampling_rate) for i in range(total_samples)]

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
    bad_intervals.append((start_index, len(quality)))

# plt.figure(figsize=(12, 6))
plt.figure()

plt.plot(time_stamps, ecg, label='Filtered ECG Signal', color='blue')
# plt.show()


# 在图上标记 "bad" 区间
for start, end in bad_intervals:
    plt.axvspan(time_stamps[start], time_stamps[end], color='red', alpha=0.3)
    # 创建 "bad" 区间的图例项
    bad_patch = mpatches.Patch(color='red', alpha=0.3, label='"Bad" Interval')

    # 添加图例
    plt.legend(handles=[bad_patch], loc='upper right')
plt.show()
