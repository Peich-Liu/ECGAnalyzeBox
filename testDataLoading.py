import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from utilities import *
from scipy.signal import savgol_filter
from scipy.signal import detrend
from datetime import timedelta, datetime

def apply_ica(signal, n_components=2):
    signal_2d = signal.reshape(-1, 1)
    ica = FastICA(n_components=n_components, random_state=42)
    sources = ica.fit_transform(signal_2d)
    
    # 如果 ICA 输出信号与原始信号的相关性为负，则翻转信号
    if np.corrcoef(signal, sources.flatten())[0, 1] < 0:
        sources = -sources
    return sources.flatten()

filePath = r"C:\Document\sc2024\250 kun HR.csv"
data = pd.read_csv(filePath, sep=';')

data['ecg'] = data['ecg'].str.replace(',', '.').astype(float)
data['ecg'] = data['ecg'].fillna(0)  # 将 NaN 填充为 0
data['abp[mmHg]'] = data['abp[mmHg]'].str.replace(',', '.').astype(float)

# extract ECG signal
ecg = data['ecg'].values
ap = data['abp[mmHg]'].values

ecg = ecg
ecgFilteredSignal = bandpass_filter(ecg, 0.5, 45, 250)
apFilteredSignal = bandpass_filter(ap, 0.5, 30, 250)


#ICA
# icaSignal = apply_ica(ecg)
# filteredSignal = bandpass_filter(icaSignal, 0.5, 45, 250)
# filteredSignal -= np.mean(filteredSignal)

# Normalization
# filteredSignal /= np.max(np.abs(filteredSignal)) 
# icaSignal /= np.max(np.abs(ecg)) 

# 示例 ECG 信号和采样率（每秒采集 1000 个样本）
sampling_rate = 250  # Hz
total_samples = len(ecg)

# 生成时间序列 (以起始时间为 00:00:00)
start_time = datetime.strptime("00:00:00", "%H:%M:%S")
time_stamps = [start_time + timedelta(seconds=i / sampling_rate) for i in range(total_samples)]


# 设置伪迹检测的阈值
artifact_threshold_ap = 50  # 你可以根据具体情况调整阈值
artifact_threshold_ecg = 1.6  # 你可以根据具体情况调整阈值

# 标记伪迹的位置（True 为伪迹点）
ap_artifacts = np.abs(apFilteredSignal) > artifact_threshold_ap
ecg_artifacts = np.abs(ecgFilteredSignal) > artifact_threshold_ecg

# 可视化处理
plt.figure(figsize=(12, 10))

# 子图 1: 原始滤波后的 AP 信号
plt.subplot(2, 1, 1)
plt.plot(time_stamps, apFilteredSignal, label='Filtered AP Signal', color='blue')
plt.plot(np.array(time_stamps)[ap_artifacts], np.array(apFilteredSignal)[ap_artifacts], 'r.', label='Artifact', markersize=4)
plt.title('Filtered AP Signal')
plt.xlabel('Time (HH:MM:SS)')
plt.ylabel('Amplitude')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
plt.legend()

# 子图 2: ICA 去伪影后的 ECG 信号
plt.subplot(2, 1, 2)
plt.plot(time_stamps, ecgFilteredSignal, label='Filtered ECG Signal', color='blue')
plt.plot(np.array(time_stamps)[ecg_artifacts], np.array(ecgFilteredSignal)[ecg_artifacts], 'r.', label='Artifact', markersize=4)
plt.title('Filtered ECG Signal')
plt.xlabel('Time (HH:MM:SS)')
plt.ylabel('Amplitude')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
plt.legend()

plt.tight_layout()
plt.show()
