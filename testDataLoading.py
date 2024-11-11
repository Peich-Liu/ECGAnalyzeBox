import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from utilities import *
from scipy.signal import savgol_filter
from scipy.signal import detrend


#### 进行 ICA 去伪迹
# def apply_ica(signal, n_components=1):
#     # Reshape 为 2D 数组，适合 FastICA 的输入格式
#     signal_2d = signal.reshape(-1, 1)
#     ica = FastICA(n_components=n_components, random_state=42)
#     sources = ica.fit_transform(signal_2d)  # 提取独立成分
#     return sources.flatten()

def apply_ica(signal, n_components=2):
    signal_2d = signal.reshape(-1, 1)
    ica = FastICA(n_components=n_components, random_state=42)
    sources = ica.fit_transform(signal_2d)
    
    # 如果 ICA 输出信号与原始信号的相关性为负，则翻转信号
    if np.corrcoef(signal, sources.flatten())[0, 1] < 0:
        sources = -sources
    return sources.flatten()

filePath = r"C:\Document\sc2024\ICM+_data.csv"
data = pd.read_csv(filePath, sep=';')

# 填充 NaN 值为 0
data['ecg'] = data['ecg'].str.replace(',', '.').astype(float)
data['ecg'] = data['ecg'].fillna(0)  # 将 NaN 填充为 0
data['abp[mmHg]'] = data['abp[mmHg]'].str.replace(',', '.').astype(float)

# extract ECG signal
ecg = data['ecg'].values
ecg = ecg[17500:18500]
onlyFilterSignal = bandpass_filter(ecg, 0.5, 45, 250)
icaSignal = apply_ica(ecg)

#Band pass filter
filteredSignal = bandpass_filter(icaSignal, 0.5, 45, 250)
filteredSignal -= np.mean(filteredSignal)

# Normalization
# filteredSignal /= np.max(np.abs(filteredSignal)) 
# icaSignal /= np.max(np.abs(ecg)) 

# 使用 Savgol 滤波器去除尖刺
window_length = 31  # 调整窗口大小以适应信号
polyorder = 2       # 多项式阶数
smoothed_icaSignal = savgol_filter(icaSignal, window_length, polyorder)

# 计算去伪影信号的差异
artifact_signal = onlyFilterSignal - smoothed_icaSignal

# 可视化处理
plt.figure(figsize=(12, 10))

# 子图 1: 原始滤波后的 ECG 信号
plt.subplot(3, 1, 1)
plt.plot(ecg, label='Original Signal')
plt.title('Original Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()

# 子图 2: ICA 去伪影后的 ECG 信号
plt.subplot(3, 1, 2)
plt.plot(onlyFilterSignal, label='Only BP Filter ECG Signal')
plt.title('ICA Processed ECG Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()

# 子图 3: 被 ICA 去除的伪影成分
plt.subplot(3, 1, 3)
plt.plot(artifact_signal, label='Extracted Artifact Signal', color='green')
plt.title('Extracted Artifact Signal (ICA)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()

# plt.tight_layout()
plt.show()


# 使用 Savgol 滤波器去除尖刺
window_length = 51  # 调整窗口大小以适应信号
polyorder = 3       # 多项式阶数
smoothed_signal = savgol_filter(filteredSignal, window_length, polyorder)

# 可视化平滑后的信号
plt.figure(figsize=(10, 6))
plt.plot(filteredSignal, label='Original Signal', alpha=0.7)
plt.plot(smoothed_signal, label='Smoothed Signal', linestyle='--')
plt.title('Signal Before and After Savgol Smoothing')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()