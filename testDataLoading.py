import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from utilities import *
from scipy.signal import savgol_filter
# 进行 ICA 去伪迹
def apply_ica(signal, n_components=1):
    # Reshape 为 2D 数组，适合 FastICA 的输入格式
    signal_2d = signal.reshape(-1, 1)
    ica = FastICA(n_components=n_components, random_state=42)
    sources = ica.fit_transform(signal_2d)  # 提取独立成分
    return sources

filePath = r"C:\Document\sc2024\ICM+_data.csv"
data = pd.read_csv(filePath, sep=';')


# 查找列名中包含 'ecg' 的列
# ecg_columns = [col for col in data.columns if 'ecg' in col.lower()]
# ecg_column = ecg_columns[0] if ecg_columns else None

# # print(data['ecg'])
# data['ecg'] = data['ecg'].str.replace(',', '.').astype(float)
# data['abp[mmHg]'] = data['abp[mmHg]'].str.replace(',', '.').astype(float)

# ecg = data['ecg'][0:50000]
# ecg -= np.mean(ecg)

# apData = data['abp[mmHg]']
# filteredAp = bandpass_filter(apData, 0.5, 50, 250)

# filteredSignal = bandpass_filter(ecg, 0.5, 50, 250)
# filteredSignal -= np.mean(filteredSignal)

# 填充 NaN 值为 0
data['ecg'] = data['ecg'].str.replace(',', '.').astype(float)
data['ecg'] = data['ecg'].fillna(0)  # 将 NaN 填充为 0
data['abp[mmHg]'] = data['abp[mmHg]'].str.replace(',', '.').astype(float)

# 提取 ECG 和 ABP 数据
ecg = data['ecg']
ecg -= np.mean(ecg)

# apData = data['abp[mmHg]']
# filteredAp = bandpass_filter(apData, 0.5, 50, 250)

# 对 ECG 信号进行带通滤波
filteredSignal = bandpass_filter(ecg, 0.5, 45, 250)
filteredSignal -= np.mean(filteredSignal)

smoothed_ecg = savgol_filter(filteredSignal, window_length=11, polyorder=3)

# filteredSignal = apply_ica(filteredSignal)

# artifSignal = derivative_filter(filteredSignal)

# 可视化 ECG 信号
plt.figure(figsize=(12, 6))
plt.plot(filteredSignal, label='Filtered ECG Signal')
plt.title('Filtered ECG Signal Visualization')
plt.xlabel('Time')
plt.ylabel('Amplitude')
# plt.ylim(-0.5,0.6)
plt.legend()
plt.show()

# plt.figure(figsize=(12, 6))
# plt.plot(filteredSignal, label='ECG Signal')
# plt.title('ECG Signal Visualization')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.show()

