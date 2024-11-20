import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from scipy import signal
import csv

# 滤波器函数
def filter2Sos(low, high, fs=1000, order=4):
    nyquist = fs / 2
    sos = signal.butter(order, [low / nyquist, high / nyquist], btype='band', output='sos')
    return sos

def ziFilter(sos, data_point, zi):
    filtered_point, zi = signal.sosfilt(sos, [data_point], zi=zi)
    return filtered_point, zi

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

low = 0.5
high = 45
sos = filter2Sos(low, high)

zi = signal.sosfilt_zi(sos)

filePath = r"c:\Document\sc2024\250 kun HR.csv"
data = pd.read_csv(filePath, sep=';')

data['ecg'] = data['ecg'].str.replace(',', '.').astype(float)
data['ecg'] = data['ecg'].fillna(0)

ecg = data['ecg'].values

window_length = 1000
overlap_length = 500
ecgFilteredWindow = deque(maxlen=window_length)

output_file = r"c:\Document\sc2024\filtered_ecg_with_snr.csv"
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["sample_index", "ecg", "filtered_ecg", "snr", "quality"])

pqrst_list = []  # 存储窗口内的PQRST波形

bad_windows = []

with open(output_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    
    for i in range(len(ecg)):
        # 滤波处理
        filtered_ecg, zi = ziFilter(sos, ecg[i], zi)
        ecgFilteredWindow.append(filtered_ecg[0])

        # 每个窗口进行SNR计算
        if i % overlap_length == 0:
            # PQRST提取
            peaks, _ = signal.find_peaks(ecgFilteredWindow, distance=200, height=np.mean(ecgFilteredWindow) * 1.2)
            pqrst_list = [list(ecgFilteredWindow)[max(0, peak-50):min(len(ecgFilteredWindow), peak+50)] for peak in peaks]
            pqrst_list = [wave for wave in pqrst_list if len(wave) == 100]
            # print("pqrst_list",pqrst_list)

            # pqrst_list = [ecgFilteredWindow[max(0, peak-50):min(len(ecgFilteredWindow), peak+50)] for peak in peaks]

            if len(pqrst_list) > 1:
                average_pqrst = calculate_average_pqrst(pqrst_list)
                snr = calculate_snr(pqrst_list, average_pqrst)
            else:
                snr = 0  # 若PQRST提取失败

            # 信号质量评估
        quality = "Good" if snr > 10 else "Bad"  # 设定10 dB为良好信号的门限

            # 写入结果
        writer.writerow([i, ecg[i], filtered_ecg[0], snr, quality])

        # 记录“Bad”窗口
        if quality == "Bad":
            bad_windows.append(i)

# 绘图
plt.figure(figsize=(15, 6))
plt.plot(ecg, label="ECG Signal", alpha=0.7)
for start in bad_windows:
    plt.axvspan(start, start + window_length, color='red', alpha=0.3, label="Bad Window" if start == bad_windows[0] else "")

plt.title("ECG Signal with SNR-Based Bad Windows Highlighted")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()
