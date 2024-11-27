import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from scipy import signal
import csv

def normalize_signal(signal):
    # 避免分母为零
    if np.max(signal) == np.min(signal):
        return np.zeros_like(signal)
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
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

def signalQualityEva(window, pqrst_list, threshold):
        rr = 0
        window = normalize_signal(window)
        peaks, _ = signal.find_peaks(window, distance=200, height=np.mean(window) * 1.2)
        if len(peaks) > 1:
            rr = peaks[-1] - peaks[-2]
        pqrst_list = [list(window)[max(0, peak-50):min(len(window), peak+50)] for peak in peaks]
        pqrst_list = [wave for wave in pqrst_list if len(wave) == 100]

        if len(pqrst_list) > 1:
            average_pqrst = calculate_average_pqrst(pqrst_list)
            snr = calculate_snr(pqrst_list, average_pqrst)
        else:
            snr = 0  # 若PQRST提取失败
        # 信号质量评估
        quality = "Good" if (snr > threshold) else "Bad"  # 设定10 dB为良好信号的门限
        return snr, quality, rr


low = 0.5
high = 45
sos = filter2Sos(low, high)

zi = signal.sosfilt_zi(sos)

filePath = r"C:\Document\sc2024/250 kun HR.csv"
data = pd.read_csv(filePath, sep=';')

data['ecg'] = data['ecg'].str.replace(',', '.').astype(float)
data['ecg'] = data['ecg'].fillna(0)
ecg = data['ecg'].values

data['abp[mmHg]'] = data['abp[mmHg]'].str.replace(',', '.').astype(float)
data['abp[mmHg]'] = data['abp[mmHg]'].fillna(0)
ap = data['abp[mmHg]'].values


window_length = 1000
overlap_length = 500
ecgFilteredWindow = deque(maxlen=window_length)

output_file = r"C:\Document\sc2024/filtered_ecg_with_snr333.csv"
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["sample_index", "ecg","ap", "filtered_ecg", "snr", "rr", "quality"])

pqrst_list = []  # 存储窗口内的PQRST波形
# all_snr = []
all_snr = deque(maxlen=50000)
thresholds = 10

bad_windows = []
rr = 0
with open(output_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    
    for i in range(len(ecg)):
        # 滤波处理
        filtered_ecg, zi = ziFilter(sos, ecg[i], zi)
        ecgFilteredWindow.append(filtered_ecg[0])

        # 每个窗口进行SNR计算
        if i % overlap_length == 0:
            # PQRST提取
            mean_snr = np.mean(all_snr)
            std_snr = np.std(all_snr)
            thresholds_min = max(mean_snr - 2 * std_snr, 0) 
            thresholds_max = mean_snr + 2 * std_snr

            snr, quality, rr = signalQualityEva(list(ecgFilteredWindow), pqrst_list, thresholds_min)
        all_snr.append(snr)

            # 写入结果
        writer.writerow([i, ecg[i], ap[i], filtered_ecg[0], snr, rr, quality])