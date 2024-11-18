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
    #Compute the z-score 
    return (value - mean) / std if std > 0 else 0


def signalQualityEva(window, stats, thresholds):
    if np.isnan(window).any() or np.isinf(window).any():
        return "Bad"

    if np.std(window) < 1e-10:  # 标准差过小，说明信号几乎恒定
        return "Bad"

    # 特征计算
    signal_mean = np.mean(window)
    signal_std = np.std(window)
    snr = 10 * np.log10(signal_mean / signal_std) if signal_std > 1e-10 else 0

    if np.std(window) < 1e-10:  # 防止精度损失
        kurtosis = 0
        skewness = 0
    else:
        kurtosis = calc_kurtosis(window)
        skewness = calc_skew(window)

    template = np.sin(np.linspace(0, 2 * np.pi, len(window)))
    if np.std(window) < 1e-10:  # 防止相关性矩阵中出现 NaN
        correlation = 0
    else:
        correlation = np.corrcoef(window, template)[0, 1]

    # 阈值筛选
    if not (
        correlation >= thresholds['corr'] and
        snr >= thresholds['snr'] and
        thresholds['kur_min'] <= kurtosis <= thresholds['kur_max'] and
        thresholds['ske_min'] <= skewness <= thresholds['ske_max']
    ):
        return "Bad"

    # Z-test
    stats_means = stats['means']
    stats_stds = stats['stds']
    z_scores = {
        "snr": compute_z_score(snr, stats_means['snr'], stats_stds['snr']),
        "correlation": compute_z_score(correlation, stats_means['correlation'], stats_stds['correlation']),
        "kurtosis": compute_z_score(kurtosis, stats_means['kurtosis'], stats_stds['kurtosis']),
        "skewness": compute_z_score(skewness, stats_means['skewness'], stats_stds['skewness']),
    }

    if any(abs(z) > 1.96 for z in z_scores.values()):
        return "Bad"

    return "Good"



low = 0.5
high = 45
sos = filter2Sos(low, high)

zi = signal.sosfilt_zi(sos)

stats = {
    "means": {"snr": 15, "correlation": 0.8, "kurtosis": 3, "skewness": 0},
    "stds": {"snr": 3, "correlation": 0.1, "kurtosis": 0.5, "skewness": 0.2}
}

thresholds = {
    "corr": 0.7,
    "snr": 10,
    "kur_min": 2,
    "kur_max": 4,
    "ske_min": -1,
    "ske_max": 1
}
filePath = r"C:\Users\60427\Desktop\250 kun HR.csv"
data = pd.read_csv(filePath, sep=';')

data['ecg'] = data['ecg'].str.replace(',', '.').astype(float)
data['ecg'] = data['ecg'].fillna(0)

# 提取 ECG 信号
ecg = data['ecg'].values
ecg = ecg

window_length = 1000
overlap_length = 500  
ecgWindow = deque(maxlen=window_length)
ecgFilteredWindow = deque(maxlen=window_length)
rrInterval = deque([0] * window_length, maxlen=window_length)

output_file = r"C:\Users\60427\Desktop\filtered_ecg_with_quality.csv"
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["sample_index", "ecg", "filtered_ecg", "rr_interval", "quality"])


r_peaks = []

with open(output_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    
    step_count = 0  # counter
    for i in range(len(ecg)):
        # simulate the rt signal
        # ecg[i] --> sample point
        # ecgWindow.append(ecg[i])

        #pre-processing
        filtered_ecg, zi = ziFilter(sos, ecg[i], zi)
        ecgFilteredWindow.append(filtered_ecg[0])


        quality = signalQualityEva(list(ecgFilteredWindow), stats, thresholds)

        # rr interval calculation
        ecg_window_data = np.array(ecgFilteredWindow)
        peaks, _ = signal.find_peaks(ecg_window_data, distance=200, height=np.mean(ecg_window_data) * 1.2)

        if len(peaks) > 1:
            rr_interval = peaks[-1] - peaks[-2]
        else:
            rr_interval = 0
        rrInterval.append(rr_interval)  # maybe delete

        writer.writerow([i, ecg[i], filtered_ecg[0], rr_interval, quality])

    step_count += 1
    if step_count >= (window_length - overlap_length):  # 如果到达滑动步长
        for _ in range(overlap_length):  # 按 overlap 长度移除旧数据
            ecgFilteredWindow.popleft()
        step_count = 0  # 重置计数器
