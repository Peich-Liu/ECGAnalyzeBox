import numpy as np
from scipy.stats import linregress
from utilities import *

def preprocess_ecg(ecg, fs=250):
    """对 ECG 信号进行带通滤波预处理"""
    # 设置带通滤波器参数（0.5 - 50 Hz，适用于 ECG）
    nyquist = 0.5 * fs
    low = 0.5 / nyquist
    high = 50 / nyquist
    b, a = butter(1, [low, high], btype="band")
    ecg_filtered = filtfilt(b, a, ecg)
    return ecg_filtered

def detect_r_peaks(ecg, fs=250):
    """检测 ECG 信号中的 R 波峰值"""
    # 寻找 R 波峰值
    peaks, _ = find_peaks(ecg, distance=fs*0.6)  # 设置最小间隔为 0.6 秒（基于人类心率）
    return peaks

def calculate_rr_intervals(r_peaks, fs=250):
    """计算 RR 间隔，单位为毫秒"""
    rr_intervals = np.diff(r_peaks) / fs * 1000  # 转换为毫秒
    return rr_intervals

def resample_ap_to_rr(ap, r_peaks, fs_ap):
    """
    将 AP 信号降采样到 RR 间隔的时间点
    :param ap: 动脉压信号 (AP)
    :param r_peaks: ECG 信号中 R 波峰值的位置
    :param fs_ap: AP 信号的采样率
    :return: 对应于 RR 间隔的 AP 值
    """
    r_peaks_time = r_peaks / fs_ap  # 将 R 波位置转换为时间
    ap_resampled = [ap[int(t * fs_ap)] for t in r_peaks_time[:-1]]  # 对齐到每个 RR 间隔
    return np.array(ap_resampled)

def calculate_brs(bp, rr, threshold=3):
    # 找到 BP 和 RR 的变化
    bp_diff = np.diff(bp)
    rr_diff = np.diff(rr)

    # 识别符合序列的点
    seq_indices = np.where((bp_diff > threshold) & (rr_diff > 0))[0]

    brs_values = []
    for i in range(len(seq_indices) - 2):  # 至少需要 3 点序列
        start, end = seq_indices[i], seq_indices[i + 2]
        if end - start == 2:  # 确保序列连续
            slope, _, _, _, _ = linregress(bp[start:end+1], rr[start:end+1])
            brs_values.append(slope)

    return np.mean(brs_values) if brs_values else None



data_directory = r'c:\Document\sc2024\filtered_ecg_with_snr.csv'
ecg, ap, rr, _ = loadRtDatawithRR(data_directory)
fs = 250

ecg = preprocess_ecg(ecg)
r_peaks = detect_r_peaks(ecg, fs)

brs_value = calculate_brs(ap, rr, threshold=3)
print(brs_value)