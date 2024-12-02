import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy import stats
import matplotlib.pyplot as plt
from utilities import bandpass_filter

def detect_r_peaks(ecg_signal, fs):
    """
    从ECG信号中检测R峰。

    参数:
    ecg_signal : array-like
        ECG信号数据。
    fs : float
        采样频率，单位：Hz。

    返回:
    r_peaks_indices : array
        R峰的位置索引。
    r_peaks_times : array
        R峰的时间，单位：秒。
    """
    # 滤波（带通滤波器，移除低频和高频噪声）
    sos = signal.butter(2, [0.5, 15], btype='bandpass', fs=fs, output='sos')
    # filtered_ecg = signal.sosfiltfilt(sos, ecg_signal)
    filtered_ecg = ecg_signal

    # 寻找R峰
    distance = int(0.2 * fs)  # 假设心率不低于每分钟50次，两个R峰之间的最小距离为0.2秒
    # r_peaks_indices, _ = signal.find_peaks(filtered_ecg, distance=distance, height=np.mean(filtered_ecg))
    r_peaks_indices, _ = signal.find_peaks(filtered_ecg, distance=200, height=np.mean(filtered_ecg) * 1.2)

    r_peaks_times = r_peaks_indices / fs
    return r_peaks_indices, r_peaks_times

def compute_rr_intervals(r_peaks_times):
    """
    计算RR间期。

    参数:
    r_peaks_times : array
        R峰的时间，单位：秒。

    返回:
    rr_intervals : array
        RR间期序列，单位：毫秒。
    rr_times : array
        RR间期对应的时间（取两个R峰的中点），单位：秒。
    """
    rr_intervals = np.diff(r_peaks_times) * 1000  # 转换为毫秒
    rr_times = r_peaks_times[:-1] + np.diff(r_peaks_times) / 2
    return rr_intervals, rr_times

def detect_sbp(ap_signal, fs):
    """
    从AP信号中检测收缩压（SBP）。

    参数:
    ap_signal : array-like
        动脉压力信号数据。
    fs : float
        采样频率，单位：Hz。

    返回:
    sbp_values : array
        收缩压值序列，单位：mmHg。
    sbp_times : array
        收缩压对应的时间，单位：秒。
    """
    # 寻找收缩压峰值
    distance = int(0.2 * fs)  # 假设心率不低于每分钟50次
    sbp_indices, _ = signal.find_peaks(ap_signal, distance=200)
    sbp_values = ap_signal[sbp_indices]
    sbp_times = sbp_indices / fs
    return sbp_values, sbp_times

def compute_brs_sequence(rr_intervals, sbp_values):
    """
    使用序列法计算BRS。

    参数:
    rr_intervals : array-like
        RR间期时间序列（单位：毫秒）
    sbp_values : array-like
        收缩压时间序列（单位：mmHg）

    返回:
    brs_sequences : list of dictionaries
        包含每个序列的信息：斜率、相关系数、起始索引、结束索引。
    mean_brs : float
        BRS平均值（斜率的平均值）
    """
    rr_intervals = np.asarray(rr_intervals)
    sbp_values = np.asarray(sbp_values)
    n = len(rr_intervals)

    # 初始化变量
    brs_sequences = []

    # 遍历信号
    i = 0
    while i < n - 2:
        # 检查上升序列（RR和SBP同时增加）
        if sbp_values[i+1] > sbp_values[i] and rr_intervals[i+1] > rr_intervals[i]:
            # 上升序列的起点
            seq_sbp = [sbp_values[i], sbp_values[i+1]]
            seq_rr = [rr_intervals[i], rr_intervals[i+1]]
            start_idx = i
            j = i + 2
            while j < n and sbp_values[j] > sbp_values[j-1] and rr_intervals[j] > rr_intervals[j-1]:
                seq_sbp.append(sbp_values[j])
                seq_rr.append(rr_intervals[j])
                j += 1
            end_idx = j - 1
            if len(seq_sbp) >= 3:
                # 线性回归
                slope, intercept, r_value, p_value, std_err = stats.linregress(seq_sbp, seq_rr)
                brs_sequences.append({
                    'slope': slope,
                    'r_value': r_value,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
            i = end_idx  # 从序列的末尾继续
        # 检查下降序列（RR和SBP同时减少）
        elif sbp_values[i+1] < sbp_values[i] and rr_intervals[i+1] < rr_intervals[i]:
            # 下降序列的起点
            seq_sbp = [sbp_values[i], sbp_values[i+1]]
            seq_rr = [rr_intervals[i], rr_intervals[i+1]]
            start_idx = i
            j = i + 2
            while j < n and sbp_values[j] < sbp_values[j-1] and rr_intervals[j] < rr_intervals[j-1]:
                seq_sbp.append(sbp_values[j])
                seq_rr.append(rr_intervals[j])
                j += 1
            end_idx = j -1
            if len(seq_sbp) >= 3:
                # 线性回归
                slope, intercept, r_value, p_value, std_err = stats.linregress(seq_sbp, seq_rr)
                brs_sequences.append({
                    'slope': slope,
                    'r_value': r_value,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
            i = end_idx  # 从序列的末尾继续
        else:
            i +=1

    # 计算平均BRS（相关系数绝对值大于等于0.85的序列的斜率平均值）
    slopes = [seq['slope'] for seq in brs_sequences if abs(seq['r_value']) >= 0.85]
    if slopes:
        mean_brs = np.mean(slopes)
    else:
        mean_brs = np.nan  # 如果没有找到有效的序列，返回NaN

    return brs_sequences, mean_brs

def main():
    # 读取数据
    fs = 250
    quality_file = r"c:\Document\sc2024\filtered_ecg_with_snr.csv"
    quality_data = pd.read_csv(quality_file)

    ecg_signal = quality_data['ecg'].values
    ap_signal = quality_data['ap'].values

    for i in range(0, len(ecg_signal), 100000):
        print(i)

        filtered_ecg_signal = bandpass_filter(ecg_signal[i:i+100000], 0.5, 45, fs)
        filtered_ap_signal = bandpass_filter(ap_signal[i:i+100000], 0.5, 10, fs)
        t = np.arange(len(filtered_ecg_signal)) / fs


        # 从ECG信号中检测R峰
        r_peaks_indices, r_peaks_times = detect_r_peaks(filtered_ecg_signal, fs)

        # 计算RR间期
        rr_intervals, rr_times = compute_rr_intervals(r_peaks_times)

        # 从AP信号中检测SBP
        sbp_values, sbp_times = detect_sbp(filtered_ap_signal, fs)

        brs_sequences, mean_brs = compute_brs_sequence(rr_intervals, sbp_values)
        rr_intervals -= np.mean(rr_intervals)

        # 打印结果
        # print("BRS序列：", brs_sequences)
        print("平均BRS：", mean_brs)

        # 可视化
        plt.figure(figsize=(12, 10))

        plt.subplot(3, 1, 1)
        plt.plot(t, filtered_ecg_signal, label='ECG signal')
        plt.plot(r_peaks_times, filtered_ecg_signal[r_peaks_indices], 'ro', label='R peak')
        plt.title('ECG signal and RR interval')
        plt.xlabel('Time(sample)')
        plt.ylabel('Ampulitude')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(t, filtered_ap_signal, label='AP signal')
        sbp_indices = (sbp_times * fs).astype(int)
        sbp_indices = np.clip(sbp_indices, 0, len(ap_signal) - 1)  # 防止索引越界
        plt.plot(sbp_times, filtered_ap_signal[sbp_indices], 'go', label='SBP peak')
        plt.title('AP signal and SBP')
        plt.xlabel('Time (s)')
        plt.ylabel('Pressure (mmHg)')
        plt.legend()

        # 绘制同步后的RR间期和SBP值，并在图中标记BRS序列
        plt.subplot(3, 1, 3)
        # sync_times = np.arange(len(rr_intervals_sync)) * (rr_times[1] - rr_times[0])
        plt.plot(rr_times, rr_intervals, label='RR interval')
        plt.plot(sbp_times, sbp_values, label='SBP')
        plt.title('RR interval and SBP after sync')
        # plt.ylim(-100,100)
        plt.xlabel('Time (s)')
        plt.ylabel('Value')

        # 标记BRS序列
        for seq in brs_sequences:
            if abs(seq['r_value']) >= 0.85:
                start_idx = seq['start_idx']
                end_idx = seq['end_idx']
                plt.plot(rr_times[start_idx:end_idx+1], rr_intervals[start_idx:end_idx+1], 'r', linewidth=2)
                plt.plot(sbp_times[start_idx:end_idx+1], sbp_values[start_idx:end_idx+1], 'g', linewidth=2)
                # 添加斜率注释
                plt.text(sbp_times[start_idx], rr_intervals[start_idx], f"Slope: {seq['slope']:.2f}", color='black')
                # plt.text(rr_times[0], rr_intervals[start_idx], f"SBP斜率: {seq['slope_sbp_time']:.2f}", color='blue')


        plt.legend()
        plt.tight_layout()
        plt.show()

        print("rr_times",rr_times,"sbp_times",sbp_times)


if __name__ == "__main__":
    main()