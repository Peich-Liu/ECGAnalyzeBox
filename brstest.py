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

def synchronize_rr_sbp(rr_times, rr_intervals, sbp_times, sbp_values):
    """
    同步RR间期和SBP序列。

    参数:
    rr_times : array
        RR间期对应的时间，单位：秒。
    rr_intervals : array
        RR间期序列，单位：毫秒。
    sbp_times : array
        SBP对应的时间，单位：秒。
    sbp_values : array
        SBP值序列，单位：mmHg。

    返回:
    rr_intervals_sync : array
        同步后的RR间期序列。
    sbp_values_sync : array
        同步后的SBP值序列。
    """
    # 使用线性插值对RR间期和SBP进行同步
    common_times = np.union1d(rr_times, sbp_times)
    rr_interp = np.interp(common_times, rr_times, rr_intervals)
    sbp_interp = np.interp(common_times, sbp_times, sbp_values)
    return rr_interp, sbp_interp
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

def synchronize_rr_sbp_cycle_based(r_peaks_times, rr_intervals, sbp_times, sbp_values):
    """
    基于心动周期的同步方法，将RR间期和SBP值同步。

    参数:
    r_peaks_times : array
        R峰的时间，单位：秒。
    rr_intervals : array
        RR间期序列，单位：毫秒。
    sbp_times : array
        SBP对应的时间，单位：秒。
    sbp_values : array
        SBP值序列，单位：mmHg。

    返回:
    rr_intervals_sync : array
        同步后的RR间期序列。
    sbp_values_sync : array
        同步后的SBP值序列。
    rr_times_sync : array
        RR间期对应的时间，单位：秒。
    """
    rr_intervals_sync = []
    sbp_values_sync = []
    rr_times_sync = []

    for i in range(len(rr_intervals)):
        # 定义心动周期的时间范围
        start_time = r_peaks_times[i]
        end_time = r_peaks_times[i+1] if i+1 < len(r_peaks_times) else r_peaks_times[i] + rr_intervals[i]/1000

        # 在该时间范围内寻找SBP峰值
        mask = (sbp_times >= start_time) & (sbp_times < end_time)
        sbp_in_cycle = sbp_values[mask]
        sbp_times_in_cycle = sbp_times[mask]

        if len(sbp_in_cycle) > 0:
            # 如果有多个SBP峰值，取第一个或平均值
            sbp_value = sbp_in_cycle[0]  # 或者使用np.mean(sbp_in_cycle)
            sbp_time = sbp_times_in_cycle[0]
            rr_intervals_sync.append(rr_intervals[i])
            sbp_values_sync.append(sbp_value)
            rr_times_sync.append((start_time + end_time) / 2)
        else:
            # # 如果没有找到SBP值，可以选择跳过或插值处理
            pass
            # rr_intervals_sync.append()


    return np.array(rr_intervals_sync), np.array(sbp_values_sync), np.array(rr_times_sync)


# 以下是完整的流程：

def main():
    # 读取数据
    fs = 250
    quality_file = r"c:\Document\sc2024\filtered_ecg_with_snr.csv"
    quality_data = pd.read_csv(quality_file)

    ecg_signal = quality_data['ecg'].values
    ecg_signal = bandpass_filter(ecg_signal, 0.5, 45, fs)

    ap_signal = quality_data['ap'].values
    ap_signal = bandpass_filter(ap_signal, 0.5, 10, fs)
    
    t = np.arange(len(ecg_signal)) / fs

    # 从ECG信号中检测R峰
    r_peaks_indices, r_peaks_times = detect_r_peaks(ecg_signal, fs)

    # 计算RR间期
    rr_intervals, rr_times = compute_rr_intervals(r_peaks_times)

    # 从AP信号中检测SBP
    sbp_values, sbp_times = detect_sbp(ap_signal, fs)

    # 同步RR间期和SBP序列
    # rr_intervals_sync, sbp_values_sync = synchronize_rr_sbp(rr_times, rr_intervals, sbp_times, sbp_values)

    # 基于心动周期的同步
    rr_intervals_sync, sbp_values_sync, rr_times_sync = synchronize_rr_sbp_cycle_based(r_peaks_times, rr_intervals, sbp_times, sbp_values)
    print("rr_intervals_sync",len(rr_intervals_sync),"ecg:",len(ecg_signal))


    # 计算BRS
    brs_sequences, mean_brs = compute_brs_sequence(rr_intervals_sync, sbp_values_sync)
    # brs_sequences, mean_brs = compute_brs_sequence(rr_intervals, sbp_values)
    # rr_intervals_sync -= np.mean(rr_intervals_sync)

    # 打印结果
    print("BRS序列：", brs_sequences)
    print("平均BRS：", mean_brs)

    # 可视化
    plt.figure(figsize=(12, 10))

    plt.subplot(4, 1, 1)
    plt.plot(t, ecg_signal, label='ECG signal')
    plt.plot(r_peaks_times, ecg_signal[r_peaks_indices], 'ro', label='R peak')
    plt.title('ECG signal and RR interval')
    plt.xlabel('Time(sample)')
    plt.ylabel('Ampulitude')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(t, ap_signal, label='AP signal')
    sbp_indices = (sbp_times * fs).astype(int)
    sbp_indices = np.clip(sbp_indices, 0, len(ap_signal) - 1)  # 防止索引越界
    plt.plot(sbp_times, ap_signal[sbp_indices], 'go', label='SBP peak')
    plt.title('AP signal and SBP')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (mmHg)')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(rr_times, rr_intervals, label='RR interval')
    plt.plot(sbp_times, sbp_values, label='SBP')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    # plt.title('同步前的RR间期和SBP序列')
    plt.title('RR and SBP before sync')


    # 绘制同步后的RR间期和SBP值，并在图中标记BRS序列
    plt.subplot(4, 1, 4)
    # sync_times = np.arange(len(rr_intervals_sync)) * (rr_times[1] - rr_times[0])
    # sync_times = range(len(rr_intervals_sync))
    plt.plot(rr_times_sync, rr_intervals_sync, label='RR interval (after sync)')
    plt.plot(rr_times_sync, sbp_values_sync, label='SBP (after sync)')
    plt.title('RR interval and SBP after sync')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')

    # 标记BRS序列
    for seq in brs_sequences:
        if abs(seq['r_value']) >= 0.85:
            start_idx = seq['start_idx']
            end_idx = seq['end_idx']
            plt.plot(rr_times_sync[start_idx:end_idx+1], rr_intervals_sync[start_idx:end_idx+1], 'r', linewidth=2)
            plt.plot(rr_times_sync[start_idx:end_idx+1], sbp_values_sync[start_idx:end_idx+1], 'g', linewidth=2)
            # plt.plot(sync_times[start_idx:end_idx+1], rr_intervals_sync[start_idx:end_idx+1], 'r', linewidth=2)
            # plt.plot(sync_times[start_idx:end_idx+1], sbp_values_sync[start_idx:end_idx+1], 'g', linewidth=2)
            # 添加斜率注释
            # plt.text(sync_times[start_idx], rr_intervals_sync[start_idx], f"Slope: {seq['slope']:.2f}", color='black')
            # plt.text(sync_times[0], sbp_values_sync[start_idx], f"SBP斜率: {seq['slope_sbp_time']:.2f}", color='blue')


    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
