import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from utilities import bandpass_filter
import matplotlib.pyplot as plt

def plot_brs_sequences(t_rr, rr_intervals, sbp_values, brs_results):
    """
    绘制RR间期和SBP值，并在图中标记BRS序列的位置。

    参数：
    t_rr -- RR间期对应的时间数组
    rr_intervals -- RR间期序列（单位：毫秒）
    sbp_values -- SBP值序列
    brs_results -- compute_brs_sequences函数的输出结果
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制RR间期
    color = 'tab:blue'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('RR (ms)', color=color)
    ax1.plot(t_rr, rr_intervals, marker='o', linestyle='-', color=color, label='RR间期')
    ax1.tick_params(axis='y', labelcolor=color)

    # 绘制BRS序列在RR间期图上的标记
    for result in brs_results:
        seqs = result['brs_sequences']
        for sbp_seq, rr_seq in seqs:
            indices = []
            for i in range(len(rr_seq)):
                idx = np.where((rr_intervals == rr_seq[i]) & (sbp_values == sbp_seq[i]))[0]
                if len(idx) > 0:
                    indices.append(idx[0])
            if indices:
                ax1.plot(t_rr[indices], rr_intervals[indices], 'ro', markersize=8, label='BRS序列' if 'BRS序列' not in ax1.get_legend_handles_labels()[1] else "")

    # 创建第二个y轴，绘制SBP值
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('SBP值', color=color)
    ax2.plot(t_rr, sbp_values, marker='s', linestyle='--', color=color, label='SBP值')
    ax2.tick_params(axis='y', labelcolor=color)

    # 添加图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.title('RR间期和SBP值中的BRS序列标记')
    plt.tight_layout()
    plt.show()
def compute_brs_sequences(rr_intervals, sbp_values, window_size=1000, step_size=500):
    """
    计算RR间期序列和对应的SBP值的BRS序列。

    参数：
    rr_intervals -- RR间期序列的数组（单位：毫秒）
    sbp_values -- 对应的SBP值序列
    window_size -- 窗口大小，表示心跳数，默认10
    step_size -- 窗口滑动的步长，默认1

    返回：
    brs_results -- 包含每个窗口的BRS斜率和序列的列表
    """
    brs_results = []

    n_beats = len(rr_intervals)

    for i in range(0, n_beats - window_size + 1, step_size):
        rr_window = rr_intervals[i:i+window_size]
        sbp_window = sbp_values[i:i+window_size]

        # 计算RR间期和SBP的差分和变化方向
        rr_diff = np.diff(rr_window)
        sbp_diff = np.diff(sbp_window)

        rr_dir = np.sign(rr_diff)
        sbp_dir = np.sign(sbp_diff)

        # 找到RR和SBP同时增加或减少的位置
        same_dir = rr_dir * sbp_dir
        indices = np.where(same_dir == 1)[0]

        if len(indices) == 0:
            continue

        # 提取连续的序列
        sequences = []
        current_seq = [indices[0]]
        for idx in indices[1:]:
            if idx == current_seq[-1] + 1:
                current_seq.append(idx)
            else:
                if len(current_seq) >= 2:
                    sequences.append(current_seq)
                current_seq = [idx]
        if len(current_seq) >= 2:
            sequences.append(current_seq)

        # 计算每个序列的线性回归斜率
        brs_slopes = []
        brs_sequences = []
        for seq in sequences:
            seq_indices = np.array(seq)
            rr_seq = rr_window[seq_indices]
            sbp_seq = sbp_window[seq_indices]

            if len(rr_seq) >= 3:
                slope, intercept = np.polyfit(sbp_seq, rr_seq, 1)
                brs_slopes.append(slope)
                brs_sequences.append((sbp_seq, rr_seq))

        # 保存结果
        if brs_slopes:
            brs_results.append({
                'window_start': i,
                'window_end': i + window_size,
                'brs_slopes': brs_slopes,
                'brs_sequences': brs_sequences
            })

    return brs_results

sampling_rate = 250

# 读取数据
quality_file = r"/Users/liu/Documents/SC2024fall/filtered_ecg_with_snr.csv"
quality_data = pd.read_csv(quality_file)

ecg = quality_data['ecg'].values
filtered_ecg = quality_data['filtered_ecg'].values

ap = quality_data['ap'].values
filtered_abp = bandpass_filter(ap, 0.5, 10, sampling_rate)

# 检测ECG的R峰位置
peaks_ecg, _ = find_peaks(filtered_ecg, height=0.5, distance=sampling_rate * 0.6)
t_ecg_peaks = peaks_ecg / sampling_rate  # R峰的时间位置

# 计算RR间期（毫秒）
rr_intervals = np.diff(t_ecg_peaks) * 1000  # RR间期（毫秒）

# 检测ABP的SBP峰值位置
peaks_abp, _ = find_peaks(filtered_abp, distance=int(sampling_rate * 0.5))
t_abp_peaks = peaks_abp / sampling_rate  # SBP峰的时间位置
sbp_values = filtered_abp[peaks_abp]

# 为每个RR间期找到对应的SBP值
sbp_at_rr = []
for i in range(len(rr_intervals)):
    t_start = t_ecg_peaks[i]
    t_end = t_ecg_peaks[i+1]

    # 获取当前RR间期内的SBP值
    mask = (t_abp_peaks >= t_start) & (t_abp_peaks < t_end)
    sbp_in_interval = sbp_values[mask]

    if len(sbp_in_interval) == 0:
        # 如果没有找到SBP峰值，使用前一个值或平均值
        sbp_mean = sbp_values[i-1] if i > 0 else np.mean(sbp_values)
    else:
        sbp_mean = np.max(sbp_in_interval)  # 或者使用np.mean(sbp_in_interval)

    sbp_at_rr.append(sbp_mean)

sbp_at_rr = np.array(sbp_at_rr)

# 构建时间数组
t_ecg = np.arange(len(filtered_ecg)) / sampling_rate  # ECG信号的时间轴

# 构建RR间期方波信号
rr_signal = np.zeros_like(filtered_ecg)
for i in range(len(rr_intervals)):
    start_idx = peaks_ecg[i]
    end_idx = peaks_ecg[i+1] if i+1 < len(peaks_ecg) else len(rr_signal)
    rr_signal[start_idx:end_idx] = rr_intervals[i]

# 构建SBP方波信号
sbp_signal = np.zeros_like(filtered_abp)
for i in range(len(peaks_abp)-1):
    start_idx = peaks_abp[i]
    end_idx = peaks_abp[i+1]
    sbp_signal[start_idx:end_idx] = sbp_values[i]

# 如果最后一个峰值后还有数据
if peaks_abp[-1] < len(sbp_signal):
    sbp_signal[peaks_abp[-1]:] = sbp_values[-1]

# 调用函数计算BRS序列
brs_results = compute_brs_sequences(rr_intervals, sbp_at_rr)

# 绘制BRS序列和方波信号
plot_brs_sequences(t_ecg, rr_signal, sbp_signal, brs_results)
