import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from utilities import bandpass_filter

def compute_brs_sequences(rr_signal, sbp_signal, window_size=1000, step_size=500):
    """
    计算RR间期信号和收缩压信号的BRS序列。

    参数：
    rr_signal -- RR间期信号的数组或列表
    sbp_signal -- 收缩压信号的数组或列表
    window_size -- 窗口大小，默认1000
    step_size -- 窗口滑动的步长，默认1

    返回：
    brs_results -- 包含每个窗口的BRS斜率和序列的列表
    """
    brs_results = []

    for i in range(0, len(rr_signal) - window_size + 1, step_size):
        rr_window = rr_signal[i:i+window_size]
        sbp_window = sbp_signal[i:i+window_size]

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
            beat_indices = np.arange(seq[0], seq[-1]+2)
            rr_seq = rr_window[beat_indices]
            sbp_seq = sbp_window[beat_indices]

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

# 根据预期心率范围（50-100 bpm）计算心跳间距
min_heart_rate = 50  # bpm
max_heart_rate = 100  # bpm

# 转换为每秒心跳数
min_heartbeat_interval = 60 / max_heart_rate  # seconds
max_heartbeat_interval = 60 / min_heart_rate  # seconds

# 转换为采样点数
min_distance = int(sampling_rate * min_heartbeat_interval)
max_distance = int(sampling_rate * max_heartbeat_interval)

# 选择一个适合的值，例如最小距离
distance = min_distance

quality_file = r"/Users/liu/Documents/SC2024fall/filtered_ecg_with_snr.csv"
quality_data = pd.read_csv(quality_file)

# filtered_ecg = quality_data['filtered_ecg'].values
ecg = quality_data['ecg'].values
filtered_ecg = quality_data['filtered_ecg'].values

ap = quality_data['ap'].values
filtered_abp = bandpass_filter(ap, 0.5, 10, 250)
# filtered_ecg = bandpass_filter(ecg, 0.5, 45, 250)
# 设置峰值检测参数
peaks, _ = find_peaks(filtered_abp, distance=distance)  # 根据采样率调整 distance

# 提取 SBP 值
sbp_values = filtered_abp[peaks]

# 创建与原始信号等长的 SBP 方波信号
sbp_signal = np.zeros_like(filtered_abp)  # 初始化为零

# 将 SBP 值填充到对应的峰值位置
sbp_signal[peaks] = sbp_values

# 如果需要阶梯型方波（保持上一峰值）
for i in range(1, len(sbp_signal)):
    if sbp_signal[i] == 0:  # 如果当前位置为零
        sbp_signal[i] = sbp_signal[i - 1]  # 复制上一时刻的值

# 检测 R 波位置
peaks, _ = find_peaks(filtered_ecg, height=0.5, distance=sampling_rate * 0.6)

# 计算 RR 间期（秒）
rr_intervals = 1000 * np.diff(peaks) / sampling_rate

# 创建与原始信号等长的 RR 间期方波
rr_signal = np.zeros_like(filtered_ecg)  # 初始化为零

# 填充方波
for i in range(len(rr_intervals)):
    rr_signal[peaks[i]:peaks[i + 1]] = rr_intervals[i]  # 在两个峰值之间填充 RR 值
time = range(len(filtered_ecg))


brs_results = compute_brs_sequences(rr_signal, sbp_signal)
print(brs_results)
# sequence = find_sequences(rr_signal[0:1000], sbp_signal[0:1000])

# for i in range(0, len(filtered_ecg), 500):
#     sequence = find_sequences(rr_signal[i:i+1000], sbp_signal[i:i+1000])
#     if len(sequence) != 0:
#         # 创建两个子图
#         fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

#         # 绘制第一个子图（RR Interval）
#         axs[0].plot(1000, rr_signal[0:1000], label="RR Interval (Aligned)", linestyle="--")
#         axs[0].set_ylabel("RR Interval (ms)")
#         axs[0].set_title("RR Interval Signal")
#         axs[0].legend()
#         axs[0].set_ylim(600, 1500)

#         # 绘制第二个子图（SBP Signal）
#         axs[1].plot(1000, sbp_signal[0:1000], label="SBP Signal (Aligned)", linestyle="--")
#         axs[1].set_xlabel("Time (s)")
#         axs[1].set_ylabel("SBP")
#         axs[1].set_title("SBP Signal")
#         axs[1].legend()

#         axs[1].plot(1000, sequence, label="BRS", linestyle="--")
#         axs[1].set_xlabel("Time (s)")
#         axs[1].set_ylabel("BRS")
#         axs[1].set_title("BRS Signal")
#         axs[1].legend()
#         # 调整布局
#         plt.tight_layout()
#         plt.show()