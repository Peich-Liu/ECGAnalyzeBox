import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_sequences(rr_intervals, sbp_values, min_sequence_length=3):
    rr_intervals = np.array(rr_intervals)
    sbp_values = np.array(sbp_values)

    diff_rr = np.diff(rr_intervals)
    diff_sbp = np.diff(sbp_values)
    # 同步上升或下降的序列
    up_sequence = (diff_rr > 0) & (diff_sbp > 0)
    down_sequence = (diff_rr < 0) & (diff_sbp < 0)

    valid_indices = np.where(up_sequence | down_sequence)[0]
    # 提取符合条件的序列
    sequences = []
    temp_sequence = [valid_indices[0]] if len(valid_indices) > 0 else []

    for i in range(1, len(valid_indices)):
        if valid_indices[i] == valid_indices[i - 1] + 1:
            temp_sequence.append(valid_indices[i])
        else:
            if len(temp_sequence) >= min_sequence_length:
                sequences.append(temp_sequence)
            temp_sequence = [valid_indices[i]]
    if len(temp_sequence) >= min_sequence_length:
        sequences.append(temp_sequence)
    
    #计算slope
    for seq in sequences:
        rr_seq = rr_intervals[seq]
        sbp_seq = sbp_values[seq]
        slope, _ = np.polyfit(sbp_seq, rr_seq, 1)
    return sequences

quality_file = r"C:\Users\60427\Desktop\filtered_ecg_with_quality333.csv"
quality_data = pd.read_csv(quality_file)

quality_data['rr_interval'] = quality_data['rr_interval'].apply(
    lambda x: list(map(int, re.findall(r"\d+", x))) if isinstance(x, str) and "[" in x else []
)

filtered_ecg = quality_data['filtered_ecg'].values
filtered_abp = quality_data['filtered_abp'].values
rr_intervals = quality_data['rr_interval'].values

print("Sample RR Intervals:")
print(rr_intervals[1000])
# 选取第 i 个窗口
i=10
row = quality_data.iloc[i]
start_index = int(row['sample_index'])

window_ecg = filtered_ecg[start_index:start_index + 1000]
window_abp = filtered_abp[start_index:start_index + 1000]
rr_intervals = rr_intervals[start_index]
#补全RRI

# 检查 RR 和 SBP 数据长度
sbp = window_abp[:len(rr_intervals)]  
sequences = find_sequences(rr_intervals, sbp)
print(sequences)

fig, axes = plt.subplots(3, 1, figsize=(12, 10))
#ECG 
axes[0].plot(window_ecg, label="ECG", color="blue")
axes[0].set_title(f"ECG Signal (Window {i})")
axes[0].set_ylabel("Amplitude")
axes[0].legend()
axes[0].grid()

#ABP
axes[1].plot(window_abp, label="ABP", color="orange")
axes[1].set_title(f"ABP Signal (Window {i})")
axes[1].set_ylabel("Pressure (mmHg)")
axes[1].legend()
axes[1].grid()

#拟合 RR-SBP
for seq in sequences:
    rr_seq = rr_intervals[seq]
    sbp_seq = sbp[seq]
    slope, intercept = np.polyfit(sbp_seq, rr_seq, 1)

    axes[2].scatter(sbp_seq, rr_seq, label=f"Sequence {sequences.index(seq) + 1}")
    axes[2].plot(sbp_seq, slope * sbp_seq + intercept, label=f"Slope: {slope:.2f}")

axes[2].set_title("RR vs SBP (Sequence Method)")
axes[2].set_xlabel("SBP (mmHg)")
axes[2].set_ylabel("RR (ms)")
axes[2].legend()
axes[2].grid()

plt.tight_layout()
plt.show()