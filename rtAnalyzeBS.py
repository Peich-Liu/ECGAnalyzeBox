import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# BSA using sequence method
def calculate_brs_with_mismatches(rr_intervals, sbp_values, min_sequence=3):
    if len(rr_intervals) < min_sequence or len(sbp_values) < min_sequence:
        return 0, []  # 返回默认值和空索引

    rr_intervals = np.array(rr_intervals)
    sbp_values = np.array(sbp_values)

    # Detect sequences of increasing or decreasing values
    diff_rr = np.diff(rr_intervals)
    diff_sbp = np.diff(sbp_values)

    up_sequences = (diff_rr > 0) & (diff_sbp > 0)
    down_sequences = (diff_rr < 0) & (diff_sbp < 0)
    mismatched_indices = np.where((diff_rr > 0) & (diff_sbp < 0) | (diff_rr < 0) & (diff_sbp > 0))[0]

    valid_indices = np.where(up_sequences | down_sequences)[0]
    slopes = []
    for idx in valid_indices:
        if idx + min_sequence - 1 < len(rr_intervals):
            rr_seq = rr_intervals[idx:idx + min_sequence]
            sbp_seq = sbp_values[idx:idx + min_sequence]
            slope, _ = np.polyfit(sbp_seq, rr_seq, 1)
            slopes.append(slope)
    return np.mean(slopes) if slopes else 0, mismatched_indices

quality_file = r"C:\Users\60427\Desktop\filtered_ecg_with_quality.csv"
bp_file = r"C:\Users\60427\Desktop\250 kun HR.csv"  

quality_data = pd.read_csv(quality_file)
filtered_ecg = quality_data['filtered_ecg'].values  
filtered_abp = quality_data['filtered_abp'].values

'''
print(filtered_ecg[555000:555020])
print(filtered_abp[555000:555020])
'''
# Parse rr_interval as numpy arrays
quality_data['rr_interval'] = quality_data['rr_interval'].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=" ") if isinstance(x, str) else x
)

# Analyze BRS and mismatched points
window_length = 1000
brs_values = []
mismatch_points = []

for _, row in quality_data.iterrows():
    if row['quality'] == "Good":
        try:
            rri = row['rr_interval']

            # Check RRI length
            if len(rri) < 3:
                brs_values.append(0)
                continue

            # Ensure ABP data is sufficient
            start_index = int(row['sample_index'])
            if start_index + window_length > len(filtered_abp):
                brs_values.append(0)
                continue

            sbp = filtered_abp[start_index:start_index + len(rri)]
            brs, mismatches = calculate_brs_with_mismatches(rri, sbp)
            brs_values.append(brs)

            # Adjust mismatched indices to global
            global_mismatches = [start_index + i for i in mismatches]
            mismatch_points.extend(global_mismatches)

        except Exception as e:
            print(f"Error processing window at index {row['sample_index']}: {e}")
            brs_values.append(0)
    else:
        brs_values.append(0)  # Mark bad windows with 0 BRS

# Create a shared x-axis plot
fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

# ECG
axes[0].plot(filtered_ecg, label="Filtered ECG", alpha=0.7)
axes[0].set_title("Filtered ECG Signal")
axes[0].set_ylabel("Amplitude")
axes[0].grid()

#ABP
axes[1].plot(filtered_abp, label="ABP (mmHg)", alpha=0.7, color="orange")
axes[1].set_title("Arterial Blood Pressure (ABP)")
axes[1].set_ylabel("Pressure (mmHg)")
axes[1].grid()

# Plot BRS and mark mismatched points
axes[2].plot(brs_values, label="BRS (ms/mmHg)", alpha=0.7, color="green")
axes[2].scatter(mismatch_points, [brs_values[i] for i in mismatch_points], color="red", label="Mismatch", alpha=0.7)
axes[2].set_title("Baroreflex Sensitivity (BRS)")
axes[2].set_xlabel("Sample Index")
axes[2].set_ylabel("BRS (ms/mmHg)")
axes[2].grid()

for ax in axes:
    ax.legend()

plt.tight_layout()
plt.show()
