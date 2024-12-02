import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy import stats
import matplotlib.pyplot as plt
from utilities import bandpass_filter, detect_r_peaks, compute_rr_intervals, detect_sbp, synchronize_rr_sbp_cycle_based, compute_brs_sequence
def brs_seqence_analyze(ecg_signal, ap_signal, fs=250):
    
    ecg_signal = bandpass_filter(ecg_signal, 0.5, 45, fs)
    ap_signal = bandpass_filter(ap_signal, 0.5, 10, fs)
    
    t = np.arange(len(ecg_signal)) / fs

    # 从ECG信号中检测R峰
    r_peaks_indices, r_peaks_times = detect_r_peaks(ecg_signal, fs)

    # 计算RR间期
    rr_intervals, rr_times = compute_rr_intervals(r_peaks_times)

    # 从AP信号中检测SBP
    sbp_values, sbp_times = detect_sbp(ap_signal, fs)

    rr_intervals_sync, sbp_values_sync, rr_times_sync = synchronize_rr_sbp_cycle_based(r_peaks_times, rr_intervals, sbp_times, sbp_values)
    print("rr_intervals_sync",len(rr_intervals_sync),"ecg:",len(ecg_signal))

    brs_sequences, mean_brs = compute_brs_sequence(rr_intervals_sync, sbp_values_sync)

    results_dict = {
    "r_peaks_indices": r_peaks_indices,
    "r_peaks_times": r_peaks_times,
    "rr_intervals": rr_intervals,
    "rr_times": rr_times,
    "sbp_values": sbp_values,
    "sbp_times": sbp_times,
    "rr_intervals_sync": rr_intervals_sync,
    "sbp_values_sync": sbp_values_sync,
    "rr_times_sync": rr_times_sync,
    "brs_sequences": brs_sequences,
    "mean_brs": mean_brs
    }

    return results_dict

def calculate_brs():
    # # 读取数据
    fs = 250
    quality_file = r"c:\Document\sc2024\filtered_ecg_with_snr.csv"
    quality_data = pd.read_csv(quality_file)

    ecg_signal = quality_data['ecg'].values
    ap_signal = quality_data['ap'].values

    
    t = np.arange(len(ecg_signal)) / fs

    brs_result = brs_seqence_analyze(ecg_signal, ap_signal)

    brs_sequences = brs_result['brs_sequences']
    mean_brs = brs_result['mean_brs']
    r_peaks_times = brs_result['r_peaks_times']
    r_peaks_indices = brs_result['r_peaks_indices']
    sbp_times = brs_result['sbp_times']
    rr_times = brs_result['rr_times']
    rr_intervals = brs_result['rr_intervals']
    sbp_values = brs_result['sbp_values']
    rr_times_sync = brs_result['rr_times_sync']
    rr_intervals_sync = brs_result['rr_intervals_sync']
    sbp_values_sync = brs_result['sbp_values_sync']

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
            # 添加斜率注释
            # plt.text(rr_times_sync[start_idx], rr_intervals_sync[start_idx], f"Slope: {seq['slope']:.2f}", color='black')


    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    calculate_brs()