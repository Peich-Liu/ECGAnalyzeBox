import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy import stats
import matplotlib.pyplot as plt
from utilities import bandpass_filter, detect_r_peaks, compute_rr_intervals, detect_sbp, synchronize_rr_sbp_cycle_based, compute_brs_sequence, getFigure
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

def calculate_brs(ecg_signal, ap_signal, brs_canvas_frame, fs=250):
    t = np.arange(len(ecg_signal)) / fs

    brs_result = brs_seqence_analyze(ecg_signal, ap_signal, fs=250)

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

    figure = plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    plt.plot(t, ecg_signal, label='ECG signal')
    plt.plot(r_peaks_times, ecg_signal[r_peaks_indices], 'ro', label='R peak')
    plt.title('ECG signal and RR interval')
    plt.xlabel('Time(sample)')
    plt.ylabel('Ampulitude')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t, ap_signal, label='AP signal')
    sbp_indices = (sbp_times * fs).astype(int)
    sbp_indices = np.clip(sbp_indices, 0, len(ap_signal) - 1)  # 防止索引越界
    plt.plot(sbp_times, ap_signal[sbp_indices], 'go', label='SBP peak')
    plt.title('AP signal and SBP')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (mmHg)')
    plt.legend()
    
    # 绘制同步后的RR间期和SBP值，并在图中标记BRS序列
    plt.subplot(3, 1, 3)
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
    # figure.show()
    getFigure(figure, brs_canvas_frame)
    # figure, axs = plt.subplots(4, 1, figsize=(12, 10))

    # # 第一个子图
    # ax = axs[0]
    # ax.plot(t, ecg_signal, label='ECG signal')
    # ax.plot(r_peaks_times, ecg_signal[r_peaks_indices], 'ro', label='R peak')
    # ax.set_title('ECG signal and RR interval')
    # ax.set_xlabel('Time(sample)')
    # ax.set_ylabel('Amplitude')
    # ax.legend()

    # # 第二个子图
    # ax = axs[1]
    # ax.plot(t, ap_signal, label='AP signal')
    # sbp_indices = (sbp_times * fs).astype(int)
    # sbp_indices = np.clip(sbp_indices, 0, len(ap_signal) - 1)  # 防止索引越界
    # ax.plot(sbp_times, ap_signal[sbp_indices], 'go', label='SBP peak')
    # ax.set_title('AP signal and SBP')
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Pressure (mmHg)')
    # ax.legend()

    # # 第三个子图
    # ax = axs[2]
    # ax.plot(rr_times, rr_intervals, label='RR interval')
    # ax.plot(sbp_times, sbp_values, label='SBP')
    # ax.legend()
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Value')
    # ax.set_title('RR and SBP before sync')

    # # 第四个子图
    # ax = axs[3]
    # ax.plot(rr_times_sync, rr_intervals_sync, label='RR interval (after sync)')
    # ax.plot(rr_times_sync, sbp_values_sync, label='SBP (after sync)')
    # ax.set_title('RR interval and SBP after sync')
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Value')

    # # 标记BRS序列
    # for seq in brs_sequences:
    #     if abs(seq['r_value']) >= 0.85:
    #         start_idx = seq['start_idx']
    #         end_idx = seq['end_idx']
    #         ax.plot(rr_times_sync[start_idx:end_idx+1], rr_intervals_sync[start_idx:end_idx+1], 'r', linewidth=2)
    #         ax.plot(rr_times_sync[start_idx:end_idx+1], sbp_values_sync[start_idx:end_idx+1], 'g', linewidth=2)
    #         # 添加斜率注释
    #         # ax.text(rr_times_sync[start_idx], rr_intervals_sync[start_idx], f"Slope: {seq['slope']:.2f}", color='black')

    # ax.legend()
    # figure.tight_layout()
    # getFigure(figure, brs_canvas_frame)


    return brs_result