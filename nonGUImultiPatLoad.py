import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from utilities import readPatientRecords2, concatenateECG
from scipy import signal
import pandas as pd
from scipy.stats import pearsonr

patient_id = 'f2o01'
data_directory = r'C:\Users\60427\Desktop\fantasia-database-1.0.0/'
records, annotations = readPatientRecords2(patient_id, data_directory)
start = 0
end = 100000
end = 100000
ecgSignal = concatenateECG(records, start, end)

# Calculate RR Intervals from ECG Signal
def calculate_rr_intervals(ecg_signal, sampling_rate=1000):
    # Dynamically set peak detection threshold and distance for robustness
    peaks, _ = find_peaks(ecg_signal, height=np.mean(ecg_signal), distance=sampling_rate * 0.6)
    rr_intervals = np.diff(peaks) / sampling_rate * 1000  # Convert to milliseconds
    return rr_intervals, peaks

sampling_rate = 250
rr_intervals, peaks = calculate_rr_intervals(ecgSignal, sampling_rate)

# ---------------------------- Quality Check ----------------------------
def calculate_hr_variability(rr_intervals):
    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals)
    hr_variation = (np.max(rr_intervals) - np.min(rr_intervals)) / mean_rr
    return mean_rr, std_rr, hr_variation

def check_hrv_correction_necessity(rr_intervals, sampling_rate, measurement_duration, analysis_purpose='vagal_tone'):
    print("HRV Correction Necessity Report:")
    mean_rr, std_rr, hr_variation = calculate_hr_variability(rr_intervals)
    print(f"Mean RR Interval: {mean_rr:.2f} ms, Std RR Interval: {std_rr:.2f} ms, HR Variation Range: {hr_variation:.2%}")
    correction_needed = (
        hr_variation > 0.2 or 
        measurement_duration > 300 or 
        analysis_purpose == 'vagal_tone' or 
        'RMSSD' in ['RMSSD', 'HF-HRV']
    )
    print("\nOverall HRV Correction Recommendation:", "Yes" if correction_needed else "No")
    return correction_needed

measurement_duration = (end - start) / sampling_rate
correction_needed = check_hrv_correction_necessity(rr_intervals, sampling_rate, measurement_duration)

# ---------------------------- HRV Correction ----------------------------
def interpolate_rr_intervals(rr_intervals, target_rate=8):  # Adjusted target_rate for HRV analysis resolution
    rr_times = np.cumsum(rr_intervals) - rr_intervals[0]
    target_times = np.arange(0, rr_times[-1], 1/target_rate)  # New time points for interpolation
    interpolated_rr = np.interp(target_times, rr_times, rr_intervals)
    return interpolated_rr, target_rate

# Calculate HF and  power using Welch's method in the 0.15-0.4 Hz and 0.03-0.15 Hz ranges
def calculate_hf_power(interpolated_rr, sampling_rate):
    # Dynamically adjust nperseg to avoid exceeding signal length
    nperseg = min(512, len(interpolated_rr))  # Avoid nperseg warning
    f, pxx = signal.welch(interpolated_rr, fs=sampling_rate, nperseg=nperseg)
    
    # Define HF  bands
    hf_band = (f >= 0.15) & (f <= 0.4)

    # Calculate HF power
    hf_power = np.trapz(pxx[hf_band], f[hf_band])  # HF power
    
    return hf_power
def correct_hrv_metrics(rr_intervals):
    """
    Correct HRV measures using cvSDNN, cvRMSSD, cvHF.
    Assumes RR intervals are in milliseconds.
    """
    # Convert RR intervals to seconds for calculations
    rr_intervals_sec = rr_intervals / 1000  # Convert to seconds

    # Calculate mean RR interval in seconds
    mean_rr_sec = np.mean(rr_intervals_sec)
    
    # Calculate SDNN and apply cvSDNN correction
    sdnn = np.std(rr_intervals)  # Standard SDNN in ms
    cv_sdnn = sdnn / mean_rr_sec  

    # Calculate RMSSD and apply cvRMSSD correction
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))  # Standard RMSSD in ms
    cv_rmssd = rmssd / mean_rr_sec  

    # Calculate HF  Power based on interpolated RR intervals
    interpolated_rr, target_rate = interpolate_rr_intervals(rr_intervals_sec)  # Interpolated RR intervals
    hf_power= calculate_hf_power(interpolated_rr, target_rate)  # HF power based on spectral analysis
    cv_hf = hf_power / (mean_rr_sec ** 2)  

    # Print results
    print(f"Original SDNN: {sdnn:.2f} ms, Corrected cvSDNN: {cv_sdnn:.2f}")
    print(f"Original RMSSD: {rmssd:.2f} ms, Corrected cvRMSSD: {cv_rmssd:.2f}")
    print(f"Original HF Power: {hf_power:.2f} ms^2, Corrected cvHF: {cv_hf:.2f}")
    
    # Plot original and corrected HRV metrics for comparison
    metrics_1 = ['SDNN', 'RMSSD']
    original_values_1 = [sdnn, rmssd]
    corrected_values_1 = [cv_sdnn, cv_rmssd]

    plt.figure(figsize=(8, 6))
    x1 = np.arange(len(metrics_1))
    plt.bar(x1 - 0.2, original_values_1, 0.4, label='Original')
    plt.bar(x1 + 0.2, corrected_values_1, 0.4, label='Corrected')
    plt.xticks(x1, metrics_1)
    plt.ylabel('HRV Metrics (ms)')
    plt.title('Comparison of Original and Corrected SDNN & RMSSD')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.bar(['Original', 'Corrected'], [hf_power, cv_hf], width=0.4)
    plt.ylabel('HF Power (ms^2)')
    plt.title('Comparison of Original and Corrected HF Power')
    plt.show()
    
    return cv_sdnn, cv_rmssd, cv_hf

# Calculate HR series based on a sliding window, window size: the number of RR intervals
def calculate_hr_series(rr_intervals, window_size=10):
    hr_series = [60 / (np.mean(rr_intervals[i:i + window_size]) / 1000) for i in range(len(rr_intervals) - window_size)]
    return hr_series

# Calculate HRV metrics 
def calculate_hrv_series(rr_intervals, window_size=10):
    sdnn_series = [np.std(rr_intervals[i:i + window_size]) for i in range(len(rr_intervals) - window_size)]
    rmssd_series = [np.sqrt(np.mean(np.square(np.diff(rr_intervals[i:i + window_size])))) for i in range(len(rr_intervals) - window_size)]
    hf_power_series = [
        calculate_hf_power(interpolate_rr_intervals(rr_intervals[i:i + window_size] / 1000)[0], sampling_rate) 
        for i in range(len(rr_intervals) - window_size)
    ]
    return sdnn_series, rmssd_series, hf_power_series

# Generate a correlation table for HRV and HR before and after correction
def generate_hrv_correlation_table(rr_intervals, cv_sdnn, cv_rmssd, cv_hf, window_size):

    hr_series = calculate_hr_series(rr_intervals, window_size)
    sdnn_series_before, rmssd_series_before, hf_series_before = calculate_hrv_series(rr_intervals, window_size)

    # Calculate corrected HRV series
    corrected_rr_intervals = interpolate_rr_intervals(rr_intervals / 1000)[0]  # RR intervals after correction
    sdnn_series_after, rmssd_series_after, hf_series_after = calculate_hrv_series(corrected_rr_intervals * 1000, window_size)

    # Ensure all series have the same length
    min_length = min(len(hr_series), len(sdnn_series_before), len(sdnn_series_after), 
                     len(rmssd_series_before), len(rmssd_series_after), 
                     len(hf_series_before), len(hf_series_after))
    
    hr_series = hr_series[:min_length]
    sdnn_series_before = sdnn_series_before[:min_length]
    rmssd_series_before = rmssd_series_before[:min_length]
    hf_series_before = hf_series_before[:min_length]
    sdnn_series_after = sdnn_series_after[:min_length]
    rmssd_series_after = rmssd_series_after[:min_length]
    hf_series_after = hf_series_after[:min_length]

    # Calculate correlations between HR and HRV metrics before and after correction
    corr_sdnn_before, _ = pearsonr(hr_series, sdnn_series_before)
    corr_sdnn_after, _ = pearsonr(hr_series, sdnn_series_after)
    corr_rmssd_before, _ = pearsonr(hr_series, rmssd_series_before)
    corr_rmssd_after, _ = pearsonr(hr_series, rmssd_series_after)
    corr_hf_before, _ = pearsonr(hr_series, hf_series_before)
    corr_hf_after, _ = pearsonr(hr_series, hf_series_after)

    correlation_data = {
        'Metric': ['SDNN', 'RMSSD', 'HF Power'],
        'Correlation of HRV to HR Before Correction': [corr_sdnn_before, corr_rmssd_before, corr_hf_before],
        'Correlation of HRV to HR After Correction': [corr_sdnn_after, corr_rmssd_after, corr_hf_after]
    }
    correlation_df = pd.DataFrame(correlation_data)
    
    return correlation_df

window_size=10
cv_sdnn, cv_rmssd, cv_hf = correct_hrv_metrics(rr_intervals)
correlation_table = generate_hrv_correlation_table(rr_intervals, cv_sdnn, cv_rmssd, cv_hf, window_size)
print(correlation_table)
