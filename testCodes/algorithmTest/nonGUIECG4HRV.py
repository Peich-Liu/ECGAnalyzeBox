'''
This is the non GUI ECG based hrv analysis.
Two main parts now,
    1.RR interval analysis
    2.AR based PSD analysis 
'''
import wfdb
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import welch
from scipy.integrate import simpson
from statsmodels.tsa.ar_model import AutoReg
import neurokit2 as nk
#Data input -- at here, we assume that it is a dat, hea file
##############################################################################
##Load the record and annotation file (this will load the .dat and .hea file)#
##############################################################################
record = wfdb.rdrecord(r'C:\Document\sc2024\mit-bih-arrhythmia-database-1.0.0/101')
annotation = wfdb.rdann(r'C:\Document\sc2024\mit-bih-arrhythmia-database-1.0.0\101', 'atr')
# wfdb.plot_wfdb(record=record, title='MIT-BIH Record 100')
# Access the ECG signal
ecgSignal = record.p_signal[:, 0]  
#####################################################################################################
##After getting the new file, change the previous code to the right version and delete this comment##
#####################################################################################################
#########################################################
##Detect R-peaks using a simple peak detection algorithm#
#########################################################

peaks, _ = find_peaks(ecgSignal, distance=200)
rrIntervalSamples = np.diff(peaks)
# Convert RR intervals to time
samplingRate = record.fs
rrIntervalsSecond = rrIntervalSamples / samplingRate
# rrIntervalsSecond = rrIntervalSamples
# print(rrIntervalSamples)
##########################################################################
##This is a simply visualization of the ECG signal with detected R-peaks##
##########################################################################
# Visual the ECG in 3000 sample with RR interval
plt.plot(ecgSignal)  
plt.scatter(peaks, ecgSignal[peaks], color='red', label='R-peaks')
plt.title('ECG Signal with Detected R-peaks')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# Visual the whole ECG with RR interval
# plt.plot(ecgSignal)  # Plot first 3000 samples
# plt.scatter(peaks, ecgSignal[peaks], color='red', label='R-peaks')
# plt.title('ECG Signal with Detected R-peaks')
# plt.xlabel('Samples')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.show()

###########################
##Time Domain measurments##
###########################

# # 1. Heart Rate (HR)
# hr = 60 / rrIntervalsSecond
# meanHR = np.mean(hr)

# # 2. InterBeat Interval (IBI) = RR intervals
# ibi = rrIntervalsSecond  # Interbeat interval is just RR intervals

# # 3. SDNN (Standard deviation of RR intervals)
# sdnn = np.std(rrIntervalsSecond)

# # 4. RMSSD (Root Mean Square of Successive Differences)
# rrDiff = np.diff(rrIntervalsSecond)  # Differences between successive RR intervals
# rmssd = np.sqrt(np.mean(rrDiff**2))

# # 7. Poincare analysis (SD1, SD2)
# sd1 = np.sqrt(np.var(rrDiff) / 2)
# sd2 = np.sqrt(2 * np.var(rrIntervalsSecond) - np.var(rrDiff) / 2)

# # Print the results
# print(f"Heart Rate (HR): {meanHR:.2f} bpm")
# print(f"InterBeat Interval (IBI): {ibi}")
# print(f"SDNN: {sdnn:.4f} seconds")
# print(f"RMSSD: {rmssd:.4f} seconds")
# print(f"SD1 (Poincare): {sd1:.4f}")
# print(f"SD2 (Poincare): {sd2:.4f}")

###########################
##Freq Domain measurments##
###########################

arOrder = 10
arModel = AutoReg(rrIntervalsSecond, lags=arOrder, old_names=False).fit()
arParams = arModel.params
f, psd = welch(rrIntervalsSecond, fs=samplingRate, nperseg=len(rrIntervalsSecond))

mask = (f >= 0) & (f <= 0.5)
f = f[mask]
psd = psd[mask]

plt.plot(f, psd)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('Power Spectral Density (PSD) of RR Intervals')
plt.show()

lf_band = (f >= 0.04) & (f < 0.15)
hf_band = (f >= 0.15) & (f < 0.4)

lf_power = np.trapz(psd[lf_band], f[lf_band])
hf_power = np.trapz(psd[hf_band], f[hf_band])

lf_hf_ratio = lf_power / hf_power

print("LF Power:", lf_power)
print("HF Power:", hf_power)
print("LF/HF Ratio:", lf_hf_ratio)
print("fs",record.fs)

# #Use AR calculate the psd
###懒得写ar了 过两天再写
# rpeaks_converted = nk.intervals_to_peaks(rrIntervalsSecond, sampling_rate=record.fs)

# hrv_results = nk.hrv_frequency(rpeaks_converted, sampling_rate=record.fs, method='ar')
# print(hrv_results)

#2

# ecg_processed, rpeaks = nk.ecg_process(ecgSignal, sampling_rate=record.fs)
# rpeaks_indices = rpeaks['ECG_R_Peaks']

# hrv_results = nk.hrv_frequency(rpeaks_indices, sampling_rate=record.fs, method='ar')
# print(hrv_results)



# # LF and HF
# lf_band = (0.04, 0.15)
# hf_band = (0.15, 0.4)
# lf = np.trapz(psd[(freqs >= lf_band[0]) & (freqs <= lf_band[1])])
# hf = np.trapz(psd[(freqs >= hf_band[0]) & (freqs <= hf_band[1])])

# print(f"LF: {lf}, HF: {hf}, LF/HF ratio: {lf/hf}")

# # 计算SD1和SD2
# hrv_nonlinear = nk.hrv_nonlinear(rrIntervalsSecond, show=True)

# # 输出SD1和SD2
# sd1 = hrv_nonlinear['HRV_SD1'][0]
# sd2 = hrv_nonlinear['HRV_SD2'][0]

# print(f"SD1: {sd1}, SD2: {sd2}")