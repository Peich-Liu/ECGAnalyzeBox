import wfdb
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import find_peaks
from scipy.signal import welch
from scipy.integrate import simpson
from statsmodels.tsa.ar_model import AutoReg
import neurokit2 as nk
#Data input -- at here, we assume that it is a dat, hea file
##############################################################################
##Load the record and annotation file (this will load the .dat and .hea file)#
##############################################################################
record = wfdb.rdrecord(r'C:\Document\sc2024\mit-bih-arrhythmia-database-1.0.0/100')
annotation = wfdb.rdann(r'C:\Document\sc2024\mit-bih-arrhythmia-database-1.0.0\100', 'atr')
# wfdb.plot_wfdb(record=record, title='MIT-BIH Record 100')
# Access the ECG signal
ecgSignal = record.p_signal[:, 0]  

# Clean signal and Find peaks
ecg_cleaned = nk.ecg_clean(ecgSignal, sampling_rate=record.fs)

peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=record.fs, correct_artifacts=True)

# Get the indices of the detected R-peaks
r_peaks_indices = info['ECG_R_Peaks']

# # Plot the cleaned ECG signal
# plt.figure(figsize=(12, 6))
# plt.plot(ecg_cleaned, label='Cleaned ECG Signal', color='blue')

# # Mark the R-peaks
# plt.scatter(r_peaks_indices, ecg_cleaned[r_peaks_indices], color='red', label='R Peaks', marker='o')

# # Add labels and legend
# plt.title('Cleaned ECG Signal with R Peaks')
# plt.xlabel('Samples')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.show()

hrv_welch = nk.hrv_frequency(peaks, sampling_rate=100, show=True, psd_method="welch")
plt.show()