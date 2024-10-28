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
record = wfdb.rdrecord(r'C:\Document\sc2024\mit-bih-arrhythmia-database-1.0.0/101')
annotation = wfdb.rdann(r'C:\Document\sc2024\mit-bih-arrhythmia-database-1.0.0\101', 'atr')
# wfdb.plot_wfdb(record=record, title='MIT-BIH Record 100')
# Access the ECG signal
ecgSignal = record.p_signal[:, 0]  

# Clean signal and Find peaks
ecg_cleaned = nk.ecg_clean(ecgSignal, sampling_rate=record.fs)

peaks, info = nk.ecg_peaks(ecgSignal, sampling_rate=record.fs, correct_artifacts=True)


# Get the indices of the detected R-peaks
r_peaks_indices = info['ECG_R_Peaks']
rr_intervals = np.diff(r_peaks_indices) / 100
rr_intervals_mean = np.mean(rr_intervals)
rr_intervals_centered = rr_intervals - rr_intervals_mean

#Welch function
freq, psd = welch(rr_intervals_centered, fs=4, nperseg=256)

#AR function
# order = 16 
# ar_model = AutoReg(rr_intervals_centered, lags=order).fit()
# ar_coefficients = ar_model.params
# sampling_rate = 4 
# frequencies, response = signal.freqz([1], np.concatenate(([1], -ar_coefficients[1:])), worN=512, fs=sampling_rate)
# power_spectrum = np.abs(response) ** 2

mask = (freq >= 0) & (freq <= 0.5)
freq = freq[mask]
psd = psd[mask]

ulf_range = (0.0, 0.0033)
vlf_range = (0.0033, 0.04)
lf_range = (0.04, 0.15)
hf_range = (0.15, 0.4)
vhf_range = (0.4, 0.5)

plt.figure()
plt.plot(freq, psd, color='black', linewidth=1, label='PSD')

plt.fill_between(freq, psd, where=((freq >= ulf_range[0]) & (freq <= ulf_range[1])),
                color='purple', alpha=0.5, label='ULF')
plt.fill_between(freq, psd, where=((freq >= vlf_range[0]) & (freq <= vlf_range[1])),
                color='blue', alpha=0.5, label='VLF')
plt.fill_between(freq, psd, where=((freq >= lf_range[0]) & (freq <= lf_range[1])),
                color='green', alpha=0.5, label='LF')
plt.fill_between(freq, psd, where=((freq >= hf_range[0]) & (freq <= hf_range[1])),
                color='orange', alpha=0.5, label='HF')
plt.fill_between(freq, psd, where=((freq >= vhf_range[0]) & (freq <= vhf_range[1])),
                color='red', alpha=0.5, label='VHF')

plt.title('Power Spectral Density (PSD) for Frequency Domains')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Spectrum (ms^2/Hz)')
plt.legend()
plt.show()
# hrv_welch = nk.hrv_frequency(peaks, sampling_rate=record.fs, show=True, psd_method="welch")
# plt.show()
