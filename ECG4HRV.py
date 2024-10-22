'''
This is the ECG based hrv analysis.
Two main parts now,
    1.RR interval analysis
    2.AR based PSD analysis 
'''
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import welch
from scipy.integrate import simpson

#Data input -- at here, we assume that it is a dat, hea file
##############################################################################
##Load the record and annotation file (this will load the .dat and .hea file)#
##############################################################################
record = wfdb.rdrecord(r'C:\Document\sc2024\mit-bih-arrhythmia-database-1.0.0/100')
annotation = wfdb.rdann(r'C:\Document\sc2024\mit-bih-arrhythmia-database-1.0.0\100', 'atr')
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

##########################################################################
##This is a simply visualization of the ECG signal with detected R-peaks##
##########################################################################
# Visual the ECG in 3000 sample with RR interval
# plt.plot(ecgSignal[:3000])  
# plt.scatter(peaks[:10], ecgSignal[peaks[:10]], color='red', label='R-peaks')
# plt.title('ECG Signal with Detected R-peaks')
# plt.xlabel('Samples')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.show()

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

# 1. Heart Rate (HR)
hr = 60 / rrIntervalsSecond
meanHR = np.mean(hr)

# 2. InterBeat Interval (IBI) = RR intervals
ibi = rrIntervalsSecond  # Interbeat interval is just RR intervals

# 3. SDNN (Standard deviation of RR intervals)
sdnn = np.std(rrIntervalsSecond)

# 4. RMSSD (Root Mean Square of Successive Differences)
rrDiff = np.diff(rrIntervalsSecond)  # Differences between successive RR intervals
rmssd = np.sqrt(np.mean(rrDiff**2))

# 7. Poincare analysis (SD1, SD2)
sd1 = np.sqrt(np.var(rrDiff) / 2)
sd2 = np.sqrt(2 * np.var(rrIntervalsSecond) - np.var(rrDiff) / 2)

# Print the results
print(f"Heart Rate (HR): {meanHR:.2f} bpm")
print(f"InterBeat Interval (IBI): {ibi}")
print(f"SDNN: {sdnn:.4f} seconds")
print(f"RMSSD: {rmssd:.4f} seconds")
print(f"SD1 (Poincare): {sd1:.4f}")
print(f"SD2 (Poincare): {sd2:.4f}")

