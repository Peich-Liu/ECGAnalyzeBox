import numpy as np
import matplotlib.pyplot as plt
import wfdb


from methodLibrary import *
from scipy.signal import correlate

# Example signals (replace with your real signals)
# For example, load ECG and EEG signals here
# ecg_signal = <Your ECG Signal>
# eeg_signal = <Your EEG Signal>
'''
['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
Channel in this example EEG signal
'''
# Signal Loading
np.random.seed(0)
ecgFilePath = r'C:\\Document\\sc2024\\data\\testData\\0284_001_004_ECG'
eegFilePath = r'C:\\Document\\sc2024\\data\\testData\\0284_001_004_EEG'
otherFilePath = r'C:\\Document\\sc2024\\data\\testData\\0284_001_004_OTHER'
ecgRecord = wfdb.rdrecord(ecgFilePath)
eegRecord = wfdb.rdrecord(eegFilePath)
otherRecord = wfdb.rdrecord(otherFilePath)


fs = eegRecord.fs #This is based on the EEG and ECG are in the same sample rate
#signal recording
eegIndex = eegRecord.sig_name.index('Fp1')
eegSignal = eegRecord.p_signal[:, eegIndex]
ecgIndex = ecgRecord.sig_name.index('ECG')
ecgSignal = ecgRecord.p_signal[:, ecgIndex]

#Filter
sosEcg= signal.butter(4, [0.5,5], btype='bandpass', output='sos', fs=500)

sosEegAlpha = signal.butter(4, [8,13], btype='bandpass', output='sos', fs=500)
sosEegBeta  = signal.butter(4, [14,30], btype='bandpass', output='sos', fs=500)
sosEegGamma = signal.butter(4, [31,50], btype='bandpass', output='sos', fs=500)
sosEegDelta = signal.butter(4, [1,3], btype='bandpass', output='sos', fs=500)
sosEegTheta = signal.butter(4, [4,7], btype='bandpass', output='sos', fs=500)




filteredEcg = signal.sosfiltfilt(sosEcg, ecgSignal)

eegAlpha = signal.sosfiltfilt(sosEegAlpha, eegSignal)
eegBeta  = signal.sosfiltfilt(sosEegBeta, eegSignal)
eegGamma = signal.sosfiltfilt(sosEegGamma, eegSignal)
eegDelta = signal.sosfiltfilt(sosEegDelta, eegSignal)
eegTheta = signal.sosfiltfilt(sosEegTheta, eegSignal)

# Cross-correlation
correlation = correlate(filteredEcg, eegDelta, mode='full')

# Calculate time delays for the cross-correlation
lags = np.arange(-len(filteredEcg) + 1, len(eegDelta))





plt.figure(figsize=(12, 6))
# Plot ECG signal
plt.subplot(3, 1, 1)
plt.plot( filteredEcg, label='ECG Signal')
plt.title('ECG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot EEG signal
plt.subplot(3, 1, 2)
plt.plot( eegDelta, label='EEG Signal', color='orange')
plt.title('EEG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot cross-correlation
plt.subplot(3, 1, 3)
plt.plot(lags, correlation)
plt.title('Cross-Correlation Between ECG and EEG')
plt.xlabel('Lag')
plt.ylabel('Correlation')

plt.tight_layout()
plt.show()
