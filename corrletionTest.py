import numpy as np
import matplotlib.pyplot as plt
import wfdb

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
#EEG signal recording
eegIndex = eegRecord.sig_name.index('Fp1')
eegSignal = eegRecord.p_signal[:, eegIndex]
#ECG signal recording
ecgIndex = ecgRecord.sig_name.index('ECG')
ecgSignal = ecgRecord.p_signal[:, ecgIndex]


#Filter


#artifacts 



# Cross-correlation
correlation = correlate(ecgSignal, eegSignal, mode='full')

# Calculate time delays for the cross-correlation
lags = np.arange(-len(ecgSignal) + 1, len(ecgSignal))





plt.figure(figsize=(12, 6))
# Plot ECG signal
plt.subplot(3, 1, 1)
plt.plot( ecgSignal, label='ECG Signal')
plt.title('ECG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot EEG signal
plt.subplot(3, 1, 2)
plt.plot( eegSignal, label='EEG Signal', color='orange')
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
