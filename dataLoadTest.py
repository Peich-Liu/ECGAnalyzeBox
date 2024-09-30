import wfdb
import wfdb
import matplotlib.pyplot as plt
import numpy as np

"""
It's a note for the record parameter
====================================
n_sig : int, optional
    Total number of signals.
fs : int, float, optional
    The sampling frequency of the record.
counter_freq : float, optional
    The frequency used to start counting.
base_counter : float, optional
    The counter used at the start of the file.
sig_len : int, optional
    The total length of the signal.
base_time : datetime.time, optional
    The time of day at the beginning of the record.
base_date : datetime.date, optional
    The date at the beginning of the record.
base_datetime : datetime.datetime, optional
    The date and time at the beginning of the record, equivalent to
    `datetime.combine(base_date, base_time)`.
comments : list, optional
    A list of string comments to be written to the header file.
sig_name : str, optional
    A list of strings giving the signal name of each signal channel.
"""

# The file is not store in the Git repo, and dont push the data into the git repo
# Dont write the ".hea" in the file path loading
ecgFilePath = r'C:\\Document\\sc2024\\data\\testData\\0284_001_004_ECG'
eegFilePath = r'C:\\Document\\sc2024\\data\\testData\\0284_001_004_EEG'
otherFilePath = r'C:\\Document\\sc2024\\data\\testData\\0284_001_004_OTHER'

ecgRecord = wfdb.rdrecord(ecgFilePath)
eegRecord = wfdb.rdrecord(eegFilePath)
otherRecord = wfdb.rdrecord(otherFilePath)
print(eegRecord.sig_name)
#EEG signal recording
eegIndex = eegRecord.sig_name.index('FP1')
eegSignal = eegRecord.p_signal[:, eegIndex]
fs = eegRecord.fs
eegSamples = eegRecord.sig_len
time = np.linspace(0, eegSamples/fs, eegSamples)

#ECG signal recording
ecgIndex = ecgRecord.sig_name.index('ECG')
ecgSignal = ecgRecord.p_signal[:, ecgIndex]
fs = ecgRecord.fs
ecgSamples = ecgRecord.sig_len
time = np.linspace(0, ecgSamples/fs, ecgSamples)

plt.figure(figsize=(15, 10))

# ECG figure
plt.subplot(2, 1, 1)
plt.plot(time, ecgSignal, color='blue')
plt.title('ECG')
plt.xlabel('s')
plt.ylabel('mV')

# EEG figure
plt.subplot(2, 1, 2)
plt.plot(time, eegSignal, color='green')
plt.title('EEG-FP1') #"FP1 should be change in the other channel"
plt.xlabel('s')
plt.ylabel('mV')

plt.tight_layout()
plt.show()