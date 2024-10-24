import numpy as np
import pandas as pd 
import scipy.signal as signal
import wfdb
import os
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


def bandPass(data, zi, lowCut, highCut, fs, order=4):
    nyq = fs
    low = lowCut / nyq
    high = highCut / nyq
    sos = signal.butter(order, [lowCut, highCut], btype='band',output='sos',fs=fs)
    sig, zi = signal.sosfilt(sos, data, zi=zi)
    return sig, zi

def lowPass(data, zi, lowCut, fs, order=4):
    nyq = fs
    low = lowCut / nyq
    sos = signal.butter(order, lowCut, btype='low',output='sos',fs=fs)
    sig, zi = signal.sosfilt(sos, data, zi=zi)
    return sig, zi

def highPass(data, zi, highCut, fs, order=4):
    nyq = fs
    high = highCut / nyq
    sos = signal.butter(order, highCut, btype='high',output='sos',fs=fs)
    sig, zi = signal.sosfilt(sos, data, zi=zi)
    return sig, zi

def filterSignalwithoutMean(data, zi=None, low=None, high=None, fs=250):
    data = np.array(data)
    if low != 0 and high != 0:
        return bandPass(data, zi,low, high, fs, order=4)
    elif low == 0 and high != 0:
        return lowPass(data, zi,high, fs, order=4)
    elif high == 0 and low != 0:
        return highPass(data, zi,low, fs, order=4)
    elif low == 0 and high == 0:
        return data

def filterSignal(data, zi=None, low=None, high=None, fs=250, order=4, diff=False):
    data = np.array(data)
    data -= np.mean(data)
    if low != 0 and high != 0:
        return bandPass(data, zi,low, high, fs, order=4)
    elif low == 0 and high != 0:
        return lowPass(data, zi,high, fs, order=4)
    elif high == 0 and low != 0:
        return highPass(data, zi,low, fs, order=4)
    elif low == 0 and high == 0:
        return data

#Multi-File loading -- it is a test based on mit dataset, so, the data is not right
def readPatientRecords(patient_id, data_dir):
    patient_prefix = patient_id[:2]
    records = []
    annotations = []

    for file in os.listdir(data_dir):
        if file.startswith(patient_prefix) and file.endswith('.dat'):
            record_base = os.path.splitext(file)[0]
            
            record = wfdb.rdrecord(os.path.join(data_dir, record_base))
            records.append(record)
            
            # annotation_file = os.path.join(data_dir, f"{record_base}.atr")
            # if os.path.exists(annotation_file):
            #     annotation = wfdb.rdann(os.path.join(data_dir, record_base), 'atr')
            #     annotations.append(annotation)
    
    return records, annotations

def concatenateSignals(records, start, end):
    allSignals = []
    for record in records:
        allSignals.append(record.p_signal[:, 0])

    concatenatedSignal = np.concatenate(allSignals)
    return concatenatedSignal[start:end]

def visualizeSignal(signal):
    time = range(len(signal))
    
    plt.figure(figsize=(12, 6))
    plt.plot(time, signal)
    plt.title('Concatenated ECG Signal Visualization')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()