import numpy as np
import pandas as pd 
import scipy.signal as signal
import wfdb
import os
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import welch
from scipy.integrate import simpson
from statsmodels.tsa.ar_model import AutoReg
import neurokit2 as nk

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
            #here assume all the fs is same in the same signal
            fs = record.fs
            records.append(record)


            
            # annotation_file = os.path.join(data_dir, f"{record_base}.atr")
            # if os.path.exists(annotation_file):
            #     annotation = wfdb.rdann(os.path.join(data_dir, record_base), 'atr')
            #     annotations.append(annotation)
    
    return records, fs, annotations

def concatenateSignals(records, start, end):
    allSignals = []
    for record in records:
        allSignals.append(record.p_signal[:, 0])

    concatenatedSignal = np.concatenate(allSignals)
    return concatenatedSignal[start:end]

def bandpass_filter(signal, low_cutoff, high_cutoff, sampling_rate):
    # Design a bandpass filter with the given cutoffs and sampling rate
    nyquist = 0.5 * sampling_rate
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(4, [low, high], btype='band')
    # Apply the filter to the signal using filtfilt to avoid phase distortion
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def remove_artifacts(signal, threshold=0.5):
    # Simple artifact removal by thresholding: replace values above a threshold with the median of the signal
    median_value = np.median(signal)
    signal[np.abs(signal) > threshold] = median_value
    return signal

def concatenateandProcessSignals(records, start, end, low_cutoff=0.5, high_cutoff=10, sampling_rate=250):
    allSignals = []
    for record in records:
        # Extract the ECG signal from each record (assuming the signal is in the first channel)
        allSignals.append(record.p_signal[:, 0])

    # Concatenate all extracted signals into a single signal
    concatenatedSignal = np.concatenate(allSignals)
    ## Here, add the filter and artifacts processing for concatenatedSignal[start:end]
    filteredSignal = bandpass_filter(concatenatedSignal[start:end], low_cutoff, high_cutoff, sampling_rate)

    # Artifacts processing: Apply an artifact removal technique, such as thresholding or signal interpolation
    processedSignal = remove_artifacts(filteredSignal)

    # Return the filtered and artifact-processed segment of the signal from start to end
    return processedSignal

def visualizeSignal(signal):
    time = range(len(signal))
    
    plt.figure(figsize=(12, 6))
    plt.plot(time, signal)
    plt.title('Concatenated ECG Signal Visualization')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()


def visualizeSignalinGUI(signal):


    # Plot the ECG signal in the GUI
    figure = plt.Figure(figsize=(10, 4), dpi=100)
    ax = figure.add_subplot(111)
    ax.plot(signal)
    ax.set_title('ECG Signal')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Amplitude')
    ax.grid()

    return figure


def calculateSignalProperties(signal, fs):
    #under constraction

    #0. RR interval
    peaks, _ = find_peaks(signal, distance=200)
    rrIntervalSamples = np.diff(peaks)
    samplingRate = fs
    rrIntervalsSecond = rrIntervalSamples / samplingRate

    # 1. Mean Heart Rate (HR)
    hr = 60 / rrIntervalsSecond
    meanHR = np.mean(hr)

    # 2. SDNN (Standard deviation of RR intervals)
    sdnn = np.std(rrIntervalsSecond)

    # 3. RMSSD (Root Mean Square of Successive Differences)
    rrDiff = np.diff(rrIntervalsSecond)  # Differences between successive RR intervals
    rmssd = np.sqrt(np.mean(rrDiff**2))

    properties = {
        # "RR Interval Second": rrIntervalsSecond,
        "Heart Rate":meanHR,
        "Standard deviation of RR intervals(SDNN)":sdnn,
        "Root Mean Square of Successive Differences(RMSSD)":rmssd,
    }
    return properties