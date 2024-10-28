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
from scipy.signal import butter, filtfilt
import numpy as np
import os
import wfdb
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

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
    print("low, high, len", low, high, len(signal))

    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def remove_artifacts(signal, threshold=0.5):
    # Simple artifact removal by thresholding: replace values above a threshold with the median of the signal
    median_value = np.median(signal)
    signal[np.abs(signal) > threshold] = median_value
    return signal

def concatenateSignals(records, start, end, low_cutoff=0.5, high_cutoff=10, sampling_rate=250):
    allSignals = []
    for record in records:
        # Extract the ECG signal from each record (assuming the signal is in the first channel)
        allSignals.append(record.p_signal)
    concatenatedSignal = np.concatenate(allSignals, axis=0)
    # Return the filtered and artifact-processed segment of the signal from start to end
    return concatenatedSignal[start:end, :]

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


def visualizeSignalinGUIMultiChannel(signals):
    # Plot the ECG signal in the GUI
    num_channels = signals.shape[1]
    figure, axes = plt.subplots(num_channels, 1, figsize=(10, 6), sharex=True)

    if num_channels == 1:
        axes = [axes]

    for i in range(num_channels):
        axes[i].plot(signals[:, i])
        axes[i].set_title(f'Channel {i+1}')
        axes[i].set_ylabel('Amplitude')
    
    axes[-1].set_xlabel('Samples')

    figure.tight_layout()
    return figure

def visualizeSignalinGUISelectChannel(signals, selected_channel):
    figure = plt.Figure(figsize=(10, 6))
    ax = figure.add_subplot(111)

    ax.plot(signals[:, selected_channel], label=f'Channel {selected_channel + 1}')
    ax.set_title(f'ECG Signal Visualization - Channel {selected_channel + 1}')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Amplitude')
    ax.legend()

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


def PSDAnalyze(ecgSignal, fs):
    peaks, info = nk.ecg_peaks(ecgSignal, sampling_rate=fs, correct_artifacts=True)
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

    return freq, psd

def visualPSD(freq, psd):
    ulf_range = (0.0, 0.0033)
    vlf_range = (0.0033, 0.04)
    lf_range = (0.04, 0.15)
    hf_range = (0.15, 0.4)
    vhf_range = (0.4, 0.5)

    figure = plt.figure(figsize=(4, 2))
    ax = figure.add_subplot(111)  # 添加子图

    ax.plot(freq, psd, color='black', linewidth=1, label='PSD')

    ax.fill_between(freq, psd, where=((freq >= ulf_range[0]) & (freq <= ulf_range[1])),
                    color='purple', alpha=0.5, label='ULF')
    ax.fill_between(freq, psd, where=((freq >= vlf_range[0]) & (freq <= vlf_range[1])),
                    color='blue', alpha=0.5, label='VLF')
    ax.fill_between(freq, psd, where=((freq >= lf_range[0]) & (freq <= lf_range[1])),
                    color='green', alpha=0.5, label='LF')
    ax.fill_between(freq, psd, where=((freq >= hf_range[0]) & (freq <= hf_range[1])),
                    color='orange', alpha=0.5, label='HF')
    ax.fill_between(freq, psd, where=((freq >= vhf_range[0]) & (freq <= vhf_range[1])),
                    color='red', alpha=0.5, label='VHF')

    ax.set_title('PSD for Frequency Domains')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Spectrum (ms^2/Hz)')

    # 将图例放在图表的外部右侧
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))  # 图例在图外右侧
    figure.tight_layout()  # 自动调整子图以适应图例
    # figure = plt.figure(figsize=(5, 2))
    # plt.plot(freq, psd, color='black', linewidth=1, label='Power Spectral Density')

    # plt.fill_between(freq, psd, where=((freq >= ulf_range[0]) & (freq <= ulf_range[1])),
    #                 color='purple', alpha=0.5, label='ULF')
    # plt.fill_between(freq, psd, where=((freq >= vlf_range[0]) & (freq <= vlf_range[1])),
    #                 color='blue', alpha=0.5, label='VLF')
    # plt.fill_between(freq, psd, where=((freq >= lf_range[0]) & (freq <= lf_range[1])),
    #                 color='green', alpha=0.5, label='LF')
    # plt.fill_between(freq, psd, where=((freq >= hf_range[0]) & (freq <= hf_range[1])),
    #                 color='orange', alpha=0.5, label='HF')
    # plt.fill_between(freq, psd, where=((freq >= vhf_range[0]) & (freq <= vhf_range[1])),
    #                 color='red', alpha=0.5, label='VHF')

    # plt.title('Power Spectral Density (PSD) for Frequency Domains')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Spectrum (ms^2/Hz)')
    # plt.legend()
    # plt.show()
    return figure

def getFigure(figure, canvas_frame):
    # Clear previous plot if it exists
    for widget in canvas_frame.winfo_children():
        widget.destroy()
    # Embed the plot in the Tkinter canvas
    canvas = FigureCanvasTkAgg(figure, master=canvas_frame)
    canvas.draw()
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)

    # Add navigation toolbar for better interactivity
    toolbar = NavigationToolbar2Tk(canvas, canvas_frame)
    toolbar.update()
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)