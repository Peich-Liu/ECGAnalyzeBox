import numpy as np
import pandas as pd 
import scipy.signal as signal
from scipy.signal import butter, filtfilt

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


# # Example usage:
# # ECG typically in range 0.5-50 Hz
# lowcut_ecg = 0.5
# highcut_ecg = 50.0

# # EEG typically in range 0.5-40 Hz
# lowcut_eeg = 0.5
# highcut_eeg = 40.0

# fs = 500  # Sampling frequency, for example

# # Assuming you have ecg_signal and eeg_signal
# filtered_ecg = bandpass_filter(ecg_signal, lowcut_ecg, highcut_ecg, fs)
# filtered_eeg = bandpass_filter(eeg_signal, lowcut_eeg, highcut_eeg, fs)
