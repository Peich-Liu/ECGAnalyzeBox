import scipy.signal
from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

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
