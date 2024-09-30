import wfdb
import matplotlib.pyplot as plt
import numpy as np
import time

ecgFilePath = r'C:\\Document\\sc2024\\data\\testData\\0284_001_004_ECG'
eegFilePath = r'C:\\Document\\sc2024\\data\\testData\\0284_001_004_EEG'
otherFilePath = r'C:\\Document\\sc2024\\data\\testData\\0284_001_004_OTHER'

ecgRecord = wfdb.rdrecord(ecgFilePath)
eegRecord = wfdb.rdrecord(eegFilePath)
otherRecord = wfdb.rdrecord(otherFilePath)
# Load the record (example with both ECG and EEG signals)
record = wfdb.rdrecord('C:/Document/sc2024/data/testData/0284_001_004_ECG')

# Assuming EEG is stored in the second channel and ECG in the first
ecg_signal = ecgRecord.p_signal[:, 0]  # ECG signal
eeg_signal = eegRecord.p_signal[:, 1]  # EEG signal
fs = record.fs  # Sampling frequency

# Define parameters for real-time simulation
window_size = 2 * fs  # 2-second window of data
step_size = int(0.1 * fs)  # Update every 0.1 second
total_length = len(ecg_signal)

# Create time axis for window
time_window = np.linspace(0, window_size / fs, window_size)
# Initialize figure with two subplots
plt.ion()  # Turn on interactive mode
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Set up the first plot for EEG
line1, = ax1.plot([], [], lw=2)
ax1.set_xlim(0, window_size / fs)  # X-axis is time in seconds
ax1.set_ylim(np.min(eeg_signal), np.max(eeg_signal))  # Y-axis limits based on EEG signal
ax1.set_title('EEG Signal')
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Amplitude (µV)')

# Set up the second plot for ECG
line2, = ax2.plot([], [], lw=2, color='red')
ax2.set_xlim(0, window_size / fs)  # X-axis is time in seconds
ax2.set_ylim(np.min(ecg_signal), np.max(ecg_signal))  # Y-axis limits based on ECG signal
ax2.set_title('ECG Signal')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Amplitude (mV)')

plt.tight_layout()
timeStop = window_size / fs
# Start the real-time plotting
for i in range(0, total_length - window_size, step_size):
    # Extract the next segment of the signal
    eeg_segment = eeg_signal[i:i + window_size]
    ecg_segment = ecg_signal[i:i + window_size]
    
    # Update the EEG plot
    line1.set_xdata(time_window)
    line1.set_ydata(eeg_segment)
    
    # Update the ECG plot
    line2.set_xdata(time_window)
    line2.set_ydata(ecg_segment)
    
    # Redraw the plots
    fig.canvas.draw()
    fig.canvas.flush_events()

    time.sleep(0.1) 
    
    # Pause to simulate real-time (adjust the speed as needed)
    # time.sleep(0.1)  # Adjust sleep time to match real-time speed
