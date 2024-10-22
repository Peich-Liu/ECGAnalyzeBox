import wfdb
import matplotlib.pyplot as plt
import numpy as np
import time

# Load the record (example with ECG signal)
record = wfdb.rdrecord('C:/Document/sc2024/data/testData/0284_001_004_ECG')
ecg_signal = record.p_signal[:, 0]  # Assuming this is your ECG signal
fs = record.fs  # Sampling frequency

# Define parameters for real-time simulation
window_size = 2 * fs  # 2-second window of data
step_size = int(0.1 * fs)  # Update every 0.1 second
total_length = len(ecg_signal)

# Initialize figure
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, window_size / fs)  # X-axis is time in seconds
ax.set_ylim(np.min(ecg_signal), np.max(ecg_signal))  # Y-axis limits based on signal
plt.xlabel('Time (seconds)')
plt.ylabel('ECG Signal (mV)')

# Create time axis for window
time_window = np.linspace(0, window_size / fs, window_size)

# Start the real-time plotting
for i in range(0, total_length - window_size, step_size):
    # Extract the next segment of the signal
    ecg_segment = ecg_signal[i:i + window_size]
    
    # Update the plot data
    line.set_xdata(time_window)
    line.set_ydata(ecg_segment)
    
    # Redraw the plot
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    # Pause to simulate real-time (adjust the speed as needed)
    time.sleep(0.1)  # Adjust sleep time to match the real-time speed
