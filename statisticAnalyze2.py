# import wfdb
# import pandas as pd
# import matplotlib.pyplot as plt

# # Function to load ECG data
# def load_ecg(file_path):
#     record = wfdb.rdrecord(file_path)
#     return record.p_signal[:,0], record.fs  # Assuming ECG is the first channel

# # Function to read quality annotations
# def read_annotations(file_path):
#     return pd.read_csv(file_path, header=None, names=['start', 'end', 'quality'])

# # Plot ECG with quality annotations
# def plot_ecg(signal, fs, annotations):
#     time_axis = [i / fs for i in range(len(signal))]
#     plt.figure(figsize=(14, 4))
#     plt.plot(time_axis, signal, label='ECG Signal')
    
#     # Highlight different quality segments
#     for _, row in annotations.iterrows():
#         start_time = row['start'] / fs
#         end_time = row['end'] / fs
#         plt.axvspan(start_time, end_time, color='red' if row['quality'] == 3 else 'yellow', alpha=0.3)

#     plt.xlabel('Time (seconds)')
#     plt.ylabel('Amplitude')
#     plt.title('ECG Signal with Quality Annotations')
#     plt.legend()
#     plt.show()

# # Main function
# def main():
#     ecg_file_path = r'C:\Document\sc2024\data\testData\100001_ECG'  # without the .dat or .hea extension
#     annotations_file_path = 'c:\Document\sc2024\data\testData\100001_ANN.csv'
    
#     signal, fs = load_ecg(ecg_file_path)
#     annotations = read_annotations(annotations_file_path)
#     plot_ecg(signal, fs, annotations)

# if __name__ == '__main__':
#     main()

import wfdb
import pandas as pd
import matplotlib.pyplot as plt

# Function to load ECG data
def load_ecg(file_path):
    record = wfdb.rdrecord(file_path)
    return record.p_signal[:,0], record.fs  # Assuming ECG is the first channel

# Function to read quality annotations
def read_annotations(file_path):
    return pd.read_csv(file_path, header=None, names=['start', 'end', 'quality'])

# Plot ECG with quality annotations
def plot_ecg(signal, fs, annotations):
    time_axis = [i / fs for i in range(len(signal))]
    plt.figure(figsize=(14, 4))
    plt.plot(time_axis, signal, label='ECG Signal', linewidth=0.5)
    
    # Highlight different quality segments for quality 1 and 2
    for _, row in annotations.iterrows():
        if row['quality'] in [1, 2]:  # Focus only on quality levels 1 and 2
            start_time = row['start'] / fs
            end_time = row['end'] / fs
            color = 'green' if row['quality'] == 1 else 'yellow'
            plt.axvspan(start_time, end_time, color=color, alpha=0.5, label=f'Quality {row["quality"]}')

    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('ECG Signal with Quality Annotations')
    plt.legend()
    plt.show()

# Main function
def main():
    ecg_file_path = r'C:\Document\sc2024\data\testData\100001_ECG'  # without the .dat or .hea extension
    annotations_file_path = r'c:\Document\sc2024\data\testData\100001_ANN.csv'
    
    signal, fs = load_ecg(ecg_file_path)
    annotations = read_annotations(annotations_file_path)
    plot_ecg(signal, fs, annotations)

if __name__ == '__main__':
    main()
