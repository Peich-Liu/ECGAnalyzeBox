import wfdb
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from scipy import signal
from scipy.stats import kurtosis as calc_kurtosis, skew as calc_skew
import csv

from utilities import signalQualityEva, fixThreshold, dynamicThreshold, bandPass, filter2Sos, ziFilter
from scipy import signal

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
    plt.show()

# Main function
def main():
    ecg_file_path = r'C:\Document\sc2024\data\testData\100001_ECG'  # without the .dat or .hea extension
    annotations_file_path = r'c:\Document\sc2024\data\testData\100001_ANN.csv'
    
    ecg_signal, fs = load_ecg(ecg_file_path)
    annotations = read_annotations(annotations_file_path)
    # plot_ecg(ecg_signal, fs, annotations)

    all_kurtosis = []  
    all_skewness = []  
    all_snr = []
    low_ecg = 0.5
    high_ecg = 40
    low_abp = 0.5
    high_abp = 20
    sos_ecg = filter2Sos(low_ecg, high_ecg)
    sos_abp = filter2Sos(low_abp, high_abp)

    zi_ecg = signal.sosfilt_zi(sos_ecg)
    zi_abp = signal.sosfilt_zi(sos_abp)

    #thresholds

    kur_min=2
    kur_max= 4
    ske_min=-1
    ske_max=1

    window_length = 1000
    overlap_length = 500  
    ecgFilteredWindow = deque(maxlen=window_length)
    qualityResult = "Good"

    output_file = r"C:\Document\sc2024/filtered_ecg_with_qualitynew.csv"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["sample_index", "ecg", "filtered_ecg", "quality"])

    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        for i in range(len(ecg_signal)):
            filtered_ecg, zi_ecg = ziFilter(sos_ecg, ecg_signal[i], zi_ecg)
            # ecgFilteredWindow.append(filtered_ecg[0])
            ecgFilteredWindow.append(ecg_signal[i])


            if(i % overlap_length == 0):
                #fix threshold
                qualityResult = fixThreshold(list(ecgFilteredWindow))
                if qualityResult == "Good":
                    #动态阈值, [mu-2sigma, mu+2sigma], 95%
                    mean_kurtosis = np.mean(all_kurtosis)
                    std_kurtosis = np.std(all_kurtosis)
                    kur_min = mean_kurtosis - 2 * std_kurtosis
                    kur_max = mean_kurtosis + 2 * std_kurtosis

                    mean_skewness = np.mean(all_skewness)
                    std_skewness = np.std(all_skewness)
                    ske_min = mean_skewness - 2 * std_skewness
                    ske_max = mean_skewness + 2 * std_skewness
                    
                    mean_snr = np.mean(all_snr)
                    std_snr = np.std(all_snr)
                    # snr_min = mean_snr - 2 * std_snr
                    snr_min = max(mean_snr - 2 * std_snr, 0)
                    snr_max = mean_snr + 2 * std_snr

                    qualityResult, snr, kurtosis, skewness = dynamicThreshold(list(ecgFilteredWindow),
                                                                    kur_min, kur_max, 
                                                                    ske_min, ske_max,
                                                                    snr_min, snr_max)
                    all_kurtosis.append(kurtosis)  # 动态记录
                    all_skewness.append(skewness)  
                    all_snr.append(snr)

            writer.writerow([i, ecg_signal[i], filtered_ecg[0], qualityResult])


if __name__ == '__main__':
    main()
