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

from utilities import signalQualityEva, fixThreshold, dynamicThreshold2, bandPass, filter2Sos, ziFilter
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
    ecg_file_path = r'/Users/liu/Documents/SC2024fall/testdata/100001_ECG'  # without the .dat or .hea extension
    annotations_file_path = r'/Users/liu/Documents/SC2024fall/testdata/100001_ANN.csv'
    
    ecg_signal, fs = load_ecg(ecg_file_path)
    annotations = read_annotations(annotations_file_path)
    # plot_ecg(ecg_signal, fs, annotations)

    all_kurtosis = []  
    all_skewness = []  
    all_snr = []
    fs = 1000
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

    window_length = 4000
    overlap_length = 2000  
    ecgFilteredWindow = deque(maxlen=window_length)
    qualityResult = "Good"

    output_file = r"/Users/liu/Documents/SC2024fall/filtered_ecg_with_qualitynew.csv"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["sample_index", "ecg", "filtered_ecg", "quality"])

    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        for i in range(len(ecg_signal)):
            filtered_ecg, zi_ecg = ziFilter(sos_ecg, ecg_signal[i], zi_ecg)
            ecgFilteredWindow.append(filtered_ecg[0])
            # ecgFilteredWindow.append(ecg_signal[i])
            # 计算峰度的均值和标准差
            kurtosis_mean = np.mean(all_kurtosis)
            kurtosis_std = np.std(all_kurtosis)
            
            # 计算偏度的均值和标准差
            skewness_mean = np.mean(all_skewness)
            skewness_std = np.std(all_skewness)
            
            # 计算 95% 和 99% 置信区间阈值
            z_score_95 = 1.96
            z_score_99 = 3.291
    
            if(i % overlap_length == 0):
                #fix threshold
                qualityResult = fixThreshold(list(ecgFilteredWindow), fs)
                if qualityResult == "Good":
                    kur_lower_95 = kurtosis_mean - z_score_95 * kurtosis_std
                    kur_upper_95 = kurtosis_mean + z_score_95 * kurtosis_std
                    kur_lower_99 = kurtosis_mean - z_score_99 * kurtosis_std
                    kur_upper_99 = kurtosis_mean + z_score_99 * kurtosis_std
                    
                    # 偏度阈值
                    ske_lower_95 = skewness_mean - z_score_95 * skewness_std
                    ske_upper_95 = skewness_mean + z_score_95 * skewness_std
                    ske_lower_99 = skewness_mean - z_score_99 * skewness_std
                    ske_upper_99 = skewness_mean + z_score_99 * skewness_std
                    
                    snr_lower_95 = skewness_mean - z_score_95 * skewness_std
                    snr_lower_99 = skewness_mean - z_score_99 * skewness_std
                    
                    # 初始化为 "Good"
                    qualityResult, snr, kurtosis, skewness = dynamicThreshold2(list(ecgFilteredWindow),fs,
                                                                    kur_lower_99, kur_lower_95, 
                                                                    kur_upper_99, kur_upper_95,
                                                                    ske_lower_99, ske_lower_95,
                                                                    ske_upper_99, ske_upper_95,
                                                                    snr_lower_99, snr_lower_95)
                    all_kurtosis.append(kurtosis)  # 动态记录
                    all_skewness.append(skewness)  
                    all_snr.append(snr)

            writer.writerow([i, ecg_signal[i], filtered_ecg[0], qualityResult])


if __name__ == '__main__':
    main()
