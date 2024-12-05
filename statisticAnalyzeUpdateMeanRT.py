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
import numpy as np

def update_mean_std(n_old, mean_old, M2_old, x_new):
    n_new = n_old + 1
    delta = x_new - mean_old
    mean_new = mean_old + delta / n_new
    delta2 = x_new - mean_new
    M2_new = M2_old + delta * delta2

    if n_new < 2:
        std_new = 0.0
    else:
        variance_new = M2_new / (n_new - 1)
        std_new = np.sqrt(variance_new)

    return n_new, mean_new, M2_new, std_new

# Function to load ECG data
def load_ecg(file_path):
    record = wfdb.rdrecord(file_path)
    return record.p_signal[:,0], record.fs  # Assuming ECG is the first channel

# Function to read quality annotations
def read_annotations(file_path):
    return pd.read_csv(file_path, header=None, names=['start', 'end', 'quality'])

# Main function
def main():
    # 初始化峰度的统计变量
    n_kurtosis = 0
    mean_kurtosis = 0.0
    M2_kurtosis = 0.0

    # 初始化偏度的统计变量
    n_skewness = 0
    mean_skewness = 0.0
    M2_skewness = 0.0

    # 初始化信噪比的统计变量
    n_snr = 0
    mean_snr = 0.0
    M2_snr = 0.0
    ecg_file_path = r'C:\Document\sc2024\brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0\105001/105001_ECG'  # without the .dat or .hea extension
    annotations_file_path = r'C:\Document\sc2024\brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0\105001/105001_ANN.csv'

    # ecg_file_path = r'c:\Document\sc2024\brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0\111001\111001_ECG'  # without the .dat or .hea extension
    # annotations_file_path = r'c:\Document\sc2024\brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0\111001\111001_ANN.csv'

    # ecg_file_path = r'c:\Document\sc2024\brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0\100001\100001_ECG'  # without the .dat or .hea extension
    # annotations_file_path = r'c:\Document\sc2024\brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0\100001\100001_ANN.csv'    
    
    ecg_signal, fs = load_ecg(ecg_file_path)
    # ecg_signal = ecg_signal[47323501:]
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
    snr_min = 10
    snr_max = None

    window_length = 4000
    overlap_length = 2000  
    ecgFilteredWindow = deque(maxlen=window_length)
    qualityResult = "Good"

    # output_file = r"C:\Document\sc2024/filtered_ecg_with_qualitynew.csv"
    output_file = r"C:\Document\sc2024\filtered_ecg_with_quality105001.csv"

    
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["sample_index", "ecg", "quality"])

    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        for i in range(len(ecg_signal)):
            # filtered_ecg, zi_ecg = ziFilter(sos_ecg, ecg_signal[i], zi_ecg)
            # ecgFilteredWindow.append(filtered_ecg[0])
            ecgFilteredWindow.append(ecg_signal[i])


            if(i % overlap_length == 0):
                #fix threshold
                qualityResult = fixThreshold(list(ecgFilteredWindow), fs)
                if qualityResult == "Good":
                    qualityResult, snr, kurtosis, skewness = dynamicThreshold(list(ecgFilteredWindow), fs,
                                                                    kur_min, kur_max, 
                                                                    ske_min, ske_max,
                                                                    snr_min, snr_max)
                    # update mean and std
                    n_kurtosis, mean_kurtosis, M2_kurtosis, std_kurtosis = update_mean_std(
                        n_kurtosis, mean_kurtosis, M2_kurtosis, kurtosis)
                    n_skewness, mean_skewness, M2_skewness, std_skewness = update_mean_std(
                        n_skewness, mean_skewness, M2_skewness, skewness)
                    n_snr, mean_snr, M2_snr, std_snr = update_mean_std(
                        n_snr, mean_snr, M2_snr, snr)
                    
                    # update the parameter
                    kur_min = mean_kurtosis - 2 * std_kurtosis
                    kur_max = mean_kurtosis + 2 * std_kurtosis
                    ske_min = mean_skewness - 2 * std_skewness
                    ske_max = mean_skewness + 2 * std_skewness
                    # snr_min = max(mean_snr - 2 * std_snr, 0)
                    snr_min = mean_snr - 2 * std_snr
                    snr_max = mean_snr + 2 * std_snr


            writer.writerow([i, ecg_signal[i], qualityResult])


if __name__ == '__main__':
    main()
