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
    # ecg_file_path = r'C:\Document\sc2024\brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0\105001/105001_ECG'  # without the .dat or .hea extension
    # annotations_file_path = r'C:\Document\sc2024\brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0\105001/105001_ANN.csv'
    # output_file = r"C:\Document\sc2024\filtered_ecg_with_quality_window_105001.csv"

    ecg_file_path = r'c:\Document\sc2024\brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0\111001\111001_ECG'  # without the .dat or .hea extension
    annotations_file_path = r'c:\Document\sc2024\brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0\111001\111001_ANN.csv'
    output_file = r"C:\Document\sc2024\filtered_ecg_with_quality_window_111001.csv"


    # ecg_file_path = r'c:\Document\sc2024\brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0\100001\100001_ECG'  # without the .dat or .hea extension
    # annotations_file_path = r'c:\Document\sc2024\brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0\100001\100001_ANN.csv'    
    # output_file = r"C:\Document\sc2024\filtered_ecg_with_quality_window_100001.csv"
    
    ecg_signal, fs = load_ecg(ecg_file_path)
    # ecg_signal = ecg_signal[47323501:]
    # annotations = read_annotations(annotations_file_path)
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
    qualityResult = 1

    # 假设您的interval CSV文件为intervals.csv
    interval_csv = annotations_file_path
    with open(interval_csv, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)  # 转为列表计算长度
        total_lines = len(rows)

    # 1. 读取intervals文件
    with open(interval_csv, 'r', newline='') as f:
        reader = csv.reader(f)
        label_intervals = deque(maxlen=total_lines)
        # label_intervals = deque(maxlen=2912)

        for row in reader:
            # print(row)
            # 检查 row 是否有足够的元素，以及最后三列是否有值
            if len(row) >= 3 and all(row[-3:]):  # 确保最后三列都有值
                    start = int(row[-3])  # 转换为整数
                    end = int(row[-2])
                    label = int(row[-1])
                    label_intervals.append((start, end, label))
    print(len(label_intervals))



    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 增加label列
        writer.writerow(["sample_index", "ecg", "quality", "label"])

    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        (start, end, lbl) = label_intervals.popleft()

        current_label = lbl
        for i in range(len(ecg_signal)):
            # filtered_ecg, zi_ecg = ziFilter(sos_ecg, ecg_signal[i], zi_ecg)
            ecgFilteredWindow.append(ecg_signal[i])
            # ecgFilteredWindow.append(filtered_ecg[0])
            if (i % overlap_length == 0):
                qualityResult = fixThreshold(list(ecgFilteredWindow), fs)
                if qualityResult == 1:
                    qualityResult, snr, kurtosis, skewness = dynamicThreshold(
                        list(ecgFilteredWindow), fs,
                        kur_min, kur_max, 
                        ske_min, ske_max,
                        snr_min, snr_max)
                    
                    n_kurtosis, mean_kurtosis, M2_kurtosis, std_kurtosis = update_mean_std(
                        n_kurtosis, mean_kurtosis, M2_kurtosis, kurtosis)
                    n_skewness, mean_skewness, M2_skewness, std_skewness = update_mean_std(
                        n_skewness, mean_skewness, M2_skewness, skewness)
                    n_snr, mean_snr, M2_snr, std_snr = update_mean_std(
                        n_snr, mean_snr, M2_snr, snr)

                    kur_min = mean_kurtosis - 2 * std_kurtosis
                    kur_max = mean_kurtosis + 2 * std_kurtosis
                    ske_min = mean_skewness - 2 * std_skewness
                    ske_max = mean_skewness + 2 * std_skewness
                    # snr_min = mean_snr - 2 * std_snr
                    snr_min = max(mean_snr - 2 * std_snr, -50)
                    snr_max = mean_snr + 2 * std_snr
                    # print("kur_min",kur_min,"ske_min",ske_min,"snr_min",snr_min)

                    
                    # kur_min = mean_kurtosis - 1.5 * std_kurtosis
                    # kur_max = mean_kurtosis + 1.5 * std_kurtosis
                    # ske_min = mean_skewness - 1.5 * std_skewness
                    # ske_max = mean_skewness + 1.5 * std_skewness
                    # snr_min = mean_snr - 1.5 * std_snr
                    # snr_max = mean_snr + 1.5 * std_snr

                writer.writerow([i, ecg_signal[i], qualityResult, current_label])
            if (i+1) == end:
                (start, end, lbl) = label_intervals.popleft()
                current_label = lbl



if __name__ == '__main__':
    main()
