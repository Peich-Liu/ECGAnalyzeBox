import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from scipy import signal
from scipy.stats import kurtosis as calc_kurtosis, skew as calc_skew
import csv
from utilities import *
# PQRST波形计算和SNR估计
def calculate_average_pqrst(pqrst_list):
    """
    计算平均PQRST波形
    """

    pqrst_array = np.array(pqrst_list)
    # print("pqrst_list",pqrst_array)
    return np.mean(pqrst_array, axis=0)

def calculate_snr(pqrst_list, average_pqrst):
    """
    计算信噪比SNR
    """
    snr_values = []
    for beat in pqrst_list:
        noise = beat - average_pqrst
        signal_power = np.mean(average_pqrst**2)
        noise_power = np.mean(noise**2)
        snr = 10 * np.log10(signal_power / noise_power)  # SNR单位为dB
        snr_values.append(snr)
    return min(snr_values)  # 返回最小SNR

def filter2Sos(low, high, fs=1000, order=4):
    nyquist = fs / 2
    sos = signal.butter(order, [low / nyquist, high / nyquist], btype='band', output='sos')
    return sos

def ziFilter(sos, data_point, zi):
    filtered_point, zi = signal.sosfilt(sos, [data_point], zi=zi)
    return filtered_point, zi

def compute_z_score(value, mean, std):
    # Compute the z-score 
    return (value - mean) / std if std > 0 else 0

def normalize_signal(signal):
    # 避免分母为零
    if np.max(signal) == np.min(signal):
        return np.zeros_like(signal)
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

def signalQualityEva(window, 
                        threshold_amplitude_range, 
                        zero_cross_min, zero_cross_max, 
                        peak_height, beat_length, 
                        kur_min, kur_max, 
                        ske_min, ske_max,
                        snr_min, snr_max):
    # Normalize the window to [0, 1]
    quality = "Good" 
    window = normalize_signal(window)
    # quality = "Good" if (snr > snr_min) else "Bad"  # 设定10 dB为良好信号的门限
    

    # flat line check
    amplitude_range = np.max(window) - np.min(window)
    if amplitude_range < threshold_amplitude_range:
        # return "Bad"
        print("flat line check")
        quality = "Bad"

    # pure noise check（Zero Crossing Rate (零交叉率)）
    zero_crossings = np.sum(np.diff(window > np.mean(window)) != 0)
    if zero_crossings < zero_cross_min or zero_crossings > zero_cross_max:
        quality = "Bad"
        print("Zero Crossing")
        
        # return "Bad"

    # QRS detection
    peaks, _ = signal.find_peaks(window, height=peak_height, distance=beat_length)
    if len(peaks) < 2:
        print("QRS detection")
        quality = "Bad"
        # return "Bad"
    
    # Kurtosis (峰度)
    kurtosis = calc_kurtosis(window)
    # all_kurtosis.append(kurtosis)  # 动态记录
    if kurtosis < kur_min or kurtosis > kur_max:
        print("kurtosis")
        quality = "Bad"
        # return "Bad"

    #Skewness (偏度)
    skewness = calc_skew(window)
    # all_skewness.append(skewness)  
    if skewness < ske_min or skewness > ske_max:
        print("skewness")
        quality = "Bad"

    return quality, kurtosis, skewness


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
threshold_amplitude_range=0.1     
zero_cross_min=5
zero_cross_max= 50
peak_height=0.6
beat_length=100
kur_min=2
kur_max= 4
ske_min=-1
ske_max=1


filePath = r"/Users/liu/Documents/SC2024fall/250 kun HR.csv"
data = pd.read_csv(filePath, sep=';')

data['ecg'] = data['ecg'].str.replace(',', '.').astype(float)
data['ecg'] = data['ecg'].fillna(0)
ecg = data['ecg'].values
ecg = ecg

data['abp[mmHg]'] = data['abp[mmHg]'].str.replace(',', '.').astype(float)
data['abp[mmHg]'] = data['abp[mmHg]'].fillna(0)
abp = data['abp[mmHg]'].values
abp = abp

window_length = 1000
overlap_length = 500  
ecgFilteredWindow = deque(maxlen=window_length)
abpFilteredWindow = deque(maxlen=window_length)
rrInterval = deque([0] * window_length, maxlen=window_length)

output_file = r"/Users/liu/Documents/SC2024fall/filtered_ecg_with_qualitytest.csv"
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["sample_index", "ecg", "filtered_ecg", "filtered_abp","rr_interval", "quality"])

with open(output_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    
    for i in range(len(ecg)):
        # print("i",i)
        # # Pre-processing
        filtered_ecg, zi_ecg = ziFilter(sos_ecg, ecg[i], zi_ecg)
        ecgFilteredWindow.append(filtered_ecg[0])
        filtered_abp, zi_abp = ziFilter(sos_abp, abp[i], zi_abp)
        abpFilteredWindow.append(filtered_abp[0])
        if(i % overlap_length == 0):
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

            quality, kurtosis, skewness = signalQualityEva(list(ecgFilteredWindow), 
                                        threshold_amplitude_range, 
                                        zero_cross_min, zero_cross_max, 
                                        peak_height, beat_length, 
                                        kur_min, kur_max,
                                        ske_min, ske_max,
                                        snr_min, snr_max)
            all_kurtosis.append(kurtosis)  # 动态记录
            all_skewness.append(skewness)  
            # all_snr.append(snr)
            print("i", i, "quality", quality)
            
        

        # RR interval calculation
        ecg_window_data = np.array(ecgFilteredWindow)
        peaks, _ = signal.find_peaks(ecg_window_data, distance=200, height=np.mean(ecg_window_data) * 1.2)

        if len(peaks) > 1:
            rr_interval = np.diff(peaks)  # 计算所有相邻 R 波之间的间隔
        else:
            rr_interval = [] 

        writer.writerow([i, ecg[i], filtered_ecg[0],filtered_abp[0], rr_interval, quality])