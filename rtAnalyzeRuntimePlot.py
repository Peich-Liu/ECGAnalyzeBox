import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from scipy import signal
import csv

# 定义滤波器系数函数
def filter2Sos(low, high, fs=1000, order=4):
    nyquist = fs / 2
    sos = signal.butter(order, [low / nyquist, high / nyquist], btype='band', output='sos')
    return sos

# 定义实时滤波函数
def ziFilter(sos, data_point, zi):
    filtered_point, zi = signal.sosfilt(sos, [data_point], zi=zi)
    return filtered_point, zi

low = 0.5
high = 45
sos = filter2Sos(low, high)

# 初始化滤波器状态
zi = signal.sosfilt_zi(sos)

# 读取数据
filePath = r"C:\Document\sc2024\250 kun HR.csv"
data = pd.read_csv(filePath, sep=';')

data['ecg'] = data['ecg'].str.replace(',', '.').astype(float)
data['ecg'] = data['ecg'].fillna(0)

# 提取 ECG 信号
ecg = data['ecg'].values
ecg = ecg[151000:157000]

# 设置窗口长度
window_length = 1000

# 初始化滑动窗口和数据队列
ecgWindow = deque(maxlen=window_length)
ecgFilteredWindow = deque(maxlen=window_length)
rrInterval = deque([0] * window_length, maxlen=window_length)

# 定义信号质量阈值
signal_threshold = 1.0  # 可以调整阈值

# 初始化 CSV 文件并写入表头
output_file = r"C:\Document\sc2024\filtered_ecg_with_quality.csv"
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["sample_index", "ecg", "filtered_ecg", "rr_interval", "quality"])

# 初始化 R 波检测所需的变量
r_peaks = []

# 初始化可视化
plt.ion()
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
line1, = ax[0].plot([], [], label='Filtered ECG Signal', color='b')
line2, = ax[1].plot([], [], label='RR Intervals', color='g')

ax[0].legend()
ax[0].set_title('Filtered ECG Signal')
ax[1].legend()
ax[1].set_title('RR Intervals')

plt.show()

# 模拟实时数据流
with open(output_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    
    for i in range(len(ecg)):
        # 将新数据点加入滑动窗口
        ecgWindow.append(ecg[i])

        # 滤波 ECG 信号
        filtered_ecg, zi = ziFilter(sos, ecg[i], zi)
        ecgFilteredWindow.append(filtered_ecg[0])

        # 检测信号质量
        quality = "Good" if abs(filtered_ecg[0]) < signal_threshold else "Bad"

        # 检测 R 波
        ecg_window_data = np.array(ecgFilteredWindow)
        peaks, _ = signal.find_peaks(ecg_window_data, distance=200, height=np.mean(ecg_window_data) * 1.2)

        # 计算 RR 间期
        if len(peaks) > 1:
            rr_interval = peaks[-1] - peaks[-2]
        else:
            rr_interval = 0
        rrInterval.append(rr_interval)

        # 实时写入 CSV 文件
        writer.writerow([i, ecg[i], filtered_ecg[0], rr_interval, quality])

        # 实时更新图表
        if len(ecgFilteredWindow) == window_length:
            ecg_window_data = np.array(ecgFilteredWindow)
            rr_window = np.array(rrInterval)

            start_idx = i - window_length + 1
            end_idx = i + 1

            line1.set_data(np.arange(start_idx, end_idx), ecg_window_data)
            line2.set_data(np.arange(start_idx, end_idx), rr_window)

            ax[0].set_xlim(start_idx, end_idx - 1)
            ax[1].set_xlim(start_idx, end_idx - 1)
            ax[0].set_xlabel("time (sample)")
            ax[1].set_xlabel("time (sample)")
            ax[0].set_ylabel("ecg (mv)")
            ax[1].set_ylabel("rr interval (ms)")

            ax[0].relim()
            ax[0].autoscale_view()
            ax[1].relim()
            ax[1].autoscale_view()

            plt.pause(0.01)

plt.ioff()
plt.show()
