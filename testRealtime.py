import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from scipy import signal

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
data['ecg'] = data['ecg'].fillna(0)  # 将 NaN 填充为 0

# 提取 ECG 信号
ecg = data['ecg'].values
ecg = ecg[375000:1799788]

# 设置窗口长度
window_length = 1000  # 滑动窗口的长度

# 初始化滑动窗口，使用 deque 以保持高效
ecgWindow = deque(maxlen=window_length)
ecgFilteredWindow = deque(maxlen=window_length)
rrInterval = deque([0] * window_length, maxlen=window_length)

# 初始化 R 波检测所需的变量
r_peaks = []

# 初始化可视化
plt.ion()  # 开启交互模式
fig, ax = plt.subplots(2, 1, figsize=(10, 8))  # 创建两个子图

# 定义两个子图的曲线
line1, = ax[0].plot([], [], label='Filtered ECG Signal', color='b')
line2, = ax[1].plot([], [], label='RR Intervals', color='g')

# 设置图例和标题
ax[0].legend()
ax[0].set_title('Filtered ECG Signal')
ax[1].legend()
ax[1].set_title('RR Intervals')

plt.show()

# 模拟实时数据流
for i in range(len(ecg)):
    # 将新数据点加入滑动窗口
    ecgWindow.append(ecg[i])

    # 滤波 ECG 信号
    filtered_ecg, zi = ziFilter(sos, ecg[i], zi)
    ecgFilteredWindow.append(filtered_ecg[0])

    # 检测 R 波
    # if len(ecgFilteredWindow) == window_length:

    ecg_window_data = np.array(ecgFilteredWindow)
    peaks, _ = signal.find_peaks(ecg_window_data, distance=200, height=np.mean(ecg_window_data) * 1.2)  # 设置距离和高度条件

    # 计算 RR 间期并更新到 rrInterval 中
    if len(peaks) > 1:

        if len(peaks) >= 2:
            rr_interval = peaks[-1] - peaks[-2]

        rrInterval.append(rr_interval)
        print(rrInterval)
    else:
        rrInterval.append(0)

    if len(ecgFilteredWindow) == window_length:
        # 将窗口数据转化为 numpy 数组
        ecg_window_data = np.array(ecgFilteredWindow)
        rr_window = np.array(rrInterval)

        # 计算窗口的起始和结束索引
        start_idx = i - window_length + 1
        end_idx = i + 1

        line1.set_data(np.arange(start_idx, end_idx), ecg_window_data)
        line2.set_data(np.arange(start_idx, end_idx), rr_window)

        # 设置 X 轴范围，使其随着窗口滑动
        ax[0].set_xlim(start_idx, end_idx - 1)
        ax[1].set_xlim(start_idx, end_idx - 1)
        ax[0].set_xlabel("time (sample)")
        ax[1].set_xlabel("time (sample)")
        ax[0].set_ylabel("ecg (mv)")
        ax[1].set_ylabel("rr interval (ms)")

        # 更新轴的范围
        ax[0].relim()
        ax[0].autoscale_view()
        ax[1].relim()
        ax[1].autoscale_view()

        plt.pause(0.01)

plt.ioff()
plt.show()