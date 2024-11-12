import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from collections import deque

# 加载数据
filePath = r"C:\Document\sc2024\250 kun HR.csv"
data = pd.read_csv(filePath, sep=';')

data['ecg'] = data['ecg'].str.replace(',', '.').astype(float)
data['ecg'] = data['ecg'].fillna(0)  # 将 NaN 填充为 0
data['abp[mmHg]'] = data['abp[mmHg]'].str.replace(',', '.').astype(float)

# 提取 ECG 和 AP 信号
ecg = data['ecg'].values
ap = data['abp[mmHg]'].values

# 设置窗口长度
window_length = 1000  # 滑动窗口的长度

# 初始化滑动窗口，使用 deque 以保持高效
ecgWindow = deque(maxlen=window_length)
apWindow = deque(maxlen=window_length)

# 初始化可视化
plt.ion()  # 开启交互模式
fig, ax = plt.subplots(2, 1, figsize=(10, 8))  # 创建两个子图

# 定义两个子图的曲线
line1, = ax[0].plot([], [], label='ECG Signal', color='b')
line2, = ax[1].plot([], [], label='AP Signal', color='r')

# 设置图例和标题
ax[0].legend()
ax[0].set_title('ECG Signal')
ax[1].legend()
ax[1].set_title('Arterial Pressure (AP) Signal')

plt.show()

# 模拟实时数据流
for i in range(len(ecg)):
    # 将新数据点加入滑动窗口
    ecgWindow.append(ecg[i])
    apWindow.append(ap[i])
    
    # 只有当窗口中数据点足够时才进行可视化更新
    if len(ecgWindow) == window_length and len(apWindow) == window_length:
        # 将窗口数据转化为 numpy 数组
        ecg_window_data = np.array(ecgWindow)
        ap_window_data = np.array(apWindow)

        # 计算窗口的起始和结束索引
        start_idx = i - window_length + 1
        end_idx = i + 1

        # 更新 ECG 信号的可视化
        line1.set_data(np.arange(start_idx, end_idx), ecg_window_data)
        
        # 更新 AP 信号的可视化
        line2.set_data(np.arange(start_idx, end_idx), ap_window_data)
        
        # 设置 X 轴范围，使其随着窗口滑动
        ax[0].set_xlim(start_idx, end_idx - 1)
        ax[1].set_xlim(start_idx, end_idx - 1)

        # 更新轴的范围
        ax[0].relim()
        ax[0].autoscale_view()
        ax[1].relim()
        ax[1].autoscale_view()

        # 暂停以模拟实时更新
        plt.pause(0.01)

# 关闭交互模式，显示最终的图
plt.ioff()
plt.show()
