import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilities import *

def lms_filter(signal, delay=1, mu=0.01, filter_order=4):
    """
    实现 LMS 自适应滤波器
    参数:
    - signal: 输入的 ECG 信号
    - delay: 延迟的步数，用于创建期望信号
    - mu: 学习速率（步长）
    - filter_order: 滤波器阶数
    
    返回:
    - filtered_signal: 去伪影后的信号
    - error: 误差信号
    """
    n = len(signal)
    filtered_signal = np.zeros(n)
    error = np.zeros(n)
    weights = np.zeros(filter_order)
    
    # 创建延迟的期望信号
    d = np.roll(signal, delay)
    
    for i in range(filter_order, n):
        # 获取当前的输入向量
        x = signal[i-filter_order:i][::-1]
        
        # 滤波器输出
        y = np.dot(weights, x)
        
        # 计算误差
        error[i] = d[i] - y
        
        # 更新滤波器权重
        weights += 2 * mu * error[i] * x
        
        # 保存滤波后的信号
        filtered_signal[i] = y
    
    return filtered_signal, error

filePath = r"C:\Document\sc2024\ICM+_data.csv"
data = pd.read_csv(filePath, sep=';')

# 填充 NaN 值为 0
data['ecg'] = data['ecg'].str.replace(',', '.').astype(float)
data['ecg'] = data['ecg'].fillna(0)  # 将 NaN 填充为 0
data['abp[mmHg]'] = data['abp[mmHg]'].str.replace(',', '.').astype(float)

# extract ECG signal
ecg = data['ecg'].values
ecg = ecg[0:50000]

onlyFilterSignal = bandpass_filter(ecg, 0.5, 45, 250)


# 应用 LMS 滤波器
filtered_signal, error = lms_filter(onlyFilterSignal, delay=2, mu=0.01, filter_order=8)

plt.figure(figsize=(12, 10))

# 可视化结果
plt.subplot(2,1,1)
plt.plot(ecg, label='Noisy ECG Signal', alpha=0.7)
plt.plot(filtered_signal, label='LMS Filtered Signal', linestyle='--')
plt.title('ECG Signal Denoising using LMS Filter')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
# plt.show()

plt.subplot(2,1,2)
plt.plot(error, label='Error Signal')
plt.title('Error Signal of LMS Filter')
plt.xlabel('Time')
plt.ylabel('Error')
plt.grid()
plt.show()
