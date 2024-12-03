# import numpy as np
# import pandas as pd 
# import scipy.signal as signal
# import wfdb
# import os
# from scipy.signal import butter, filtfilt
from scipy import stats

# import matplotlib.pyplot as plt
# import wfdb
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
# from scipy.signal import welch
# from scipy.integrate import simpson
# from statsmodels.tsa.ar_model import AutoReg
# import neurokit2 as nk
# from scipy.signal import butter, filtfilt
# import numpy as np
# import os
# import wfdb
# import matplotlib.pyplot as plt
# import tkinter as tk
# from tkinter import filedialog, simpledialog, messagebox
# from tkinter import ttk
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# from matplotlib.patches import Rectangle
# from datetime import datetime
# from scipy import signal
# from scipy.stats import kurtosis as calc_kurtosis, skew as calc_skew

# def bandPass(data, zi, lowCut, highCut, fs, order=4):
#     nyq = fs
#     low = lowCut / nyq
#     high = highCut / nyq
#     sos = signal.butter(order, [lowCut, highCut], btype='band',output='sos',fs=fs)
#     sig, zi = signal.sosfilt(sos, data, zi=zi)
#     return sig, zi

# def lowPass(data, zi, lowCut, fs, order=4):
#     nyq = fs
#     low = lowCut / nyq
#     sos = signal.butter(order, lowCut, btype='low',output='sos',fs=fs)
#     sig, zi = signal.sosfilt(sos, data, zi=zi)
#     return sig, zi

# def highPass(data, zi, highCut, fs, order=4):
#     nyq = fs
#     high = highCut / nyq
#     sos = signal.butter(order, highCut, btype='high',output='sos',fs=fs)
#     sig, zi = signal.sosfilt(sos, data, zi=zi)
#     return sig, zi

# def filterSignalwithoutMean(data, zi=None, low=None, high=None, fs=250):
#     data = np.array(data)
#     if low != 0 and high != 0:
#         return bandPass(data, zi,low, high, fs, order=4)
#     elif low == 0 and high != 0:
#         return lowPass(data, zi,high, fs, order=4)
#     elif high == 0 and low != 0:
#         return highPass(data, zi,low, fs, order=4)
#     elif low == 0 and high == 0:
#         return data

# def filterSignal(data, zi=None, low=None, high=None, fs=250, order=4, diff=False):
#     data = np.array(data)
#     data -= np.mean(data)
#     if low != 0 and high != 0:
#         return bandPass(data, zi,low, high, fs, order=4)
#     elif low == 0 and high != 0:
#         return lowPass(data, zi,high, fs, order=4)
#     elif high == 0 and low != 0:
#         return highPass(data, zi,low, fs, order=4)
#     elif low == 0 and high == 0:
#         return data

# #Multi-File loading -- it is a test based on mit dataset, so, the data is not right
# def readPatientRecords(patient_id, data_dir):
#     patient_prefix = patient_id[:2]
#     records = []
#     annotations = []

#     for file in os.listdir(data_dir):
#         if file.startswith(patient_prefix) and file.endswith('.dat'):
#             record_base = os.path.splitext(file)[0]
            
#             record = wfdb.rdrecord(os.path.join(data_dir, record_base))
#             #here assume all the fs is same in the same signal
#             fs = record.fs
#             records.append(record)
#             print(record.sig_name)
#             # annotation_file = os.path.join(data_dir, f"{record_base}.atr")
#             # if os.path.exists(annotation_file):
#             #     annotation = wfdb.rdann(os.path.join(data_dir, record_base), 'atr')
#             #     annotations.append(annotation)
    
#     return records, fs, annotations

# #Multi-File loading -- it is a test based on mit dataset, so, the data is not right
# def readPatientRecords2(patient_id, data_dir):
#     patient_prefix = patient_id[:5]
#     records = []
#     annotations = []

#     for file in os.listdir(data_dir):
#         file_base = os.path.splitext(file)[0]
#         if file_base.endswith(patient_prefix) and file.endswith('.dat'):
#             record_base = os.path.splitext(file)[0]
#             load_file = os.path.join(data_dir, record_base)
#             print("loading file", load_file)
#             record = wfdb.rdrecord(load_file)
#             #here assume all the fs is same in the same signal
#             records.append(record)
#             print(record.sig_name)


#             # annotation_file = os.path.join(data_dir, f"{record_base}.atr")
#             # if os.path.exists(annotation_file):
#             #     annotation = wfdb.rdann(os.path.join(data_dir, record_base), 'atr')
#             #     annotations.append(annotation)
    
#     return records, annotations

# def concatenateSignals(records, start, end):
#     allSignals = []
#     for record in records:
#         allSignals.append(record.p_signal[:, 0])

#     concatenatedSignal = np.concatenate(allSignals)
#     return concatenatedSignal[start:end]

# def bandpass_filter(signal, low_cutoff, high_cutoff, sampling_rate):
#     # Design a bandpass filter with the given cutoffs and sampling rate
#     nyquist = 0.5 * sampling_rate
#     low = low_cutoff / nyquist
#     high = high_cutoff / nyquist
#     b, a = butter(4, [low, high], btype='band')
#     # Apply the filter to the signal using filtfilt to avoid phase distortion
#     print("low, high, len", low, high, len(signal))

#     filtered_signal = filtfilt(b, a, signal)
#     return filtered_signal

# def remove_artifacts(signal, threshold=0.5):
#     # Simple artifact removal by thresholding: replace values above a threshold with the median of the signal
#     median_value = np.median(signal)
#     signal[np.abs(signal) > threshold] = median_value
#     return signal

# def concatenateSignals(records, start, end, low_cutoff=0.5, high_cutoff=10, sampling_rate=250):
#     allSignals = []
#     for record in records:
#         # Extract the ECG signal from each record (assuming the signal is in the first channel)
#         allSignals.append(record.p_signal)
#     concatenatedSignal = np.concatenate(allSignals, axis=0)
#     # Return the filtered and artifact-processed segment of the signal from start to end
#     return concatenatedSignal[start:end, :]

# def concatenateECG(records, start, end, low_cutoff=0.5, high_cutoff=10, sampling_rate=250):
#     allSignals = []
#     for record in records:
#         # Extract the ECG signal from each record (assuming the signal is in the first channel)
#         ecgIndex = record.sig_name.index('ECG')
#         print(ecgIndex)
#         allSignals.append(record.p_signal[:, ecgIndex])
#     # Concatenate all extracted signals into a single signal
#     concatenatedSignal = np.concatenate(allSignals)
#     return concatenatedSignal[start:end]

# def concatenateAP(records, start, end, low_cutoff=0.5, high_cutoff=10, sampling_rate=250):
#     allSignals = []
#     for record in records:
#         # Extract the ECG signal from each record (assuming the signal is in the first channel)
#         apIndex = record.sig_name.index('BP')
#         allSignals.append(record.p_signal[:, apIndex])
#     # Concatenate all extracted signals into a single signal
#     concatenatedSignal = np.concatenate(allSignals)

#     return concatenatedSignal[start:end]

# def concatenateandProcessSignals(records, start, end, low_cutoff=0.5, high_cutoff=10, sampling_rate=250):
#     allSignals = []
#     for record in records:
#         # Extract the ECG signal from each record (assuming the signal is in the first channel)
#         allSignals.append(record.p_signal[:, 0])

#     # Concatenate all extracted signals into a single signal
#     concatenatedSignal = np.concatenate(allSignals)
#     ## Here, add the filter and artifacts processing for concatenatedSignal[start:end]
#     filteredSignal = bandpass_filter(concatenatedSignal[start:end], low_cutoff, high_cutoff, sampling_rate)

#     # Artifacts processing: Apply an artifact removal technique, such as thresholding or signal interpolation
#     processedSignal = remove_artifacts(filteredSignal)

#     # Return the filtered and artifact-processed segment of the signal from start to end
#     return processedSignal

# def visualizeSignal(signal):
#     time = range(len(signal))
    
#     plt.figure(figsize=(12, 6))
#     plt.plot(time, signal)
#     plt.title('Concatenated ECG Signal Visualization')
#     plt.xlabel('Time (samples)')
#     plt.ylabel('Amplitude')
#     plt.grid(True)
#     plt.show()


# def visualizeSignalinGUI(signal):
#     # Plot the ECG signal in the GUI
#     figure = plt.Figure(figsize=(10, 4), dpi=100)
#     ax = figure.add_subplot(111)
#     ax.plot(signal)
#     ax.set_title('ECG Signal')
#     ax.set_xlabel('Samples')
#     ax.set_ylabel('Amplitude')
#     ax.grid()

#     return figure


# def visualizeSignalinGUIMultiChannel(signals):
#     # Plot the ECG signal in the GUI
#     num_channels = signals.shape[1]
#     figure, axes = plt.subplots(num_channels, 1, figsize=(10, 6), sharex=True)

#     if num_channels == 1:
#         axes = [axes]

#     for i in range(num_channels):
#         axes[i].plot(signals[:, i])
#         axes[i].set_title(f'Channel {i+1}')
#         axes[i].set_ylabel('Amplitude')
    
#     axes[-1].set_xlabel('Samples')

#     figure.tight_layout()
#     return figure

# def visualizeSignalinGUISelectChannel(signals, selected_channel):
#     figure = plt.Figure(figsize=(10, 6))
#     ax = figure.add_subplot(111)

#     ax.plot(signals[:, selected_channel], label=f'Channel {selected_channel + 1}')
#     ax.set_title(f'ECG Signal Visualization - Channel {selected_channel + 1}')
#     ax.set_xlabel('Samples')
#     ax.set_ylabel('Amplitude')
#     ax.legend()

#     return figure

# def calculateEcgSignalProperties(signal, fs):
#     #under constraction

#     #0. RR interval
#     peaks, _ = find_peaks(signal, distance=200)
#     rrIntervalSamples = np.diff(peaks)
#     samplingRate = fs
#     rrIntervalsSecond = rrIntervalSamples / samplingRate

#     # 1. Mean Heart Rate (HR)
#     hr = 60 / rrIntervalsSecond
#     meanHR = np.mean(hr)

#     # 2. SDNN (Standard deviation of RR intervals)
#     sdnn = np.std(rrIntervalsSecond)

#     # 3. RMSSD (Root Mean Square of Successive Differences)
#     rrDiff = np.diff(rrIntervalsSecond)  # Differences between successive RR intervals
#     rmssd = np.sqrt(np.mean(rrDiff**2))

#     # 4. SD1/SD2, put here temporily 
#     sd1, sd2, rr_mean = calculate_sd1_sd2(rrIntervalSamples)
#     figure = plot_poincare(rrIntervalSamples, sd1, sd2, rr_mean)

#     properties = {
#         # "RR Interval Second": rrIntervalsSecond,
#         "Heart Rate":meanHR,
#         "SDNN":sdnn,
#         "RMSSD":rmssd,
#     }
#     return properties, figure

# def calculateApSignalProperties(signal, fs):
#     #1. Systolic and Diastolic
#     sbp = np.max(signal) #Systolic
#     dbp = np.min(signal) #Diastolic
#     pp = sbp - dbp      

#     #2. Mean arterial pressure
#     map = (sbp + 2 * dbp) / 3

#     #3. SD of Pressures
#     sd = np.std(signal)

#     properties = {
#     "Systolic":sbp,
#     "Diastolic":dbp,
#     "Mean arterial pressure":map,
#     "SD of Pressures":sd,
#     }
    
#     return properties
# # def convert_timestamp_to_index(timestamp, fs):
# #     # 假设时间戳格式是 "%H:%M:%S"
# #     time_format = "%H:%M:%S"
# #     time_in_seconds = (datetime.strptime(timestamp, time_format) - datetime(1900, 1, 1)).total_seconds()
# #     return int(time_in_seconds * fs)

# def calculateEcgSignalRangeProperties(signal, fs, selected_ranges):
#     # print("selected_ranges",selected_ranges)
#     results = {}

#     for index, (start, end) in selected_ranges.items():
#         # 将时间戳转换为索引
#         segment = signal[start:end]


#         # start_idx = convert_timestamp_to_index(start, fs)
#         # end_idx = convert_timestamp_to_index(end, fs)
#         # segment = signal[start_idx:end_idx]

#         #under constraction

#         #0. RR interval
#         peaks, _ = find_peaks(segment, distance=200)
#         rrIntervalSamples = np.diff(peaks)
#         samplingRate = fs
#         rrIntervalsSecond = rrIntervalSamples / samplingRate

#         # 1. Mean Heart Rate (HR)
#         hr = 60 / rrIntervalsSecond
#         meanHR = np.mean(hr)

#         # 2. SDNN (Standard deviation of RR intervals)
#         sdnn = np.std(rrIntervalsSecond)

#         # 3. RMSSD (Root Mean Square of Successive Differences)
#         rrDiff = np.diff(rrIntervalsSecond)  # Differences between successive RR intervals
#         rmssd = np.sqrt(np.mean(rrDiff**2))

#         # 4. SD1/SD2, put here temporily 
#         sd1, sd2, rr_mean = calculate_sd1_sd2(rrIntervalSamples)
#         figure = plot_poincare(rrIntervalSamples, sd1, sd2, rr_mean)

#         results[index] = {
#             # "RR Interval Second": rrIntervalsSecond,
#             "Heart Rate":f"{meanHR:.2f}",
#             "SDNN":f"{sdnn:.2f}",
#             "RMSSD":f"{rmssd:.2f}",
#         }

        
#     return results, figure

# def calculateApSignalRangeProperties(signal, fs, selected_ranges):
#     print("selected_ranges",selected_ranges)
#     results = {}

#     for index, (start, end) in selected_ranges.items():
#         # 提取当前选框的信号片段
#         segment = signal[start:end]
#         # 将时间戳转换为索引
#         # start_idx = convert_timestamp_to_index(start, fs)
#         # end_idx = convert_timestamp_to_index(end, fs)
        
#         # segment = signal[start_idx:end_idx]

#         # 1. Systolic and Diastolic
#         sbp = np.max(segment)  # Systolic
#         dbp = np.min(segment)  # Diastolic
#         pp = sbp - dbp

#         # 2. Mean arterial pressure
#         map_value = (sbp + 2 * dbp) / 3

#         # 3. SD of Pressures
#         sd = np.std(segment)

#         # 将每个框选区域的计算结果存储到字典中
#         results[index] = {
#             "Systolic": f"{sbp:.2f}",
#             "Diastolic": f"{dbp:.2f}",
#             "Mean arterial pressure": f"{map_value:.2f}",
#             "SD of Pressures": f"{sd:.2f}",
#         }
#         # print("resultsAP", results)

#     return results

# def PSDAnalyze(ecgSignal, fs):
#     peaks, info = nk.ecg_peaks(ecgSignal, sampling_rate=fs, correct_artifacts=True)
#     # Get the indices of the detected R-peaks
#     r_peaks_indices = info['ECG_R_Peaks']
#     rr_intervals = np.diff(r_peaks_indices) / 100
#     rr_intervals_mean = np.mean(rr_intervals)
#     rr_intervals_centered = rr_intervals - rr_intervals_mean

#     #Welch function
#     freq, psd = welch(rr_intervals_centered, fs=4, nperseg=256)

#     #AR function
#     # order = 16 
#     # ar_model = AutoReg(rr_intervals_centered, lags=order).fit()
#     # ar_coefficients = ar_model.params
#     # sampling_rate = 4 
#     # freq, response = signal.freqz([1], np.concatenate(([1], -ar_coefficients[1:])), worN=512, fs=sampling_rate)
#     # psd = np.abs(response) ** 2

#     mask = (freq >= 0) & (freq <= 0.5)
#     freq = freq[mask]
#     psd = psd[mask]

#     return freq, psd

# def calculate_sd1_sd2(rr_intervals):
#     # Calculate mean of RR intervals
#     rr_mean = np.mean(rr_intervals)

#     # Calculate RR interval pairs (RR_n, RR_n+1)
#     rr_n = rr_intervals[:-1]
#     rr_n1 = rr_intervals[1:]

#     # Calculate SD1 and SD2
#     diff_rr = rr_n1 - rr_n
#     sd1 = np.sqrt(np.var(diff_rr) / 2)
#     sd2 = np.sqrt(2 * np.var(rr_intervals) - sd1 ** 2)

#     return sd1, sd2, rr_mean

# def plot_poincare(rr_intervals, sd1, sd2, rr_mean):
#     rr_n = rr_intervals[:-1]
#     rr_n1 = rr_intervals[1:]

#     figure = plt.figure(figsize=(3,3))
#     plt.scatter(rr_n, rr_n1, c='b', alpha=0.6, label='RR Intervals')

#     # Plot average RR line
#     plt.axvline(x=rr_mean, color='g', linestyle='--', label='Avg R-R interval')
#     plt.axhline(y=rr_mean, color='g', linestyle='--')

#     # Plot SD1 and SD2 ellipse
#     angle = 45  # The angle for SD1 and SD2
#     ellipse = plt.matplotlib.patches.Ellipse((rr_mean, rr_mean), 2*sd2, 2*sd1, angle=angle,
#                                              edgecolor='r', facecolor='none', linestyle='-', linewidth=2, label='SD1/SD2 Ellipse')
#     plt.gca().add_patch(ellipse)

#     plt.xlabel('RR(n) (seconds)')
#     plt.ylabel('RR(n+1) (seconds)')
#     plt.title('Poincare Plot')
#     plt.legend()
#     plt.grid(True)
#     plt.axis('equal')
#     return figure

# def visualPSD(freq, psd):
#     ulf_range = (0.0, 0.0033)
#     vlf_range = (0.0033, 0.04)
#     lf_range = (0.04, 0.15)
#     hf_range = (0.15, 0.4)
#     vhf_range = (0.4, 0.5)

#     figure = plt.figure(figsize=(4, 2))
#     ax = figure.add_subplot(111)  # 添加子图

#     ax.plot(freq, psd, color='black', linewidth=1, label='PSD')

#     ax.fill_between(freq, psd, where=((freq >= ulf_range[0]) & (freq <= ulf_range[1])),
#                     color='purple', alpha=0.5, label='ULF')
#     ax.fill_between(freq, psd, where=((freq >= vlf_range[0]) & (freq <= vlf_range[1])),
#                     color='blue', alpha=0.5, label='VLF')
#     ax.fill_between(freq, psd, where=((freq >= lf_range[0]) & (freq <= lf_range[1])),
#                     color='green', alpha=0.5, label='LF')
#     ax.fill_between(freq, psd, where=((freq >= hf_range[0]) & (freq <= hf_range[1])),
#                     color='orange', alpha=0.5, label='HF')
#     ax.fill_between(freq, psd, where=((freq >= vhf_range[0]) & (freq <= vhf_range[1])),
#                     color='red', alpha=0.5, label='VHF')

#     ax.set_title('PSD for Frequency Domains')
#     ax.set_xlabel('Frequency (Hz)')
#     ax.set_ylabel('Spectrum (ms^2/Hz)')

#     # 将图例放在图表的外部右侧
#     ax.legend(loc='upper left', bbox_to_anchor=(1, 1))  # 图例在图外右侧
#     figure.tight_layout()  # 自动调整子图以适应图例
#     return figure

# def getFigure(figure, canvas_frame):
#     # Clear previous plot if it exists
#     for widget in canvas_frame.winfo_children():
#         widget.destroy()
#     # Embed the plot in the Tkinter canvas
#     canvas = FigureCanvasTkAgg(figure, master=canvas_frame)
#     canvas.draw()
#     canvas_widget = canvas.get_tk_widget()
#     canvas_widget.pack(fill=tk.BOTH, expand=True)

#     # Add navigation toolbar for better interactivity
#     toolbar = NavigationToolbar2Tk(canvas, canvas_frame)
#     toolbar.update()
#     toolbar.pack(side=tk.BOTTOM, fill=tk.X)

# def plot_signals(cba_instance, canvas_frame):
#     figure = plt.figure(figsize=(8, 6), dpi=100)

#     # Create two subplots
#     ax1 = figure.add_subplot(211)
#     ax2 = figure.add_subplot(212, sharex=ax1)  # Share the x-axis with ax1

#     # Plot ECG signal
#     ax1.plot(np.arange(len(cba_instance.ecg_signal)), cba_instance.ecg_signal, label="ECG Signal", color='b')
#     ax1.set_title("Loaded ECG Signal")
#     ax1.set_ylabel("Amplitude")
#     ax1.legend()

#     # Plot AP signal
#     ax2.plot(np.arange(len(cba_instance.ap_signal)), cba_instance.ap_signal, label="AP Signal", color='r')
#     ax2.set_title("Loaded AP Signal")
#     ax2.set_xlabel("Time (samples)")
#     ax2.set_ylabel("Amplitude")
#     ax2.legend()

#     # Use the getFigure function to display the figure
#     getFigure(figure, canvas_frame)

# def plot_can_interact(cba_instance, canvas_frame):
#     fig = plt.figure(figsize=(5, 3), dpi=100)
#     ax1 = fig.add_subplot(211)
#     ax2 = fig.add_subplot(212, sharex=ax1)
    
#     x = np.arange(len(cba_instance.ecg_signal))
#     ax1.plot(x, cba_instance.ecg_signal, label="ECG Signal")
#     ax1.set_ylabel("ECG Amplitude")
#     ax1.legend()

#     ax2.plot(x, cba_instance.ap_signal, label="AP Signal", color="orange")
#     ax2.set_xlabel("Time (samples)")
#     ax2.set_ylabel("AP Amplitude")
#     ax2.legend()
    
#     # 将图形嵌入到Tkinter的Canvas中
#     canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
#     canvas.draw()
#     canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
#     toolbar = NavigationToolbar2Tk(canvas, canvas_frame)
#     toolbar.update()
#     canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
#     # 初始化矩形框变量
#     rect1 = None
#     rect2 = None
#     is_drawing = False
#     start_x = None

#     # 定义事件处理函数
#     def on_press(event):
#         nonlocal is_drawing, start_x, rect1, rect2
#         if event.inaxes not in [ax1, ax2]:  # 确保事件在绘图区内
#             return
#         is_drawing = True
#         start_x = event.xdata
#         # 创建同步的矩形框
#         rect1 = Rectangle((start_x, ax1.get_ylim()[0]), 0, np.diff(ax1.get_ylim())[0],
#                           edgecolor='r', facecolor='none')
#         rect2 = Rectangle((start_x, ax2.get_ylim()[0]), 0, np.diff(ax2.get_ylim())[0],
#                           edgecolor='r', facecolor='none')
#         ax1.add_patch(rect1)
#         ax2.add_patch(rect2)
    
#     def on_drag(event):
#         nonlocal rect1, rect2
#         if not is_drawing or event.inaxes not in [ax1, ax2]:
#             return
#         width = event.xdata - start_x
#         rect1.set_width(width)
#         rect2.set_width(width)
#         canvas.draw()
    
#     def on_release(event):
#         nonlocal is_drawing
#         is_drawing = False
#         x_min = int(rect1.get_x())
#         x_max = int(rect1.get_x() + rect1.get_width())
        
#         # 计算选定时间区间内的平均值
#         ecg_mean = np.mean(cba_instance.ecg_signal[x_min:x_max]) if x_max > x_min else 0
#         ap_mean = np.mean(cba_instance.ap_signal[x_min:x_max]) if x_max > x_min else 0
#         messagebox.showinfo("平均值", f"时间区间内ECG信号平均值: {ecg_mean:.2f}\n时间区间内AP信号平均值: {ap_mean:.2f}")

#     # 绑定鼠标事件
#     canvas.mpl_connect("button_press_event", on_press)
#     canvas.mpl_connect("motion_notify_event", on_drag)
#     canvas.mpl_connect("button_release_event", on_release)

# def displaySignalProperties(properties, properties_frame):
#     # Clear previous properties if they exist
#     for widget in properties_frame.winfo_children():
#         widget.destroy()

#     # Display signal properties
#     row = 0
#     for key, value in properties.items():
#         ttk.Label(properties_frame, text=f"{key}: {value}").grid(row=row, column=0, padx=10, pady=5, sticky=tk.W)
#         row += 1

# # def return_range(ranges):
# #     print("ranges:", ranges)
# #     return ranges
# # 示例计算函数
# # def calculate_mean(ranges):
# #     print("计算平均值:", ranges)
# #     # 执行计算逻辑...

# # def calculate_variance(ranges):
# #     print("计算方差:", ranges)

# def show_windowInfo(ranges):
#     print("show_windowInfo",ranges)



# def loadData(filePath):
#     filePath = r"C:\Document\sc2024\250 kun HR.csv"
#     data = pd.read_csv(filePath, sep=';')

#     data['ecg'] = data['ecg'].str.replace(',', '.').astype(float)
#     data['ecg'] = data['ecg'].fillna(0)  # 将 NaN 填充为 0
#     data['abp[mmHg]'] = data['abp[mmHg]'].str.replace(',', '.').astype(float)

#     # extract ECG signal
#     ecg = data['ecg'].values
#     ap = data['abp[mmHg]'].values

#     return ecg, ap
# # PQRST波形计算和SNR估计
# def calculate_average_pqrst(pqrst_list):
#     """
#     计算平均PQRST波形
#     """

#     pqrst_array = np.array(pqrst_list)
#     # print("pqrst_list",pqrst_array)
#     return np.mean(pqrst_array, axis=0)

# def calculate_snr(pqrst_list, average_pqrst):
#     """
#     计算信噪比SNR
#     """
#     snr_values = []
#     for beat in pqrst_list:
#         noise = beat - average_pqrst
#         signal_power = np.mean(average_pqrst**2)
#         noise_power = np.mean(noise**2)
#         snr = 10 * np.log10(signal_power / noise_power)  # SNR单位为dB
#         snr_values.append(snr)
#     return min(snr_values)  # 返回最小SNR

# def filter2Sos(low, high, fs=1000, order=4):
#     nyquist = fs / 2
#     sos = signal.butter(order, [low / nyquist, high / nyquist], btype='band', output='sos')
#     return sos

# def ziFilter(sos, data_point, zi):
#     filtered_point, zi = signal.sosfilt(sos, [data_point], zi=zi)
#     return filtered_point, zi

# def compute_z_score(value, mean, std):
#     # Compute the z-score 
#     return (value - mean) / std if std > 0 else 0

# def normalize_signal(signal):
#     if np.max(signal) == np.min(signal):
#         return np.zeros_like(signal)
#     return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
# def pan_tompkins(ecg_signal, fs):
#     """
#     使用Pan-Tompkins算法检测QRS波。

#     参数：
#     ecg_signal : numpy array
#         原始心电信号。
#     fs : int
#         心电信号的采样频率。

#     返回：
#     qrs_peaks : numpy array
#         检测到的QRS波的位置索引。
#     """

#     # 步骤1：带通滤波（5-15 Hz）
#     lowcut = 5.0
#     highcut = 15.0
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = signal.butter(1, [low, high], btype='band')
#     ecg_band = signal.lfilter(b, a, ecg_signal)

#     # 步骤2：微分
#     ecg_diff = np.diff(ecg_band)

#     # 步骤3：平方
#     ecg_squared = ecg_diff ** 2

#     # 步骤4：移动窗口积分
#     window_size = int(0.150 * fs)  # 窗口大小为150毫秒
#     integration_window = np.ones(window_size) / window_size
#     ecg_integrated = np.convolve(ecg_squared, integration_window, mode='same')

#     # 步骤5：检测峰值
#     threshold = np.mean(ecg_integrated) * 1.5
#     distance = int(0.2 * fs)  # 相邻QRS波之间的最小距离为200毫秒
#     peaks, _ = signal.find_peaks(ecg_integrated, height=threshold, distance=distance)

#     return peaks

# def signalQualityEva(window, 
#                         threshold_amplitude_range, 
#                         zero_cross_min, zero_cross_max, 
#                         peak_height, beat_length, 
#                         kur_min, kur_max, 
#                         ske_min, ske_max,
#                         snr_min, snr_max):
#     # Normalize the window to [0, 1]
#     quality = "Good" 
#     window = normalize_signal(window)
#     # quality = "Good" if (snr > snr_min) else "Bad"  # 设定10 dB为良好信号的门限
    

#     # flat line check
#     amplitude_range = np.max(window) - np.min(window)
#     if amplitude_range < threshold_amplitude_range:
#         # return "Bad"
#         print("flat line check")
#         quality = "Bad"

#     # pure noise check（Zero Crossing Rate (零交叉率)）
#     zero_crossings = np.sum(np.diff(window > np.mean(window)) != 0)
#     if zero_crossings < zero_cross_min or zero_crossings > zero_cross_max:
#         quality = "Bad"
#         print("Zero Crossing")
        
#         # return "Bad"

#     # # QRS detection
#     # peaks, _ = signal.find_peaks(window, height=peak_height, distance=beat_length)
#     # if len(peaks) < 2:
#     #     print("QRS detection")
#     #     quality = "Bad"
#     #     # return "Bad"
    
#     # Kurtosis (峰度)
#     kurtosis = calc_kurtosis(window)
#     # all_kurtosis.append(kurtosis)  # 动态记录
#     if kurtosis < kur_min or kurtosis > kur_max:
#         print("kurtosis")
#         quality = "Bad"
#         # return "Bad"

#     #Skewness (偏度)
#     skewness = calc_skew(window)
#     # all_skewness.append(skewness)  
#     if skewness < ske_min or skewness > ske_max:
#         print("skewness")
#         quality = "Bad"

#     return quality, kurtosis, skewness

# def fixThreshold(window):
#     threshold_amplitude_range=0.1     
#     zero_cross_min=5
#     zero_cross_max= 50
#     peak_height=0.6
#     beat_length=100

#     quality = "Good" 
#     window = normalize_signal(window)
    
#     # flat line check
#     amplitude_range = np.max(window) - np.min(window)
#     if amplitude_range < threshold_amplitude_range:
#         print("flat line check")
#         quality = "Bad"

#     # pure noise check（Zero Crossing Rate (零交叉率)）
#     zero_crossings = np.sum(np.diff(window > np.mean(window)) != 0)
#     if zero_crossings < zero_cross_min or zero_crossings > zero_cross_max:
#         quality = "Bad"
#         print("Zero Crossing")

#     # # QRS detection
#     # peaks, _ = signal.find_peaks(window, height=peak_height, distance=beat_length)
#     # if len(peaks) < 2:
#     #     print("QRS detection")
#     #     quality = "Bad"

#     return quality

# def dynamicThreshold(window,
#                     kur_min, kur_max, 
#                     ske_min, ske_max,
#                     snr_min, snr_max):
    
#     quality = "Good"
#     #SNR calculation
#     peaks_snr, _ = signal.find_peaks(window, distance=200, height=np.mean(window) * 1.2)
#     pqrst_list = [list(window)[max(0, peak-50):min(len(window), peak+50)] for peak in peaks_snr]
#     pqrst_list = [wave for wave in pqrst_list if len(wave) == 100]
    
#     if len(pqrst_list) > 1:
#         average_pqrst = calculate_average_pqrst(pqrst_list)
#         snr = calculate_snr(pqrst_list, average_pqrst)
#     else:
#         snr = 0  # 若PQRST提取失败

#     # if snr < snr_min:
#     #     print("snr")
#     #     quality = "Consider"

#     # Kurtosis (峰度) calculation
#     kurtosis = calc_kurtosis(window)

#     # all_kurtosis.append(kurtosis)
#     if kurtosis < kur_min or kurtosis > kur_max:
#         print("kurtosis")
#         quality = "Consider"


#     #Skewness (偏度)
#     skewness = calc_skew(window)
#     # all_skewness.append(skewness)  
#     if skewness < ske_min or skewness > ske_max:
#         print("skewness")
#         quality = "Consider"

#     return quality, snr, kurtosis, skewness

def detect_r_peaks(ecg_signal, fs):
    """
    从ECG信号中检测R峰。

    参数:
    ecg_signal : array-like
        ECG信号数据。
    fs : float
        采样频率，单位：Hz。

    返回:
    r_peaks_indices : array
        R峰的位置索引。
    r_peaks_times : array
        R峰的时间，单位：秒。
    """
    # 滤波（带通滤波器，移除低频和高频噪声）
    sos = signal.butter(2, [0.5, 15], btype='bandpass', fs=fs, output='sos')
    # filtered_ecg = signal.sosfiltfilt(sos, ecg_signal)
    filtered_ecg = ecg_signal

    # 寻找R峰
    distance = int(0.2 * fs)  # 假设心率不低于每分钟50次，两个R峰之间的最小距离为0.2秒
    # r_peaks_indices, _ = signal.find_peaks(filtered_ecg, distance=distance, height=np.mean(filtered_ecg))
    r_peaks_indices, _ = signal.find_peaks(filtered_ecg, distance=200, height=np.mean(filtered_ecg) * 1.2)

    r_peaks_times = r_peaks_indices / fs
    return r_peaks_indices, r_peaks_times

def compute_rr_intervals(r_peaks_times):
    """
    计算RR间期。

    参数:
    r_peaks_times : array
        R峰的时间，单位：秒。

    返回:
    rr_intervals : array
        RR间期序列，单位：毫秒。
    rr_times : array
        RR间期对应的时间（取两个R峰的中点），单位：秒。
    """
    rr_intervals = np.diff(r_peaks_times) * 1000  # 转换为毫秒
    rr_times = r_peaks_times[:-1] + np.diff(r_peaks_times) / 2
    return rr_intervals, rr_times

def detect_sbp(ap_signal, fs):
    """
    从AP信号中检测收缩压（SBP）。

    参数:
    ap_signal : array-like
        动脉压力信号数据。
    fs : float
        采样频率，单位：Hz。

    返回:
    sbp_values : array
        收缩压值序列，单位：mmHg。
    sbp_times : array
        收缩压对应的时间，单位：秒。
    """
    # 寻找收缩压峰值
    distance = int(0.2 * fs)  # 假设心率不低于每分钟50次
    sbp_indices, _ = signal.find_peaks(ap_signal, distance=200)
    sbp_values = ap_signal[sbp_indices]
    sbp_times = sbp_indices / fs
    return sbp_values, sbp_times

def synchronize_rr_sbp(rr_times, rr_intervals, sbp_times, sbp_values):
    """
    同步RR间期和SBP序列。

    参数:
    rr_times : array
        RR间期对应的时间，单位：秒。
    rr_intervals : array
        RR间期序列，单位：毫秒。
    sbp_times : array
        SBP对应的时间，单位：秒。
    sbp_values : array
        SBP值序列，单位：mmHg。

    返回:
    rr_intervals_sync : array
        同步后的RR间期序列。
    sbp_values_sync : array
        同步后的SBP值序列。
    """
    # 使用线性插值对RR间期和SBP进行同步
    common_times = np.union1d(rr_times, sbp_times)
    rr_interp = np.interp(common_times, rr_times, rr_intervals)
    sbp_interp = np.interp(common_times, sbp_times, sbp_values)
    return rr_interp, sbp_interp
def compute_brs_sequence(rr_intervals, sbp_values):
    """
    使用序列法计算BRS。

    参数:
    rr_intervals : array-like
        RR间期时间序列（单位：毫秒）
    sbp_values : array-like
        收缩压时间序列（单位：mmHg）

    返回:
    brs_sequences : list of dictionaries
        包含每个序列的信息：斜率、相关系数、起始索引、结束索引。
    mean_brs : float
        BRS平均值（斜率的平均值）
    """
    rr_intervals = np.asarray(rr_intervals)
    sbp_values = np.asarray(sbp_values)
    n = len(rr_intervals)

    # 初始化变量
    brs_sequences = []

    # 遍历信号
    i = 0
    while i < n - 2:
        # 检查上升序列（RR和SBP同时增加）
        if sbp_values[i+1] > sbp_values[i] and rr_intervals[i+1] > rr_intervals[i]:
            # 上升序列的起点
            seq_sbp = [sbp_values[i], sbp_values[i+1]]
            seq_rr = [rr_intervals[i], rr_intervals[i+1]]
            start_idx = i
            j = i + 2
            while j < n and sbp_values[j] > sbp_values[j-1] and rr_intervals[j] > rr_intervals[j-1]:
                seq_sbp.append(sbp_values[j])
                seq_rr.append(rr_intervals[j])
                j += 1
            end_idx = j - 1
            if len(seq_sbp) >= 3:
                # 线性回归
                slope, intercept, r_value, p_value, std_err = stats.linregress(seq_sbp, seq_rr)
                brs_sequences.append({
                    'slope': slope,
                    'r_value': r_value,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
            i = end_idx  # 从序列的末尾继续
        # 检查下降序列（RR和SBP同时减少）
        elif sbp_values[i+1] < sbp_values[i] and rr_intervals[i+1] < rr_intervals[i]:
            # 下降序列的起点
            seq_sbp = [sbp_values[i], sbp_values[i+1]]
            seq_rr = [rr_intervals[i], rr_intervals[i+1]]
            start_idx = i
            j = i + 2
            while j < n and sbp_values[j] < sbp_values[j-1] and rr_intervals[j] < rr_intervals[j-1]:
                seq_sbp.append(sbp_values[j])
                seq_rr.append(rr_intervals[j])
                j += 1
            end_idx = j -1
            if len(seq_sbp) >= 3:
                # 线性回归
                slope, intercept, r_value, p_value, std_err = stats.linregress(seq_sbp, seq_rr)
                brs_sequences.append({
                    'slope': slope,
                    'r_value': r_value,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
            i = end_idx  # 从序列的末尾继续
        else:
            i +=1

    # 计算平均BRS（相关系数绝对值大于等于0.85的序列的斜率平均值）
    slopes = [seq['slope'] for seq in brs_sequences if abs(seq['r_value']) >= 0.85]
    if slopes:
        mean_brs = np.mean(slopes)
    else:
        mean_brs = np.nan  # 如果没有找到有效的序列，返回NaN

    return brs_sequences, mean_brs

def synchronize_rr_sbp_cycle_based(r_peaks_times, rr_intervals, sbp_times, sbp_values):
    """
    基于心动周期的同步方法，将RR间期和SBP值同步。

    参数:
    r_peaks_times : array
        R峰的时间，单位：秒。
    rr_intervals : array
        RR间期序列，单位：毫秒。
    sbp_times : array
        SBP对应的时间，单位：秒。
    sbp_values : array
        SBP值序列，单位：mmHg。

    返回:
    rr_intervals_sync : array
        同步后的RR间期序列。
    sbp_values_sync : array
        同步后的SBP值序列。
    rr_times_sync : array
        RR间期对应的时间，单位：秒。
    """
    rr_intervals_sync = []
    sbp_values_sync = []
    rr_times_sync = []

    for i in range(len(rr_intervals)):
        # 定义心动周期的时间范围
        start_time = r_peaks_times[i]
        end_time = r_peaks_times[i+1] if i+1 < len(r_peaks_times) else r_peaks_times[i] + rr_intervals[i]/1000

        # 在该时间范围内寻找SBP峰值
        mask = (sbp_times >= start_time) & (sbp_times < end_time)
        sbp_in_cycle = sbp_values[mask]
        sbp_times_in_cycle = sbp_times[mask]

        if len(sbp_in_cycle) > 0:
            # 如果有多个SBP峰值，取第一个或平均值
            sbp_value = sbp_in_cycle[0]  # 或者使用np.mean(sbp_in_cycle)
            sbp_time = sbp_times_in_cycle[0]
            rr_intervals_sync.append(rr_intervals[i])
            sbp_values_sync.append(sbp_value)
            rr_times_sync.append((start_time + end_time) / 2)
        else:
            # # 如果没有找到SBP值，可以选择跳过或插值处理
            pass
            # rr_intervals_sync.append()


    return np.array(rr_intervals_sync), np.array(sbp_values_sync), np.array(rr_times_sync)
import wfdb
import os
import numpy as np
import pandas as pd 
import scipy.signal as signal
import matplotlib.pyplot as plt
import tkinter as tk
import neurokit2 as nk

from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
from scipy.signal import welch
from scipy.integrate import simpson
from statsmodels.tsa.ar_model import AutoReg
from tkinter import filedialog, simpledialog, messagebox
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.dates import num2date
from matplotlib.patches import Rectangle
from datetime import datetime
from decimal import Decimal, getcontext


def bandPass(data, zi, lowCut, highCut, fs, order=4):
    nyq = fs
    low = lowCut / nyq
    high = highCut / nyq
    sos = signal.butter(order, [lowCut, highCut], btype='band',output='sos',fs=fs)
    sig, zi = signal.sosfilt(sos, data, zi=zi)
    return sig, zi

def lowPass(data, zi, lowCut, fs, order=4):
    nyq = fs
    low = lowCut / nyq
    sos = signal.butter(order, lowCut, btype='low',output='sos',fs=fs)
    sig, zi = signal.sosfilt(sos, data, zi=zi)
    return sig, zi

def highPass(data, zi, highCut, fs, order=4):
    nyq = fs
    high = highCut / nyq
    sos = signal.butter(order, highCut, btype='high',output='sos',fs=fs)
    sig, zi = signal.sosfilt(sos, data, zi=zi)
    return sig, zi

def filterSignalwithoutMean(data, zi=None, low=None, high=None, fs=250):
    data = np.array(data)
    if low != 0 and high != 0:
        return bandPass(data, zi,low, high, fs, order=4)
    elif low == 0 and high != 0:
        return lowPass(data, zi,high, fs, order=4)
    elif high == 0 and low != 0:
        return highPass(data, zi,low, fs, order=4)
    elif low == 0 and high == 0:
        return data

def filterSignal(data, zi=None, low=None, high=None, fs=250, order=4, diff=False):
    data = np.array(data)
    data -= np.mean(data)
    if low != 0 and high != 0:
        return bandPass(data, zi,low, high, fs, order=4)
    elif low == 0 and high != 0:
        return lowPass(data, zi,high, fs, order=4)
    elif high == 0 and low != 0:
        return highPass(data, zi,low, fs, order=4)
    elif low == 0 and high == 0:
        return data

#Multi-File loading -- it is a test based on mit dataset, so, the data is not right
def readPatientRecords(patient_id, data_dir):
    patient_prefix = patient_id[:2]
    records = []
    annotations = []

    for file in os.listdir(data_dir):
        if file.startswith(patient_prefix) and file.endswith('.dat'):
            record_base = os.path.splitext(file)[0]
            
            record = wfdb.rdrecord(os.path.join(data_dir, record_base))
            #here assume all the fs is same in the same signal
            fs = record.fs
            records.append(record)
            print(record.sig_name)
            # annotation_file = os.path.join(data_dir, f"{record_base}.atr")
            # if os.path.exists(annotation_file):
            #     annotation = wfdb.rdann(os.path.join(data_dir, record_base), 'atr')
            #     annotations.append(annotation)
    
    return records, fs, annotations

#Multi-File loading -- it is a test based on mit dataset, so, the data is not right
def readPatientRecords2(patient_id, data_dir):
    patient_prefix = patient_id[:5]
    records = []
    annotations = []

    for file in os.listdir(data_dir):
        file_base = os.path.splitext(file)[0]
        if file_base.endswith(patient_prefix) and file.endswith('.dat'):
            record_base = os.path.splitext(file)[0]
            load_file = os.path.join(data_dir, record_base)
            print("loading file", load_file)
            record = wfdb.rdrecord(load_file)
            #here assume all the fs is same in the same signal
            records.append(record)
            print(record.sig_name)


            # annotation_file = os.path.join(data_dir, f"{record_base}.atr")
            # if os.path.exists(annotation_file):
            #     annotation = wfdb.rdann(os.path.join(data_dir, record_base), 'atr')
            #     annotations.append(annotation)
    
    return records, annotations

def concatenateSignals(records, start, end):
    allSignals = []
    for record in records:
        allSignals.append(record.p_signal[:, 0])

    concatenatedSignal = np.concatenate(allSignals)
    return concatenatedSignal[start:end]

def bandpass_filter(signal, low_cutoff, high_cutoff, sampling_rate):
    # Design a bandpass filter with the given cutoffs and sampling rate
    nyquist = 0.5 * sampling_rate
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(4, [low, high], btype='band')
    # Apply the filter to the signal using filtfilt to avoid phase distortion
    print("low, high, len", low, high, len(signal))

    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def remove_artifacts(signal, threshold=0.5):
    # Simple artifact removal by thresholding: replace values above a threshold with the median of the signal
    median_value = np.median(signal)
    signal[np.abs(signal) > threshold] = median_value
    return signal

def concatenateSignals(records, start, end, low_cutoff=0.5, high_cutoff=10, sampling_rate=250):
    allSignals = []
    for record in records:
        # Extract the ECG signal from each record (assuming the signal is in the first channel)
        allSignals.append(record.p_signal)
    concatenatedSignal = np.concatenate(allSignals, axis=0)
    # Return the filtered and artifact-processed segment of the signal from start to end
    return concatenatedSignal[start:end, :]

def concatenateECG(records, start, end, low_cutoff=0.5, high_cutoff=10, sampling_rate=250):
    allSignals = []
    for record in records:
        # Extract the ECG signal from each record (assuming the signal is in the first channel)
        ecgIndex = record.sig_name.index('ECG')
        print(ecgIndex)
        allSignals.append(record.p_signal[:, ecgIndex])
    # Concatenate all extracted signals into a single signal
    concatenatedSignal = np.concatenate(allSignals)
    return concatenatedSignal[start:end]

def concatenateAP(records, start, end, low_cutoff=0.5, high_cutoff=10, sampling_rate=250):
    allSignals = []
    for record in records:
        # Extract the ECG signal from each record (assuming the signal is in the first channel)
        apIndex = record.sig_name.index('BP')
        allSignals.append(record.p_signal[:, apIndex])
    # Concatenate all extracted signals into a single signal
    concatenatedSignal = np.concatenate(allSignals)

    return concatenatedSignal[start:end]

def concatenateandProcessSignals(records, start, end, low_cutoff=0.5, high_cutoff=10, sampling_rate=250):
    allSignals = []
    for record in records:
        # Extract the ECG signal from each record (assuming the signal is in the first channel)
        allSignals.append(record.p_signal[:, 0])

    # Concatenate all extracted signals into a single signal
    concatenatedSignal = np.concatenate(allSignals)
    ## Here, add the filter and artifacts processing for concatenatedSignal[start:end]
    filteredSignal = bandpass_filter(concatenatedSignal[start:end], low_cutoff, high_cutoff, sampling_rate)

    # Artifacts processing: Apply an artifact removal technique, such as thresholding or signal interpolation
    processedSignal = remove_artifacts(filteredSignal)

    # Return the filtered and artifact-processed segment of the signal from start to end
    return processedSignal

def visualizeSignal(signal):
    time = range(len(signal))
    
    plt.figure(figsize=(12, 6))
    plt.plot(time, signal)
    plt.title('Concatenated ECG Signal Visualization')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()


def visualizeSignalinGUI(signal):
    # Plot the ECG signal in the GUI
    figure = plt.Figure(figsize=(10, 4), dpi=100)
    ax = figure.add_subplot(111)
    ax.plot(signal)
    ax.set_title('ECG Signal')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Amplitude')
    ax.grid()

    return figure


def visualizeSignalinGUIMultiChannel(signals):
    # Plot the ECG signal in the GUI
    num_channels = signals.shape[1]
    figure, axes = plt.subplots(num_channels, 1, figsize=(10, 6), sharex=True)

    if num_channels == 1:
        axes = [axes]

    for i in range(num_channels):
        axes[i].plot(signals[:, i])
        axes[i].set_title(f'Channel {i+1}')
        axes[i].set_ylabel('Amplitude')
    
    axes[-1].set_xlabel('Samples')

    figure.tight_layout()
    return figure

def visualizeSignalinGUISelectChannel(signals, selected_channel):
    figure = plt.Figure(figsize=(10, 6))
    ax = figure.add_subplot(111)

    ax.plot(signals[:, selected_channel], label=f'Channel {selected_channel + 1}')
    ax.set_title(f'ECG Signal Visualization - Channel {selected_channel + 1}')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Amplitude')
    ax.legend()

    return figure

def calculateEcgSignalProperties(signal, fs):
    #under constraction

    #0. RR interval
    peaks, _ = find_peaks(signal, distance=200)
    rrIntervalSamples = np.diff(peaks)
    samplingRate = fs
    rrIntervalsSecond = rrIntervalSamples / samplingRate

    # 1. Mean Heart Rate (HR)
    hr = 60 / rrIntervalsSecond
    meanHR = np.mean(hr)

    # 2. SDNN (Standard deviation of RR intervals)
    sdnn = np.std(rrIntervalsSecond)

    # 3. RMSSD (Root Mean Square of Successive Differences)
    rrDiff = np.diff(rrIntervalsSecond)  # Differences between successive RR intervals
    rmssd = np.sqrt(np.mean(rrDiff**2))

    # 4. SD1/SD2, put here temporily 
    sd1, sd2, rr_mean = calculate_sd1_sd2(rrIntervalSamples)
    figure = plot_poincare(rrIntervalSamples, sd1, sd2, rr_mean)

    properties = {
        # "RR Interval Second": rrIntervalsSecond,
        "Heart Rate":meanHR,
        "SDNN":sdnn,
        "RMSSD":rmssd,
    }
    return properties, figure

def calculateApSignalProperties(signal, fs):
    #1. Systolic and Diastolic
    sbp = np.max(signal) #Systolic
    dbp = np.min(signal) #Diastolic
    pp = sbp - dbp      

    #2. Mean arterial pressure
    map = (sbp + 2 * dbp) / 3

    #3. SD of Pressures
    sd = np.std(signal)

    properties = {
    "Systolic":sbp,
    "Diastolic":dbp,
    "Mean arterial pressure":map,
    "SD of Pressures":sd,
    }
    
    return properties
# def convert_timestamp_to_index(timestamp, fs):
#     # 假设时间戳格式是 "%H:%M:%S"
#     time_format = "%H:%M:%S"
#     time_in_seconds = (datetime.strptime(timestamp, time_format) - datetime(1900, 1, 1)).total_seconds()
#     return int(time_in_seconds * fs)

def x_coord_to_seconds(x_coord, start_time="00:00:00",fs = 250):
    # Convert the x-coordinate (matplotlib date number) to a datetime object
    date_time = num2date(x_coord)
    
    # Parse the start time into a datetime object
    start_datetime = datetime.strptime(start_time, "%H:%M:%S")
    
    # Ensure both datetime objects have the same timezone information
    start_datetime = start_datetime.replace(tzinfo=date_time.tzinfo)
    
    # Calculate the time difference
    delta_time = date_time - start_datetime
    
    # Get the total seconds from the time difference
    total_seconds = delta_time.total_seconds()
    
    sample = int(total_seconds*fs)
    
    return sample

# def transSample(start, end, fs=250):
#     start_time = datetime.strptime(start, '%H:%M:%S')
#     end_time = datetime.strptime(end, '%H:%M:%S')
    
#     # 转换为秒数
#     start_seconds = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
#     end_seconds = end_time.hour * 3600 + end_time.minute * 60 + end_time.second

#     start_sample = int(start_seconds * fs)
#     end_sample = int(end_seconds * fs)
#     return start_sample, end_sample
    # sample_point_ranges[index] = (start_sample, end_sample)
        
        # return total_sample
        
        
# def x_coord_to_sample_index(x_coord, start_time="00:00:00", sample_rate=250):
#     # 设置高精度
#     getcontext().prec = 28
    
#     # 将 x_coord 转换为 Decimal
#     x_coord_decimal = Decimal(str(x_coord))
    
#     # 将 x_coord 转换为 datetime 对象
#     date_time = num2date(float(x_coord_decimal))
    
#     # 解析起始时间
#     start_datetime = datetime.strptime(start_time, "%H:%M:%S")
    
#     # 计算时间差（以秒为单位），并转换为 Decimal
#     delta_time = Decimal(str((date_time - start_datetime).total_seconds()))
    
#     # 计算采样间隔
#     sample_interval = Decimal('1') / Decimal(str(sample_rate))
    
#     # 计算样本索引
#     sample_index = delta_time / sample_interval
    
#     # 四舍五入到最近的整数
#     sample_index = int(sample_index.to_integral_value(rounding='ROUND_HALF_UP'))
    
#     return sample_index

def calculateEcgSignalRangeProperties(signal, fs, selected_ranges):
    print("selected_ranges",selected_ranges)
    results = {}
    # selected_ranges[index] = (start, end)
    
    if len(selected_ranges) != 0:
        for index, (rect1, rect2) in selected_ranges.items():
        # for index, (rect1, rect2) in selected_ranges:
            # 将时间戳转换为索引
            start = x_coord_to_seconds(rect1.get_x())
            end = x_coord_to_seconds(rect1.get_x() + rect1.get_width())
            # 提取整数部分（小时）和小数部分
            # startPoint = rect1.get_x()
            # endPoint = rect1.get_x()+ rect1.get_width()
            # start, end = transSample(startPoint, endPoint)
            
            
            print("start",start,"end",end)
            segment = signal[start:end]


            # start_idx = convert_timestamp_to_index(start, fs)
            # end_idx = convert_timestamp_to_index(end, fs)
            # segment = signal[start_idx:end_idx]

            #under constraction

            #0. RR interval
            peaks, _ = find_peaks(segment, distance=200)
            rrIntervalSamples = np.diff(peaks)
            samplingRate = fs
            rrIntervalsSecond = rrIntervalSamples / samplingRate

            # 1. Mean Heart Rate (HR)
            hr = 60 / rrIntervalsSecond
            meanHR = np.mean(hr)

            # 2. SDNN (Standard deviation of RR intervals)
            sdnn = np.std(rrIntervalsSecond)

            # 3. RMSSD (Root Mean Square of Successive Differences)
            rrDiff = np.diff(rrIntervalsSecond)  # Differences between successive RR intervals
            rmssd = np.sqrt(np.mean(rrDiff**2))

            # 4. SD1/SD2, put here temporily 
            sd1, sd2, rr_mean = calculate_sd1_sd2(rrIntervalSamples)
            figure = plot_poincare(rrIntervalSamples, sd1, sd2, rr_mean)

            results[index] = {
                # "RR Interval Second": rrIntervalsSecond,
                "Heart Rate":f"{meanHR:.2f}",
                "SDNN":f"{sdnn:.2f}",
                "RMSSD":f"{rmssd:.2f}",
            }

            
        return results, figure
    else:
        return None, None

def calculateApSignalRangeProperties(signal, fs, selected_ranges):
    # print("selected_ranges",selected_ranges)
    results = {}

    for index, (rect1, _) in selected_ranges.items():
        start = x_coord_to_seconds(rect1.get_x())
        end = x_coord_to_seconds(rect1.get_x() + rect1.get_width())
        # 提取当前选框的信号片段
        segment = signal[start:end]
        # 1. Systolic and Diastolic
        sbp = np.max(segment)  # Systolic
        dbp = np.min(segment)  # Diastolic
        pp = sbp - dbp

        # 2. Mean arterial pressure
        map_value = (sbp + 2 * dbp) / 3

        # 3. SD of Pressures
        sd = np.std(segment)

        # 将每个框选区域的计算结果存储到字典中
        results[index] = {
            "Systolic": f"{sbp:.2f}",
            "Diastolic": f"{dbp:.2f}",
            "Mean arterial pressure": f"{map_value:.2f}",
            "SD of Pressures": f"{sd:.2f}",
        }
        # print("resultsAP", results)

    return results

def PSDAnalyze(ecgSignal, fs):
    peaks, info = nk.ecg_peaks(ecgSignal, sampling_rate=fs, correct_artifacts=True)
    # Get the indices of the detected R-peaks
    r_peaks_indices = info['ECG_R_Peaks']
    rr_intervals = np.diff(r_peaks_indices) / 100
    rr_intervals_mean = np.mean(rr_intervals)
    rr_intervals_centered = rr_intervals - rr_intervals_mean

    #Welch function
    freq, psd = welch(rr_intervals_centered, fs=4, nperseg=256)

    #AR function
    # order = 16 
    # ar_model = AutoReg(rr_intervals_centered, lags=order).fit()
    # ar_coefficients = ar_model.params
    # sampling_rate = 4 
    # freq, response = signal.freqz([1], np.concatenate(([1], -ar_coefficients[1:])), worN=512, fs=sampling_rate)
    # psd = np.abs(response) ** 2

    mask = (freq >= 0) & (freq <= 0.5)
    freq = freq[mask]
    psd = psd[mask]

    return freq, psd

def calculate_sd1_sd2(rr_intervals):
    # Calculate mean of RR intervals
    rr_mean = np.mean(rr_intervals)

    # Calculate RR interval pairs (RR_n, RR_n+1)
    rr_n = rr_intervals[:-1]
    rr_n1 = rr_intervals[1:]

    # Calculate SD1 and SD2
    diff_rr = rr_n1 - rr_n
    sd1 = np.sqrt(np.var(diff_rr) / 2)
    sd2 = np.sqrt(2 * np.var(rr_intervals) - sd1 ** 2)

    return sd1, sd2, rr_mean

def plot_poincare(rr_intervals, sd1, sd2, rr_mean):
    rr_n = rr_intervals[:-1]
    rr_n1 = rr_intervals[1:]

    figure = plt.figure(figsize=(3,3))
    plt.scatter(rr_n, rr_n1, c='b', alpha=0.6, label='RR Intervals')

    # Plot average RR line
    plt.axvline(x=rr_mean, color='g', linestyle='--', label='Avg R-R interval')
    plt.axhline(y=rr_mean, color='g', linestyle='--')

    # Plot SD1 and SD2 ellipse
    angle = 45  # The angle for SD1 and SD2
    ellipse = plt.matplotlib.patches.Ellipse((rr_mean, rr_mean), 2*sd2, 2*sd1, angle=angle,
                                             edgecolor='r', facecolor='none', linestyle='-', linewidth=2, label='SD1/SD2 Ellipse')
    plt.gca().add_patch(ellipse)

    plt.xlabel('RR(n) (seconds)')
    plt.ylabel('RR(n+1) (seconds)')
    plt.title('Poincare Plot')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    return figure

def visualPSD(freq, psd):
    ulf_range = (0.0, 0.0033)
    vlf_range = (0.0033, 0.04)
    lf_range = (0.04, 0.15)
    hf_range = (0.15, 0.4)
    vhf_range = (0.4, 0.5)

    figure = plt.figure(figsize=(4, 2))
    ax = figure.add_subplot(111)  # 添加子图

    ax.plot(freq, psd, color='black', linewidth=1, label='PSD')

    ax.fill_between(freq, psd, where=((freq >= ulf_range[0]) & (freq <= ulf_range[1])),
                    color='purple', alpha=0.5, label='ULF')
    ax.fill_between(freq, psd, where=((freq >= vlf_range[0]) & (freq <= vlf_range[1])),
                    color='blue', alpha=0.5, label='VLF')
    ax.fill_between(freq, psd, where=((freq >= lf_range[0]) & (freq <= lf_range[1])),
                    color='green', alpha=0.5, label='LF')
    ax.fill_between(freq, psd, where=((freq >= hf_range[0]) & (freq <= hf_range[1])),
                    color='orange', alpha=0.5, label='HF')
    ax.fill_between(freq, psd, where=((freq >= vhf_range[0]) & (freq <= vhf_range[1])),
                    color='red', alpha=0.5, label='VHF')

    ax.set_title('PSD for Frequency Domains')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Spectrum (ms^2/Hz)')

    # 将图例放在图表的外部右侧
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))  # 图例在图外右侧
    figure.tight_layout()  # 自动调整子图以适应图例
    return figure

def getFigure(figure, canvas_frame):
    # Clear previous plot if it exists
    for widget in canvas_frame.winfo_children():
        widget.destroy()
    # Embed the plot in the Tkinter canvas
    canvas = FigureCanvasTkAgg(figure, master=canvas_frame)
    canvas.draw()
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)

    # Add navigation toolbar for better interactivity
    toolbar = NavigationToolbar2Tk(canvas, canvas_frame)
    toolbar.update()
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)

def plot_signals(cba_instance, canvas_frame):
    figure = plt.figure(figsize=(8, 6), dpi=100)

    # Create two subplots
    ax1 = figure.add_subplot(211)
    ax2 = figure.add_subplot(212, sharex=ax1)  # Share the x-axis with ax1

    # Plot ECG signal
    ax1.plot(np.arange(len(cba_instance.ecg_signal)), cba_instance.ecg_signal, label="ECG Signal", color='b')
    ax1.set_title("Loaded ECG Signal")
    ax1.set_ylabel("Amplitude")
    ax1.legend()

    # Plot AP signal
    ax2.plot(np.arange(len(cba_instance.ap_signal)), cba_instance.ap_signal, label="AP Signal", color='r')
    ax2.set_title("Loaded AP Signal")
    ax2.set_xlabel("Time (samples)")
    ax2.set_ylabel("Amplitude")
    ax2.legend()

    # Use the getFigure function to display the figure
    getFigure(figure, canvas_frame)

def plot_can_interact(cba_instance, canvas_frame):
    fig = plt.figure(figsize=(5, 3), dpi=100)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    
    x = np.arange(len(cba_instance.ecg_signal))
    ax1.plot(x, cba_instance.ecg_signal, label="ECG Signal")
    ax1.set_ylabel("ECG Amplitude")
    ax1.legend()

    ax2.plot(x, cba_instance.ap_signal, label="AP Signal", color="orange")
    ax2.set_xlabel("Time (samples)")
    ax2.set_ylabel("AP Amplitude")
    ax2.legend()
    
    # 将图形嵌入到Tkinter的Canvas中
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    toolbar = NavigationToolbar2Tk(canvas, canvas_frame)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    # 初始化矩形框变量
    rect1 = None
    rect2 = None
    is_drawing = False
    start_x = None

    # 定义事件处理函数
    def on_press(event):
        nonlocal is_drawing, start_x, rect1, rect2
        if event.inaxes not in [ax1, ax2]:  # 确保事件在绘图区内
            return
        is_drawing = True
        start_x = event.xdata
        # 创建同步的矩形框
        rect1 = Rectangle((start_x, ax1.get_ylim()[0]), 0, np.diff(ax1.get_ylim())[0],
                          edgecolor='r', facecolor='none')
        rect2 = Rectangle((start_x, ax2.get_ylim()[0]), 0, np.diff(ax2.get_ylim())[0],
                          edgecolor='r', facecolor='none')
        ax1.add_patch(rect1)
        ax2.add_patch(rect2)
    
    def on_drag(event):
        nonlocal rect1, rect2
        if not is_drawing or event.inaxes not in [ax1, ax2]:
            return
        width = event.xdata - start_x
        rect1.set_width(width)
        rect2.set_width(width)
        canvas.draw()
    
    def on_release(event):
        nonlocal is_drawing
        is_drawing = False
        x_min = int(rect1.get_x())
        x_max = int(rect1.get_x() + rect1.get_width())
        
        # 计算选定时间区间内的平均值
        ecg_mean = np.mean(cba_instance.ecg_signal[x_min:x_max]) if x_max > x_min else 0
        ap_mean = np.mean(cba_instance.ap_signal[x_min:x_max]) if x_max > x_min else 0
        messagebox.showinfo("平均值", f"时间区间内ECG信号平均值: {ecg_mean:.2f}\n时间区间内AP信号平均值: {ap_mean:.2f}")

    # 绑定鼠标事件
    canvas.mpl_connect("button_press_event", on_press)
    canvas.mpl_connect("motion_notify_event", on_drag)
    canvas.mpl_connect("button_release_event", on_release)

def displaySignalProperties(properties, properties_frame):
    # Clear previous properties if they exist
    for widget in properties_frame.winfo_children():
        widget.destroy()

    # Display signal properties
    row = 0
    for key, value in properties.items():
        ttk.Label(properties_frame, text=f"{key}: {value}").grid(row=row, column=0, padx=10, pady=5, sticky=tk.W)
        row += 1

# def return_range(ranges):
#     print("ranges:", ranges)
#     return ranges
# 示例计算函数
# def calculate_mean(ranges):
#     print("计算平均值:", ranges)
#     # 执行计算逻辑...

# def calculate_variance(ranges):
#     print("计算方差:", ranges)

# def show_windowInfo(ranges):
#     print("show_windowInfo",ranges)



def loadData(filePath):
    filePath = filePath
    data = pd.read_csv(filePath, sep=';')

    data['ecg'] = data['ecg'].str.replace(',', '.').astype(float)
    data['ecg'] = data['ecg'].fillna(0)  # 将 NaN 填充为 0
    data['abp[mmHg]'] = data['abp[mmHg]'].str.replace(',', '.').astype(float)

    # extract ECG signal
    ecg = data['ecg'].values
    ap = data['abp[mmHg]'].values

    return ecg, ap

def loadRtData(filePath):
    filePath = filePath
    data = pd.read_csv(filePath)

    # data['ecg'] = data['ecg'].str.replace(',', '.').astype(float)
    # data['ecg'] = data['ecg'].fillna(0)  # 将 NaN 填充为 0
    # data['abp[mmHg]'] = data['abp[mmHg]'].str.replace(',', '.').astype(float)

    # extract ECG signal
    ecg = data['filtered_ecg'].values
    ap = data['ap'].values
    quality = data['quality'].values


    return ecg, ap, quality

def loadRtDatawithRR(filePath):
    filePath = filePath
    data = pd.read_csv(filePath)

    # data['ecg'] = data['ecg'].str.replace(',', '.').astype(float)
    # data['ecg'] = data['ecg'].fillna(0)  # 将 NaN 填充为 0
    # data['abp[mmHg]'] = data['abp[mmHg]'].str.replace(',', '.').astype(float)

    # extract ECG signal
    ecg = data['filtered_ecg'].values
    ap = data['ap'].values
    quality = data['quality'].values
    rr = data['rr'].values



    return ecg, ap, rr, quality