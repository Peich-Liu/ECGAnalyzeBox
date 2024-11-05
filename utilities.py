import numpy as np
import pandas as pd 
import scipy.signal as signal
import wfdb
import os
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import welch
from scipy.integrate import simpson
from statsmodels.tsa.ar_model import AutoReg
import neurokit2 as nk
from scipy.signal import butter, filtfilt
import numpy as np
import os
import wfdb
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Rectangle

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
def calculateEcgSignalRangeProperties(signal, fs, selected_ranges):
    print("selected_ranges",selected_ranges)
    results = {}

    for index, (start, end) in selected_ranges.items():
        segment = signal[start:end]

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

def calculateApSignalRangeProperties(signal, fs, selected_ranges):
    """
    计算选定范围内的 AP 信号属性。
    
    参数:
    - signal: 完整的 AP 信号数组
    - fs: 采样频率
    - selected_ranges: 字典，其中每个键是选框的索引，每个值是选框的 (start, end) 范围
    
    返回:
    - results: 包含每个框选区域的计算属性的字典
    """
    print("selected_ranges",selected_ranges)
    results = {}

    for index, (start, end) in selected_ranges.items():
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
def calculate_mean(ranges):
    print("计算平均值:", ranges)
    # 执行计算逻辑...

def calculate_variance(ranges):
    print("计算方差:", ranges)