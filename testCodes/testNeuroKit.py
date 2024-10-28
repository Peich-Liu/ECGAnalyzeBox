import neurokit2 as nk

# Download data
data = nk.data("bio_resting_5min_100hz")

# Find peaks
peaks, info = nk.ecg_peaks(data["ECG"], sampling_rate=100)

# Compute HRV indices using method="welch"
hrv_welch = nk.hrv_frequency(peaks, sampling_rate=100, show=True, psd_method="welch")

print(hrv_welch)