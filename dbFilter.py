import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

from utilities import bandpass_filter

filePath = r"/Users/liu/Documents/SC2024fall/250 kun HR.csv"
data = pd.read_csv(filePath, sep=';')

data['ecg'] = data['ecg'].str.replace(',', '.').astype(float)
data['ecg'] = data['ecg'].fillna(0)
ecg = data['ecg'].values

data['abp[mmHg]'] = data['abp[mmHg]'].str.replace(',', '.').astype(float)
data['abp[mmHg]'] = data['abp[mmHg]'].fillna(0)
ap = data['abp[mmHg]'].values

filteredEcg = bandpass_filter(ecg, 0.5, 45, 250)
filteredAp = bandpass_filter(ap, 0.5, 10, 250)

# filteredEcg -= np.mean(filteredEcg)
# filteredAp -= np.mean(filteredAp)

fig, axs = plt.subplots(2, 1)

axs[0].plot(filteredEcg)
axs[1].plot(filteredAp)

plt.show()