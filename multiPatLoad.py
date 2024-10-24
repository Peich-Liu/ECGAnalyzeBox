import os
import wfdb
from methodLibrary import *

patient_id = '10'
# patient_id = input('input patient id:')
data_directory = r'C:\Document\sc2024\mit-bih-arrhythmia-database-1.0.0/'  # 替换为你的数据目录
records, annotations = readPatientRecords(patient_id, data_directory)
start = 0
end = 5000
if records:
    concatenated_signal = concatenateSignals(records, start, end)
    # print(len(concatenated_signal))
    visualizeSignal(concatenated_signal)
else:
    print("fail")

