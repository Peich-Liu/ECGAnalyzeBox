import os
import wfdb
from utilities import *

patient_id = 'f2o01'

# patient_id = input('input patient id:')
data_directory = r'C:\Document\sc2024\fantasia-database-1.0.0/'  # 替换为你的数据目录
records, annotations = readPatientRecords2(patient_id, data_directory)
start = 0
end = 50000
ecgSignal = concatenateECG(records, start, end)

plt.figure()
plt.plot(ecgSignal)
plt.show()
