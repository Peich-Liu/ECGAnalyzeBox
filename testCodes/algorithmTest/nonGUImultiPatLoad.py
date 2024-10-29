import os
import wfdb
from utilities import *

patient_id = 'f2o01'

# patient_id = input('input patient id:')
data_directory = r'C:\Document\sc2024\fantasia-database-1.0.0/'  # 替换为你的数据目录
records, annotations = readPatientRecords2(patient_id, data_directory)
start = 0
end = 50000

if records:
    ecgSignal = concatenateECG(records, start, end)
    apSignal = concatenateAP(records, start, end)
    # concatenated_signal = concatenateSignals(records, start, end)

    # print(len(concatenated_signal))
    visualizeSignal(ecgSignal)
    visualizeSignal(apSignal)

else:
    print("fail")


# def loadAll(data_dir):
#     # patient_prefix = patient_id[:5]
#     records = []
#     annotations = []

#     for file in os.listdir(data_dir):
#         file_base = os.path.splitext(file)[0]
#         if file.endswith('.dat'):
#             record_base = os.path.splitext(file)[0]
#             load_file = os.path.join(data_dir, record_base)
#             print("loading file", load_file)
#             record = wfdb.rdrecord(load_file)
#             #here assume all the fs is same in the same signal
#             records.append(record)
#             print(record.sig_name)


# # patient_id = input('input patient id:')
# data_directory = r'C:\Document\sc2024\fantasia-database-1.0.0/'  # 替换为你的数据目录
# records, annotations = loadAll(data_directory)
# start = 0
# end = 50000

# if records:
#     concatenated_signal = concatenateandProcessSignals2(records, start, end)
#     # concatenated_signal = concatenateSignals(records, start, end)

#     # print(len(concatenated_signal))
#     visualizeSignal(concatenated_signal)
# else:
#     print("fail")