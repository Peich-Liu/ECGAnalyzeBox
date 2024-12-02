# import wfdb
# import pandas as pd
# import matplotlib.pyplot as plt
# import csv
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from collections import deque
# from scipy import signal
# from scipy.stats import kurtosis as calc_kurtosis, skew as calc_skew
# import csv

# from utilities import signalQualityEva, fixThreshold, dynamicThreshold, bandPass, filter2Sos, ziFilter
# from scipy import signal

# # Function to read quality annotations
# def read_annotations(file_path):
#     return pd.read_csv(file_path, header=None, names=['start', 'end', 'quality'])

# # Main function
# def main():
#     resultFile =  r"C:\Document\sc2024/filtered_ecg_with_qualitynew.csv"
#     annotations_file_path = r'c:\Document\sc2024\data\testData\100001_ANN.csv'
    
#     # 加载数据
#     ann_df = pd.read_csv(annotations_file_path)
#     filtered_df = pd.read_csv(resultFile)


#     # 使用第10-12列提取注释片段
#     # 假设：第10列是起始样本，第11列是结束样本，第12列是质量等级
#     start_col = ann_df.columns[9]  # 第10列
#     end_col = ann_df.columns[10]   # 第11列
#     quality_col = ann_df.columns[11]  # 第12列

#     # 提取质量等级为1的片段
#     high_quality_segments = ann_df[ann_df[quality_col] == 1][[start_col, end_col]].values.tolist()

#     # 将filtered文件中`good`标记的片段提取
#     filtered_good_segments = filtered_df[filtered_df['quality'] == 'good'][['start_sample', 'end_sample']].values.tolist()

#     # 定义函数计算片段间的重叠
#     def calculate_overlap(segment1, segment2):
#         start = max(segment1[0], segment2[0])
#         end = min(segment1[1], segment2[1])
#         return max(0, end - start) > 0  # 是否有重叠

#     # 匹配高质量标注与预测
#     TP, FP, FN, TN = 0, 0, 0, 0
#     for segment in filtered_good_segments:
#         match = any(calculate_overlap(segment, ref) for ref in high_quality_segments)
#         if match:
#             TP += 1
#         else:
#             FP += 1

#     for segment in high_quality_segments:
#         match = any(calculate_overlap(segment, pred) for pred in filtered_good_segments)
#         if not match:
#             FN += 1

#     # 总样本数量
#     total_segments = len(filtered_df)

#     # 计算准确率、灵敏度和特异性
#     accuracy = (TP + TN) / total_segments
#     sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
#     specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

#     # 输出结果
#     print(f"Accuracy: {accuracy:.2f}")
#     print(f"Sensitivity: {sensitivity:.2f}")
#     print(f"Specificity: {specificity:.2f}")
    
# if __name__ == '__main__':
#     main()



import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

resultFile =  r"c:\Document\sc2024\filtered_ecg_with_qualitynewwithoutFilter.csv"
annotations_file_path = r'c:\Document\sc2024\data\testData\100001_ANN.csv'

# 读取 ANN 文件和 filtered 文件
ann_df = pd.read_csv(annotations_file_path)
ann_df = ann_df[0:20]
filtered_df = pd.read_csv(resultFile)
filtered_df = filtered_df[0:500000]

# 初始化真实标签和预测标签数组
signal_length = len(filtered_df['ecg'])
true_labels = np.zeros(signal_length)
pred_labels = np.zeros(signal_length)

# 生成真实标签
for index, row in ann_df.iterrows():
    start = int(row[9])
    end = int(row[10])
    quality = int(row[11])
    print("start, end, quality", start, end, quality)
    if quality == 1:
        true_labels[start:end] = 1  # 好
    elif quality == 2:
        true_labels[start:end] = 0  # 坏
print(len(true_labels))

# 生成预测标签
for index, row in filtered_df.iterrows():
    sample = row['sample_index']
    quality = row['quality']

    if quality == 'good':
        pred_labels[sample] = 1  # 好
    else:
        pred_labels[sample] = 0  # 坏
print(len(pred_labels))


# 计算混淆矩阵
cm = confusion_matrix(true_labels, pred_labels, labels=[1, 0])
print('混淆矩阵：\n', cm)

# 输出性能指标
TP, FP, FN, TN = cm.ravel()
accuracy = (TP + TN) / (TP + FP + FN + TN)
print('准确率：', accuracy)

