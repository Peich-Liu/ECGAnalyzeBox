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

