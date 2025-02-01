import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
# filePath = r"c:\Document\sc2024\filtered_ecg_with_quality_new_100001.csv"
# filePath = r"C:\Document\sc2024\filtered_ecg_with_quality_new_105001.csv"

# filePath = r"c:\Document\sc2024\filtered_ecg_with_quality_window_105001.csv"
# filePath = r"c:\Document\sc2024\filtered_ecg_with_quality_window_100001.csv"
# filePath = r"c:\Document\sc2024\filtered_ecg_with_quality_window_100002.csv"
filePath = r"c:\Document\sc2024\result3\filtered_ecg_with_quality_window_124001.csv"

# filePath = r"c:\Document\sc2024\filtered_ecg_with_quality_window_111001.csv"



# 假设 CSV 文件命名为 data.csv
df = pd.read_csv(filePath)
# 筛选出真实值和预测值中不为0的行
df = df[df["label"] != 0]

print(df)
# exit()
# 假设quality列为预测值，label列为真实值
y_pred = df["quality"]
y_true = df["label"]

y_true = np.where(y_true == 3, 2, y_true)
y_pred = np.where(y_pred == 3, 2, y_pred)


# mask = (y_true != 0) & (y_pred != 0)
# y_true = y_true[mask]
# y_pred = y_pred[mask]

# 判断最后两列是否相同
df["is_same"] = (y_pred == y_true)
print(df[["quality", "label", "is_same"]])

# 计算并打印混淆矩阵
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

import numpy as np

# 输入混淆矩阵
# confusion_matrix = np.array([[27646, 1492],
#                              [15858, 24578]])


# 提取混淆矩阵中的值
TP_0, FP_1 = cm[0, 0], cm[0, 1]
FN_0, TP_1 = cm[1, 0], cm[1, 1]

# 计算每个类别的 Precision 和 Recall
precision_0 = TP_0 / (TP_0 + FN_0)
precision_1 = TP_1 / (TP_1 + FP_1)

recall_0 = TP_0 / (TP_0 + FP_1)
recall_1 = TP_1 / (TP_1 + FN_0)

# 样本数
samples_0 = TP_0 + FN_0
samples_1 = TP_1 + FP_1
total_samples = samples_0 + samples_1

# 加权精确率
weighted_precision = (samples_0 * precision_0 + samples_1 * precision_1) / total_samples

# 加权召回率
weighted_recall = (samples_0 * recall_0 + samples_1 * recall_1) / total_samples

# 计算 F1 分数
f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0)
f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1)

# 加权 F1 分数
weighted_f1 = (samples_0 * f1_0 + samples_1 * f1_1) / total_samples

# 准确率
accuracy = (TP_0 + TP_1) / total_samples

# 输出结果
print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Weighted Recall: {weighted_recall:.4f}")
print(f"Weighted F1 Score: {weighted_f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

