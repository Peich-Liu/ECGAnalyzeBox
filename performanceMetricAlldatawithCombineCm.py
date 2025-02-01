import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from adjustText import adjust_text


folderPath = r"C:\Document\sc2024\result3/"
allAcc = []
allRec = []
allPre = []
allF1 = []

combined_confusion_matrix = np.zeros((2,2), dtype=int)


for fileName in os.listdir(folderPath):
    print(fileName)
    filePath = os.path.join(folderPath, fileName)
    # 假设 CSV 文件命名为 data.csv
    df = pd.read_csv(filePath)
    df = df[df["label"] != 0]
    # print(df)

    y_pred = df["quality"]
    y_true = df["label"]
    y_true = np.where(y_true == 3, 2, y_true)
    y_pred = np.where(y_pred == 3, 2, y_pred)

    # 判断最后两列是否相同
    df["is_same"] = (y_pred == y_true)
    # print(df[["quality", "label", "is_same"]])

    # 计算并打印混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    combined_confusion_matrix += cm
cm = combined_confusion_matrix
print("Confusion Matrix:")
print(cm)

# 提取混淆矩阵中的值
# TP_0, FP_1 = cm[0, 0], cm[0, 1]
# FN_0, TP_1 = cm[1, 0], cm[1, 1]

TP_0 = cm[0, 0]  # true=0,pred=0
FN_0 = cm[0, 1]  # 真实=0,pred=1 对类0来讲是 FN
FP_0 = cm[1, 0]  # 真实=1,pred=0 对类0来讲是 FP
TN_0 = cm[1, 1]

TP_1 = cm[1, 1]  # true=1,pred=1
FN_1 = cm[1, 0]  # 对类1来讲是 FN
FP_1 = cm[0, 1]  # 对类1来讲是 FP
TN_1 = cm[0, 0]

# 计算每个类别的 Precision 和 Recall
# precision_0 = TP_0 / (TP_0 + FN_0)
# precision_1 = TP_1 / (TP_1 + FP_1)

# recall_0 = TP_0 / (TP_0 + FP_1)
# recall_1 = TP_1 / (TP_1 + FN_0)
precision_0 = TP_0 / (TP_0 + FP_0)
recall_0    = TP_0 / (TP_0 + FN_0)

precision_1 = TP_1 / (TP_1 + FP_1)
recall_1    = TP_1 / (TP_1 + FN_1)

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

allAcc.append(accuracy)
allRec.append(weighted_recall)
allPre.append(weighted_precision)
allF1.append(weighted_f1)
print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Weighted Recall: {weighted_recall:.4f}")
print(f"Weighted F1 Score: {weighted_f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# accmean = np.mean(allAcc)
# recmean = np.mean(allRec)
# premean = np.mean(allPre)
# f1mean = np.mean(allF1)


# print(f"Weighted Precision: {premean}")
# print(f"Weighted Recall: {recmean}")
# print(f"Weighted F1 Score: {f1mean}")
# print(f"Accuracy: {accmean}")

# # Remove the 7th and 15th data points
# # indices_to_remove = [7, 15]  # Python uses 0-based indexing
# # allAcc = [v for i, v in enumerate(allAcc) if i not in indices_to_remove]
# # allRec = [v for i, v in enumerate(allRec) if i not in indices_to_remove]
# # allPre = [v for i, v in enumerate(allPre) if i not in indices_to_remove]
# # allF1 = [v for i, v in enumerate(allF1) if i not in indices_to_remove]

# # Prepare line plot
# fig, ax = plt.subplots(figsize=(12, 8))

# # Plot lines for all metrics
# ax.plot(range(len(allAcc)), allAcc, marker='o', label='Accuracy', alpha=0.8)
# ax.plot(range(len(allRec)), allRec, marker='o', label='Recall', alpha=0.8)
# ax.plot(range(len(allPre)), allPre, marker='o', label='Precision', alpha=0.8)
# ax.plot(range(len(allF1)), allF1, marker='o', label='F1 Score', alpha=0.8)

# # Plot mean lines
# ax.axhline(y=accmean, color='blue', linestyle='--', label='Accuracy Mean')
# ax.axhline(y=recmean, color='orange', linestyle='--', label='Recall Mean')
# ax.axhline(y=premean, color='green', linestyle='--', label='Precision Mean')
# ax.axhline(y=f1mean, color='red', linestyle='--', label='F1 Score Mean')

# texts = [
#     ax.text(len(allAcc) - 0.5, accmean, f"Accuracy Mean: {accmean:.2f}", color='blue', fontsize=15),
#     ax.text(len(allAcc) - 0.5, recmean, f"Recall Mean: {recmean:.2f}", color='orange', fontsize=15),
#     ax.text(len(allAcc) - 0.5, premean, f"Precision Mean: {premean:.2f}", color='green', fontsize=15),
#     ax.text(len(allAcc) - 0.5, f1mean, f"F1 Score Mean: {f1mean:.2f}", color='red', fontsize=15),
# ]

# adjust_text(texts)
# # Customize plot
# ax.set_title("Line Plot of Metrics with Mean Lines", fontsize=20)
# ax.set_xlabel("Patient Index", fontsize=20)
# ax.set_ylabel("Metric Value", fontsize=20)
# ax.set_xticks(range(len(allAcc)))  # Ensure all indices are marked on the x-axis
# ax.set_xticklabels(range(len(allAcc)), fontsize=15)
# ax.tick_params(axis='y', labelsize=15) 

# ax.legend(fontsize=15)
# plt.grid(True)

# # Display the plot
# plt.show()