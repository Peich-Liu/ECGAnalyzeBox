import numpy as np

conf_matrix =np.array([
[56301893,  3474124],
 [23235107,  4075876]])
conf_matrix =np.array(
[[27646,  1490,     2],
 [15853,   809,    99],
 [    5,     6, 23664]])

conf_matrix =np.array(
    [[15012,    22],
 [ 1403,   120]])

conf_matrix =np.array(
[[610,   169],
[127,   294]]
)





# 初始化存储结果
sensitivity = []
precision = []
f1_score = []

# 逐类别计算指标
for i in range(len(conf_matrix)):
    TP = conf_matrix[i, i]  # 对角线值为 TP
    FN = sum(conf_matrix[i, :]) - TP  # 行和减去 TP
    FP = sum(conf_matrix[:, i]) - TP  # 列和减去 TP
    TN = conf_matrix.sum() - (TP + FN + FP)  # 剩下的全是 TN

    # 计算 sensitivity, precision
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    prec = TP / (TP + FP) if (TP + FP) != 0 else 0

    # 计算 F1-score
    f1 = 2 * (prec * recall) / (prec + recall) if (prec + recall) != 0 else 0

    # 保存结果
    sensitivity.append(recall)
    precision.append(prec)
    f1_score.append(f1)

# 打印结果
print("Sensitivity (Recall):", sensitivity)
print("Precision:", precision)
print("F1-Score:", f1_score)