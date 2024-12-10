import pandas as pd

# 假设你的CSV文件是data.csv，并且有以下列：
# sample_index, ecg, quality, label(真实值), pred(预测值)
# filePath = r"c:\Document\sc2024\filtered_ecg_with_quality_window_100001.csv"
filePath = r"c:\Document\sc2024\filtered_ecg_with_quality_window_105001.csv"

# filePath = r"c:\Document\sc2024\filtered_ecg_with_quality_window_111001.csv"


df = pd.read_csv(filePath)

# 假设df中真实值列为'label'，预测值列为'pred'
# 如果你的文件中当前只有一个label列，需事先分离出预测值列
# 这里演示的df中假设已经有了pred列。
# 例如，你有真实值列real_label和预测值列pred_label：
# df = pd.read_csv('data.csv')
# df.rename(columns={'label':'real_label', 'some_prediction_column':'pred_label'}, inplace=True)

# 下面以真实值为 real_label 列，预测值为 pred_label 列为例：
real_label_col = 'label'  # 在你的数据中替换成真实值列名
pred_label_col = 'quality'  # 在你的数据中替换成预测值列名，如果没有就添加一个预测列

# 首先需要根据真实label划分连续区间
# 方法：当真实label变化时开始一个新的区间。
df['label_change'] = (df[real_label_col].shift() != df[real_label_col]).astype(int).cumsum()

# 每个连续的label段都有一个唯一的分组ID
# 比如根据'label_change'对df进行groupby即可获得不同区间
groups = df.groupby('label_change')
print(groups)
print(len(groups))

correct_count = 0
total_count = 0

for _, group in groups:
    real_val = group[real_label_col].iloc[0]  # 区间内真实值都是相同的

    if real_val == 2:
        # 如果真实值是2，只要该区间中出现至少一个预测值为2就算正确
        if (group[pred_label_col] == 2).any():
            correct_count += 1
    elif real_val == 1:
        # 如果真实值是1，中间出现零星2也无妨，直接算正确
        correct_count += 1

    total_count += 1

accuracy = correct_count / total_count if total_count > 0 else 0

print("分区级别的准确率：", accuracy)
